"""Application settings.

Everything about the application rather than the model: appearance, agent
behaviour, where the endpoint is and where files are looked for. Model settings —
sampling, reasoning, load-time flags — live in the Params panel, next to the
model they apply to.
"""
from __future__ import annotations

from pathlib import Path

import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox,
                               QMessageBox,
                               QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QPlainTextEdit, QPushButton,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from .. import speech as speechmod
from . import theme
from .. import update
from ..cluster import find_rpc_binary, find_server_binary
from ..config import Config, default_skill_dirs
from .widgets import Field, swatch

LEGACY_RASTER = {"8514oem", "courier", "fixedsys", "modern", "ms sans serif",
                 "ms serif", "roman", "script", "small fonts", "system", "terminal"}


def _usable_font(family: str) -> bool:
    """Windows ships bitmap-only faces with no outlines for Qt to instantiate;
    listing them logs a DirectWrite failure each and yields nothing usable."""
    return family.lower() not in LEGACY_RASTER and not family.startswith("@")


def _row(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for x in widgets:
        lay.addWidget(x)
    return w


def _swatch(colour: str):
    return swatch(colour)


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Section")
    return label


class SettingsDialog(QDialog):
    _update_result = Signal(str, bool)
    _update_line = Signal(str)
    _update_done = Signal(bool)

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self._update_result.connect(self._show_update)
        self._update_line.connect(lambda line: self.update_log.appendPlainText(line))
        self._update_done.connect(self._update_finished)
        self.cfg = cfg
        self.appearance_changed = False
        self._notes: list[str] = []
        self.setWindowTitle("Settings")
        self.setMinimumSize(620, 520)

        tabs = QTabWidget()
        tabs.addTab(self._appearance_tab(), "Appearance")
        tabs.addTab(self._agent_tab(), "Agent")
        tabs.addTab(self._endpoint_tab(), "Endpoint")
        tabs.addTab(self._folders_tab(), "Folders")
        tabs.addTab(self._updates_tab(), "Updates")
        tabs.addTab(self._about_tab(), "About")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(buttons)

    def _note(self, text: str) -> None:
        self._notes.append(text)

    def _agent_tab(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        c = self.cfg

        self.workspace = QLineEdit(c.workspace)
        pick = QPushButton("…")
        pick.setMaximumWidth(34)
        pick.clicked.connect(self._pick_workspace)
        lay.addWidget(Field("Workspace", _row(self.workspace, pick),
                            "File tools are confined here; shell commands run here."))

        self.dialect = QComboBox()
        self.dialect.addItems(["auto", "native", "text"])
        self.dialect.setCurrentText(c.tool_dialect)
        self.profile = QComboBox()
        self.profile.addItems(["auto", "nano", "small", "standard", "large"])
        self.profile.setCurrentText(c.profile_override or "auto")
        lay.addWidget(Field("Tool dialect / context profile",
                            _row(self.dialect, self.profile),
                            "auto uses the compact text protocol below 16k, or when "
                            "the model's template cannot do tool calls."))

        self.approval = QComboBox()
        self.approval.addItems(["always", "safe", "never"])
        self.approval.setCurrentText(c.approval)
        lay.addWidget(Field("Ask before running", self.approval,
                            "safe asks only for writes and shell commands."))

        self.steps = QSpinBox()
        self.steps.setRange(1, 200)
        self.steps.setValue(c.max_steps)
        lay.addWidget(Field("Max steps per task", self.steps))

        self.auto_plan = QCheckBox("Draft a checklist before multi-step work")
        self.auto_plan.setChecked(c.auto_plan)
        lay.addWidget(self.auto_plan)
        self.autostart = QCheckBox("Start llama-server when Kestrel opens")
        self.autostart.setChecked(c.auto_start_server)
        lay.addWidget(self.autostart)
        self.tool_detail = QCheckBox("Show tool arguments and raw output inline")
        self.tool_detail.setChecked(c.show_tool_detail)
        lay.addWidget(self.tool_detail)
        self.watch_skills = QCheckBox("Pick up new skills automatically")
        self.watch_skills.setChecked(c.watch_skills)
        lay.addWidget(self.watch_skills)
        self.plan_driven = QCheckBox("Keep working until the checklist is closed")
        self.plan_driven.setToolTip("A prose reply with steps still open is "
                                    "treated as a status update, not an answer")
        self.plan_driven.setChecked(c.plan_driven)
        lay.addWidget(self.plan_driven)
        self.bell = QCheckBox("Chime when a task finishes")
        self.bell.setChecked(c.bell_on_finish)
        lay.addWidget(self.bell)

        self.bell_sound = QComboBox()
        for label, path in speechmod.sound_choices(c.bell_sound):
            self.bell_sound.addItem(label, path)
        index = self.bell_sound.findData(c.bell_sound)
        self.bell_sound.setCurrentIndex(max(0, index))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_sound)
        listen = QPushButton("Play")
        listen.clicked.connect(self._play_sound)
        lay.addWidget(Field("Sound", _row(self.bell_sound, browse, listen),
                            "wav, mp3, flac or ogg. Anything but wav needs "
                            "sounddevice and soundfile, or ffmpeg."))

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply_agent)
        lay.addWidget(apply_btn)
        lay.addStretch(1)
        return inner

    def _pick_workspace(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Workspace",
                                                  self.workspace.text())
        if chosen:
            self.workspace.setText(chosen)

    def apply_agent(self) -> None:
        c = self.cfg
        c.workspace = self.workspace.text().strip() or c.workspace
        c.tool_dialect = self.dialect.currentText()
        chosen = self.profile.currentText()
        c.profile_override = "" if chosen == "auto" else chosen
        c.approval = self.approval.currentText()
        c.max_steps = self.steps.value()
        c.auto_plan = self.auto_plan.isChecked()
        c.auto_start_server = self.autostart.isChecked()
        c.show_tool_detail = self.tool_detail.isChecked()
        c.watch_skills = self.watch_skills.isChecked()
        c.plan_driven = self.plan_driven.isChecked()
        c.bell_on_finish = self.bell.isChecked()
        c.bell_sound = self.bell_sound.currentData() or ""
        c.save()
        self._note("Agent settings saved")

    def _appearance_tab(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        note = QLabel("Characters shown as empty boxes mean the chosen font has no "
                      "glyph for them. Faces that cannot be rendered at all are "
                      "left out of these lists.")
        note.setWordWrap(True)
        note.setObjectName("Dim")
        lay.addWidget(note)

        lay.addWidget(_heading("Colour"))

        self.tint = QComboBox()
        for key, (label, hue) in theme.TINTS.items():
            self.tint.addItem(_swatch(hue or theme.PALETTES[self.cfg.theme]["PANEL"]),
                              label, key)
        self.tint.setCurrentIndex(max(0, self.tint.findData(self.cfg.ui_tint)))
        self.tint.currentIndexChanged.connect(self._preview_colours)
        lay.addWidget(Field("Interface tint", self.tint,
                            "Shifts the surfaces towards a hue. Text and the "
                            "status colours are left alone."))

        self.accent = QComboBox()
        for key, (label, dark, light) in theme.ACCENTS.items():
            self.accent.addItem(_swatch(dark if self.cfg.theme == "dark" else light),
                                label, key)
        self.accent.setCurrentIndex(max(0, self.accent.findData(self.cfg.ui_accent)))
        self.accent.currentIndexChanged.connect(self._preview_colours)
        lay.addWidget(Field("Buttons and highlights", self.accent,
                            "Each is adjusted until its label can be read on "
                            "it, so none of them come out illegible."))

        self.colour_preview = QLabel("")
        self.colour_preview.setMinimumHeight(46)
        self.colour_preview.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.colour_preview)
        self._preview_colours()

        lay.addWidget(_heading("Type"))
        families = [f for f in QFontDatabase.families() if _usable_font(f)]
        fixed = [f for f in families if QFontDatabase.isFixedPitch(f)]

        self.ui_font = QComboBox()
        self.ui_font.addItem("Platform default", "")
        for family in families:
            self.ui_font.addItem(family, family)
        self.ui_font.setCurrentIndex(max(0, self.ui_font.findData(self.cfg.ui_font)))
        lay.addWidget(Field(f"Interface font  ({len(families)})", self.ui_font))

        self.mono_font_box = QComboBox()
        self.mono_font_box.addItem("Platform default", "")
        for family in (fixed or families):
            self.mono_font_box.addItem(family, family)
        self.mono_font_box.setCurrentIndex(
            max(0, self.mono_font_box.findData(self.cfg.mono_font)))
        lay.addWidget(Field(f"Monospace font  ({len(fixed)} fixed-pitch)",
                            self.mono_font_box))

        self.font_size = QSpinBox()
        self.font_size.setRange(9, 20)
        self.font_size.setValue(self.cfg.font_size)
        self.theme_box = QComboBox()
        self.theme_box.addItems(["dark", "light"])
        self.theme_box.setCurrentText(self.cfg.theme)
        lay.addWidget(Field("Base size / palette", _row(self.font_size, self.theme_box)))

        self.preview = QLabel("Sample  0123  [You] [Kestrel] [Thinking]  {}[]|\\")
        self.preview.setObjectName("Readout")
        lay.addWidget(Field("preview", self.preview))
        self.mono_font_box.currentIndexChanged.connect(self._preview_font)
        self._preview_font()

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply_appearance)
        lay.addWidget(apply_btn)
        lay.addStretch(1)
        return inner

    def _preview_font(self) -> None:
        family = self.mono_font_box.currentData() or ""
        font = QFont(family) if family else QFont()
        font.setPointSize(11)
        self.preview.setFont(font)

    def apply_appearance(self) -> None:
        self.cfg.ui_font = self.ui_font.currentData() or ""
        self.cfg.mono_font = self.mono_font_box.currentData() or ""
        self.cfg.font_size = self.font_size.value()
        self.cfg.theme = self.theme_box.currentText()
        self.cfg.ui_tint = self.tint.currentData() or "slate"
        self.cfg.ui_accent = self.accent.currentData() or "amber"
        self.cfg.save()
        self.appearance_changed = True
        self._note("Appearance applied")


    def _endpoint_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.cfg

        self.url = QLineEdit(c.server_url)
        lay.addWidget(Field("Server URL", self.url,
                            "Any OpenAI-compatible endpoint: llama-server, LM Studio, "
                            "llamafile, vLLM, or Ollama's /v1."))
        self.api_key = QLineEdit(c.api_key)
        self.api_key.setEchoMode(QLineEdit.Password)
        self.hf_token = QLineEdit(c.hf_token)
        self.hf_token.setEchoMode(QLineEdit.Password)
        lay.addWidget(Field("API key / Hugging Face token",
                            _row(self.api_key, self.hf_token),
                            "Blank for a local server. The token is only needed for "
                            "gated model repositories."))

        self.server_bin = QLineEdit(c.llama_server_bin or find_server_binary())
        pick = QPushButton("Browse…")
        pick.clicked.connect(lambda: self._pick_file(self.server_bin, "llama-server"))
        lay.addWidget(Field("llama-server binary", _row(self.server_bin, pick)))

        self.rpc_bin = QLineEdit(c.rpc_bin or find_rpc_binary())
        pick2 = QPushButton("Browse…")
        pick2.clicked.connect(lambda: self._pick_file(self.rpc_bin, "rpc-server"))
        lay.addWidget(Field("rpc-server binary (cluster worker)",
                            _row(self.rpc_bin, pick2)))

        self.model_path = QLineEdit(c.model_path)
        pick3 = QPushButton("Browse…")
        pick3.clicked.connect(lambda: self._pick_file(
            self.model_path, "GGUF model", "GGUF models (*.gguf);;All files (*)"))
        lay.addWidget(Field("Model file", _row(self.model_path, pick3),
                            "Or choose one from the Models panel, which shows what "
                            "each file contains before loading it."))
        lay.addStretch(1)
        return w

    def _folders_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Folders scanned for SKILL.md, including other "
                             "harnesses'. New skills are picked up automatically."))
        self.dirs = QListWidget()
        self.dirs.addItems(self.cfg.skills_dirs)
        lay.addWidget(self.dirs, 1)
        add = QPushButton("Add…")
        add.clicked.connect(self._add_dir)
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda: self.dirs.takeItem(self.dirs.currentRow()))
        defaults = QPushButton("Restore defaults")
        defaults.clicked.connect(self._restore_dirs)
        lay.addWidget(_row(add, remove, defaults))

        self.model_dirs = QListWidget()
        self.model_dirs.addItems(self.cfg.model_dirs)
        lay.addWidget(QLabel("Folders scanned for GGUF models."))
        lay.addWidget(self.model_dirs, 1)
        add2 = QPushButton("Add…")
        add2.clicked.connect(self._add_model_dir)
        remove2 = QPushButton("Remove")
        remove2.clicked.connect(
            lambda: self.model_dirs.takeItem(self.model_dirs.currentRow()))
        lay.addWidget(_row(add2, remove2))
        return w

    FORUM = "https://alternitech.freeforums.net/"
    DONATE = "https://alternitech.square.site/product/donation/6"

    def _about_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        title = QLabel("Kestrel")
        title.setObjectName("Section")
        lay.addWidget(title)

        version = QLabel(f"Version {update.local_version()}")
        version.setObjectName("Readout")
        lay.addWidget(version)

        blurb = QLabel(
            "An agentic harness for llama.cpp: a local model that plans a task, "
            "works through it with tools, and keeps a checklist you can read and "
            "edit while it goes.\n\n"
            "It is built to do real work inside a small context window rather "
            "than assume a large one, so it runs on the hardware people actually "
            "have. Nothing is sent anywhere unless you point it at a remote "
            "endpoint yourself — the model, the files, the memory and the "
            "conversations all stay on this machine.")
        blurb.setWordWrap(True)
        lay.addWidget(blurb)

        support = QLabel(
            "Questions, problems and suggestions are welcome on the forum:<br>"
            f'<a href="{self.FORUM}" style="color:#A0670F;">{self.FORUM}</a>')
        support.setOpenExternalLinks(True)
        support.setWordWrap(True)
        lay.addWidget(support)

        thanks = QLabel(
            "Kestrel is free. If it saves you time and you would like to put "
            "something back towards its development, there is a donation page.")
        thanks.setWordWrap(True)
        thanks.setObjectName("Dim")
        lay.addWidget(thanks)

        donate = QPushButton("Donate")
        donate.setObjectName("Primary")
        donate.setToolTip(self.DONATE)
        donate.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(self.DONATE)))
        forum = QPushButton("Open the forum")
        forum.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(self.FORUM)))
        lay.addWidget(_row(donate, forum))

        lay.addStretch(1)
        credit = QLabel("Built on llama.cpp, PySide6, Piper and whisper.cpp — "
                        "each doing the hard part.")
        credit.setWordWrap(True)
        credit.setObjectName("Dim")
        lay.addWidget(credit)
        return w

    def _updates_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        self.version_label = QLabel(f"Installed version {update.local_version()}")
        self.version_label.setObjectName("Readout")
        lay.addWidget(self.version_label)

        self.update_status = QLabel("")
        self.update_status.setWordWrap(True)
        lay.addWidget(self.update_status)

        check = QPushButton("Check for updates")
        check.clicked.connect(self.check_updates)
        lay.addWidget(check)

        self.install_btn = QPushButton("Install the update")
        self.install_btn.setObjectName("Primary")
        self.install_btn.clicked.connect(self.install_update)
        self.install_btn.hide()
        lay.addWidget(self.install_btn)

        self.update_log = QPlainTextEdit()
        self.update_log.setReadOnly(True)
        self.update_log.setMaximumHeight(120)
        self.update_log.hide()
        lay.addWidget(self.update_log)

        self.repo_link = QLabel(
            f'<a href="{update.RELEASES}" style="color:#E8A33D;">{update.RELEASES}</a>')
        self.repo_link.setOpenExternalLinks(True)
        self.repo_link.setWordWrap(True)
        lay.addWidget(self.repo_link)

        note = QLabel("Updating happens here rather than on a web page. A git "
                      "checkout is pulled; any other copy is replaced from the "
                      "published archive, keeping a backup of what was there.\n\n"
                      "Settings, memory, conversations and projects live outside "
                      "the program folder and are never touched.")
        note.setWordWrap(True)
        note.setObjectName("Dim")
        lay.addWidget(note)
        lay.addStretch(1)
        return w

    def check_updates(self) -> None:
        self.update_status.setText("Checking…")

        def work():
            result = update.check()
            self._update_result.emit(result.summary(), result.available)

        threading.Thread(target=work, daemon=True).start()

    def _show_update(self, summary: str, available: bool) -> None:
        self.update_status.setText(summary)
        colour = "#E8A33D" if available else ""
        self.update_status.setStyleSheet(f"color: {colour};" if colour else "")
        self.install_btn.setVisible(available)
        if available:
            self.install_btn.setText(
                "Install the update" + ("  (git pull)" if update.is_git_checkout()
                                        else ""))

    def install_update(self) -> None:
        if QMessageBox.question(
                self, "Install the update",
                "Replace this installation with the published version?\n\n"
                "Settings, memory, conversations and projects are kept — they "
                "live outside the program folder.\n\n"
                "Kestrel must be restarted afterwards.") != QMessageBox.Yes:
            return
        self.install_btn.setEnabled(False)
        self.update_log.clear()
        self.update_log.show()

        def say(line: str) -> None:
            self._update_line.emit(line)

        def work():
            try:
                ok, message = update.apply(say)
            except Exception as e:
                ok, message = False, str(e)
            self._update_result.emit(message, False)
            self._update_line.emit(message)
            self._update_done.emit(ok)

        threading.Thread(target=work, daemon=True).start()

    def _update_finished(self, ok: bool) -> None:
        self.install_btn.setEnabled(True)
        self.install_btn.setVisible(not ok)
        if not ok:
            return
        self.version_label.setText(f"Installed version {update.local_version()}")
        box = QMessageBox(self)
        box.setWindowTitle("Update installed")
        box.setText("Kestrel has been updated.")
        box.setInformativeText(
            "The new version runs from the next start. Restarting now closes "
            "this window and opens it again.")
        now = box.addButton("Restart now", QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        box.setDefaultButton(now)
        box.exec()
        if box.clickedButton() is now:
            self.restart_now()

    def restart_now(self) -> None:
        """Relaunch, then close this one — in that order, so a failure to start
        the new copy leaves the old one running rather than nothing at all."""
        if not update.restart():
            QMessageBox.warning(self, "Could not restart",
                                "Start Kestrel again yourself to pick up the "
                                "new version.")
            return
        window = self.window()
        self.accept()
        QApplication.instance().quit()
        _ = window

    # -- helpers --------------------------------------------------------------
    def _preview_colours(self) -> None:
        """Show the chosen pair before committing to them."""
        tint = self.tint.currentData() or "slate"
        accent = self.accent.currentData() or "amber"
        palette = theme.build_palette(self.cfg.theme, tint, accent)
        self.colour_preview.setText(
            f"  {theme.TINTS[tint][0]} · {theme.ACCENTS[accent][0]}  ")
        self.colour_preview.setStyleSheet(
            f"background: {palette['PANEL']};"
            f"color: {palette['TEXT']};"
            f"border: 1px solid {palette['LINE']};"
            f"border-left: 8px solid {palette['AMBER']};"
            "border-radius: 6px; padding: 8px;")

    def _pick_sound(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chime sound", str(Path.home()),
            "Sound files (*.wav *.mp3 *.flac *.ogg *.m4a *.aiff);;All files (*)")
        if not path:
            return
        self.bell_sound.insertItem(1, Path(path).name, path)
        self.bell_sound.setCurrentIndex(1)

    def _play_sound(self) -> None:
        chosen = self.bell_sound.currentData() or ""
        target = Path(chosen) if chosen else (
            Path(__file__).resolve().parent.parent.parent / "assets" / "bell.wav")
        if not target.exists():
            QMessageBox.information(self, "Not found", f"{target} is not there.")
            return

        def work():
            try:
                speechmod.play(target, blocking=False)
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    def _pick_file(self, target: QLineEdit, title: str,
                   filt: str = "All files (*)") -> None:
        start = str(Path(target.text()).parent) if target.text() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, title, start, filt)
        if path:
            target.setText(path)

    def _add_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Skills folder", str(Path.home()))
        if path:
            self.dirs.addItem(path)

    def _add_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Models folder", str(Path.home()))
        if path:
            self.model_dirs.addItem(path)

    def _restore_dirs(self) -> None:
        self.dirs.clear()
        self.dirs.addItems(default_skill_dirs())

    # -- commit ---------------------------------------------------------------
    def apply(self) -> Config:
        c = self.cfg
        self.apply_agent()
        self.apply_appearance()
        c.server_url = self.url.text().strip() or c.server_url
        c.api_key = self.api_key.text()
        c.hf_token = self.hf_token.text()
        c.llama_server_bin = self.server_bin.text().strip()
        c.rpc_bin = self.rpc_bin.text().strip()
        c.model_path = self.model_path.text().strip()
        c.skills_dirs = [self.dirs.item(i).text() for i in range(self.dirs.count())]
        c.model_dirs = [self.model_dirs.item(i).text()
                        for i in range(self.model_dirs.count())]
        c.save()
        return c
