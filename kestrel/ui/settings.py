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
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QMessageBox,
                               QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QPlainTextEdit, QPushButton,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from .. import update
from ..cluster import find_rpc_binary, find_server_binary
from ..config import Config, default_skill_dirs
from .widgets import Field

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
        if ok:
            QMessageBox.information(self, "Update installed",
                                    "Restart Kestrel to run the new version.")

    # -- helpers --------------------------------------------------------------
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
