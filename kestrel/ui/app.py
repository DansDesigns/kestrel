"""The Kestrel window."""
from __future__ import annotations

import atexit
import subprocess
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import (QFileSystemWatcher, QObject, QSize, QThread, Qt,
                            QTimer, Signal, Slot)
from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QTextOption
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QScrollArea, QSizePolicy,
                               QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QInputDialog, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox,
                               QSplitter, QTabWidget, QTextEdit, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import cluster as clustermod
from .. import attach as attachmod
from .. import sysmon
from .. import downloads as dlmod
from .. import sessions as sessionmod
from .. import speech as speechmod
from .. import skills as skillmod
from ..agent import Agent, strip_calls
from ..config import Config, Node
from ..llm import LlamaClient, LLMError
from . import theme
from .panels import (BackendPanel, CanvasPanel, MemoryPanel, ModelsPanel,
                     ParamsPanel,
                     PersonaPanel, PlanPanel, ProjectsPanel, SpeechPanel,
                     SystemPanel, ToolsPanel, UiThread, _row)
from .downloads_window import DownloadsWindow
from .settings import SettingsDialog
from .splash import Splash
from .widgets import (ActivityTree, ChatView, ContextGauge,
                      BusyOverlay, Field, IconTabBar, LazyTab, Readout,
                      MonitorStrip, TypingIndicator, clear_font_cache,
                      install_wheel_guard, mono_font, stretch_columns)


# ------------------------------------------------------------------ worker --
class AgentWorker(QObject):
    ready = Signal(object)
    failed = Signal(str)
    token = Signal(str)
    toolCall = Signal(str, object)
    toolResult = Signal(str, bool, str)
    assistantDone = Signal(str)
    contextUpdate = Signal(object, object, int)
    genStats = Signal(float, int)
    stepped = Signal(int, int)
    approvalNeeded = Signal(str, str)
    thinking = Signal(str)
    thinkingDone = Signal(str, int)
    memoryRecall = Signal(object)
    memorySaved = Signal(object)
    todoUpdate = Signal(object)
    turnFinished = Signal()
    pausedChanged = Signal(bool)

    def __init__(self, cfg: Config, progress=None):
        super().__init__()
        self.progress = progress or (lambda message: None)
        self.cfg = cfg
        self.agent: Agent | None = None
        self._gate = threading.Event()
        self._granted = False

    # -- called on the worker thread -----------------------------------------
    @Slot()
    def prepare(self) -> None:
        try:
            client = LlamaClient(self.cfg.server_url, self.cfg.api_key)
            agent = Agent(self.cfg, client, emit=self._emit)
            agent.approver = self._ask
            info = agent.prepare()
            self.agent = agent
            self.ready.emit(info)
        except LLMError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    @Slot(str)
    def send(self, text: str) -> None:
        if self.agent is None:
            # Connection is staged and may not have finished. Typing before it
            # does should mean waiting a moment, not losing the message.
            self.prepare()
        if self.agent is None:
            self.failed.emit("Not connected — check the endpoint under Settings.")
            self.turnFinished.emit()
            return
        try:
            self.agent.run(text)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self.turnFinished.emit()

    @Slot()
    def reset(self) -> None:
        if self.agent:
            self.agent.reset()

    @Slot(str)
    def name_session(self, label: str) -> None:
        if self.agent:
            self.agent.name_session(label)

    @Slot(str)
    def rebind(self, workspace: str) -> None:
        if self.agent:
            self.agent.rebind(workspace)

    @Slot()
    def forget_project(self) -> None:
        if self.agent:
            count = self.agent.forget_project_memories()
            self.statusLine.emit(f"Cleared {count} project memory(ies)")

    def set_paused(self, paused: bool) -> None:
        if self.agent:
            self.agent.pause() if paused else self.agent.resume()

    def cancel(self) -> None:
        if self.agent:
            self.agent.cancel()
        self._granted = False
        self._gate.set()

    # -- bridges --------------------------------------------------------------
    def _emit(self, kind: str, data: dict) -> None:
        if kind == "token":
            self.token.emit(data["text"])
        elif kind == "tool_call":
            self.toolCall.emit(data["name"], data["args"])
        elif kind == "tool_result":
            self.toolResult.emit(data["name"], data["ok"], data["shown"])
        elif kind == "assistant":
            self.assistantDone.emit(data["text"])
        elif kind == "context":
            self.contextUpdate.emit(data["usage"], data["budget"], data["compactions"])
        elif kind == "gen":
            self.genStats.emit(data["tps"], data["tokens"])
        elif kind == "step":
            self.stepped.emit(data["step"], data["max"])
        elif kind == "thinking":
            self.thinking.emit(data["text"])
        elif kind == "thinking_done":
            self.thinkingDone.emit(data["text"], data["tokens"])
        elif kind == "memory_recall":
            self.memoryRecall.emit(data["memories"])
        elif kind == "memory_saved":
            self.memorySaved.emit(data["items"])
        elif kind == "todo":
            self.todoUpdate.emit(data["todo"])
        elif kind == "paused":
            self.pausedChanged.emit(bool(data.get("paused")))
        elif kind == "loop":
            self.failed.emit(f"{data['name']} repeated identically "
                             f"{data['count']} times — asking the model to change tack")
        elif kind == "error":
            self.failed.emit(data["message"])

    def _ask(self, tool, args) -> bool:
        preview = "\n".join(f"{k} = {str(v)[:400]}" for k, v in args.items())
        self._gate.clear()
        self.approvalNeeded.emit(tool.name, preview)
        self._gate.wait(timeout=600)
        return self._granted

    def grant(self, ok: bool) -> None:
        self._granted = ok
        self._gate.set()


# ------------------------------------------------------------ node dialog --
class NodeDialog(QDialog):
    def __init__(self, node: Node | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Worker node")
        self.host = QLineEdit(node.host if node else "")
        self.host.setPlaceholderText("192.168.1.42")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(node.port if node else 50052)
        self.label = QLineEdit(node.label if node else "")
        self.mem = QSpinBox()
        self.mem.setRange(0, 1_048_576)
        self.mem.setSingleStep(1024)
        self.mem.setValue(node.mem_mb if node else 0)

        form = QFormLayout()
        form.addRow("Host", self.host)
        form.addRow("Port", self.port)
        form.addRow("Label", self.label)
        form.addRow("Memory (MB)", self.mem)
        hint = QLabel("Memory is what this worker donates. It sets the tensor-split "
                      "proportions so a big box carries more layers than a small one.")
        hint.setWordWrap(True)
        hint.setObjectName("Dim")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(hint)
        lay.addWidget(buttons)

    def node(self) -> Node:
        return Node(host=self.host.text().strip() or "127.0.0.1", port=self.port.value(),
                    label=self.label.text().strip(), mem_mb=self.mem.value())


# ---------------------------------------------------------- cluster panel --
# Narrower than this and labels wrap into unreadable columns; the panels are
# built to fit at exactly this width.
PANEL_MIN_WIDTH = 300


def _summarise_call(name: str, args: dict) -> str:
    """A short phrase describing the call, without its innards."""
    if name in ("skill_open", "skill_find"):
        return str(args.get("name") or args.get("query") or "")
    if name in ("read_file", "write_file", "edit_file", "list_dir"):
        return str(args.get("path") or "")
    if name == "shell":
        return "…"                      # the command belongs in Activity
    if name in ("remember", "recall"):
        return str(args.get("text") or args.get("query") or "")[:60]
    if name in ("plan", "todo", "plan_add"):
        return ""
    first = next(iter(args.values()), "")
    return str(first)[:60]


SHELL_NOISE = ("[exit 0]", "[exit ")

# Tools whose output is working material for the model — instructions, file
# contents, search hits — rather than an answer for the reader. Showing it makes
# a simple request look like a wall of machinery.
QUIET_TOOLS = {"skill_open", "skill_find", "read_file", "grep", "list_dir",
               "find_files", "recall", "remember", "plan", "todo", "plan_add",
               "forget", "persona"}


def _result_only(text: str, name: str = "") -> str:
    """What the reader needs, not what the model needs.

    A skill that fetches the time should show the time — not the instructions it
    read, nor the command it ran, nor the exit status.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].startswith(SHELL_NOISE):
        lines = lines[1:]
    if name in QUIET_TOOLS:
        if name == "skill_open":
            return f"read the skill ({len(text):,} characters)"
        if name in ("read_file", "grep", "find_files", "list_dir"):
            return f"{len(lines)} line(s)"
        return lines[0][:90] if lines else ""
    return "\n".join(lines) if lines else text


class ClusterPanel(UiThread, QWidget):
    uiCall = Signal(object)
    changed = Signal()
    logLine = Signal(str)

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self._init_ui_thread()
        self.cfg = cfg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)

        blurb = QLabel("Each worker runs llama.cpp's rpc-server. Kestrel starts "
                       "llama-server here with --rpc, and the weights and KV cache "
                       "spread across every machine listed.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Dim")
        lay.addWidget(blurb)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Node", "Memory", "Status"])
        self.tree.setObjectName("Flush")
        self.tree.setFont(mono_font(10))
        stretch_columns(self.tree)
        self.tree.itemChanged.connect(self._toggled)
        self.tree.itemDoubleClicked.connect(lambda *_: self.edit_node())
        lay.addWidget(self.tree, 1)

        row1 = QHBoxLayout()
        for text, slot in (("Add…", self.add_node), ("Edit…", self.edit_node),
                           ("Remove", self.remove_node)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row1.addWidget(b)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.discover_btn = QPushButton("Discover")
        self.discover_btn.setToolTip("Listen for kestrel-node beacons on the LAN")
        self.discover_btn.clicked.connect(self.discover)
        probe = QPushButton("Test all")
        probe.clicked.connect(self.probe)
        row2.addWidget(self.discover_btn)
        row2.addWidget(probe)
        lay.addLayout(row2)

        self.split_label = QLabel("")
        self.split_label.setObjectName("Dim")
        self.split_label.setWordWrap(True)
        lay.addWidget(self.split_label)

        self.start_btn = QPushButton("Start llama-server")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self.start_server)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.clicked.connect(self.stop_server)
        self.stop_btn.setEnabled(False)
        row3 = QHBoxLayout()
        row3.addWidget(self.start_btn, 1)
        row3.addWidget(self.stop_btn)
        lay.addLayout(row3)

        self.server = clustermod.ServerProcess(on_log=self.logLine.emit)
        self.refresh()

    # -- model <-> view -------------------------------------------------------
    def refresh(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        for n in self.cfg.nodes:
            item = QTreeWidgetItem([n.display, f"{n.mem_mb or '—'}", "?"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if n.enabled else Qt.Unchecked)
            item.setToolTip(0, n.addr)
            self.tree.addTopLevelItem(item)
        self.tree.blockSignals(False)
        split = clustermod.tensor_split(self.cfg)
        self.split_label.setText(f"tensor-split  {split}" if split else
                                 "tensor-split  even (set memory per node to weight it)")
        self.changed.emit()

    def _toggled(self, item, _col) -> None:
        idx = self.tree.indexOfTopLevelItem(item)
        if 0 <= idx < len(self.cfg.nodes):
            self.cfg.nodes[idx].enabled = item.checkState(0) == Qt.Checked
            self.cfg.save()
            self.refresh()

    def _selected(self) -> int:
        return self.tree.indexOfTopLevelItem(self.tree.currentItem()) if self.tree.currentItem() else -1

    def add_node(self) -> None:
        dlg = NodeDialog(parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.cfg.nodes.append(dlg.node())
            self.cfg.save()
            self.refresh()

    def edit_node(self) -> None:
        i = self._selected()
        if i < 0:
            return
        dlg = NodeDialog(self.cfg.nodes[i], parent=self)
        if dlg.exec() == QDialog.Accepted:
            keep = self.cfg.nodes[i].enabled
            self.cfg.nodes[i] = dlg.node()
            self.cfg.nodes[i].enabled = keep
            self.cfg.save()
            self.refresh()

    def remove_node(self) -> None:
        i = self._selected()
        if i < 0:
            return
        self.cfg.nodes.pop(i)
        self.cfg.save()
        self.refresh()

    # -- network --------------------------------------------------------------
    def discover(self) -> None:
        self.discover_btn.setEnabled(False)
        self.discover_btn.setText("Listening…")

        def work():
            found = clustermod.discover(self.cfg.discovery_port, seconds=4.0)
            known = {n.addr for n in self.cfg.nodes}
            added = 0
            for n in found:
                if n.addr not in known:
                    self.cfg.nodes.append(n)
                    added += 1
            if added:
                self.cfg.save()
            self.logLine.emit(f"[discovery] {len(found)} beacon(s), {added} new node(s)")

            def done():
                self.discover_btn.setText("Discover")
                self.discover_btn.setEnabled(True)
                self.refresh()
            self.ui(done)

        threading.Thread(target=work, daemon=True).start()

    def probe(self) -> None:
        def work():
            # Checked once per run of the test: a mismatch explains a node that
            # connects and drops far better than a latency figure does.
            warning = clustermod.version_warning(self.cfg.llama_server_bin,
                                                 self.cfg.rpc_bin)
            if warning:
                self.logLine.emit("[probe] " + warning)
                self.ui(lambda: self.statusLine.emit(warning))
            results = clustermod.probe_all(self.cfg.nodes)

            def show():
                for i, pr in enumerate(results):
                    item = self.tree.topLevelItem(i)
                    if item is not None:
                        item.setText(2, f"{pr.latency_ms:.0f} ms" if pr.up else "down")
            self.ui(show)
            for pr in results:
                self.logLine.emit(f"[probe] {pr.node.addr} "
                                  + (f"up ({pr.latency_ms:.0f} ms)" if pr.up else f"down: {pr.error}"))

        threading.Thread(target=work, daemon=True).start()

    def start_server(self) -> None:
        try:
            self.server.start(self.cfg, port=_port_of(self.cfg.server_url))
        except Exception as e:
            QMessageBox.warning(self, "Cannot start llama-server", str(e))
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_server(self) -> None:
        self.server.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def _port_of(url: str) -> int:
    try:
        return int(url.rsplit(":", 1)[1].split("/")[0])
    except (ValueError, IndexError):
        return 8080


# ----------------------------------------------------------- skills panel --
class SkillsPanel(QWidget):
    skillsChanged = Signal()

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.skills: list[skillmod.Skill] = []
        self.rejected: list[tuple[str, str]] = []
        # QFileSystemWatcher is not recursive, so every directory under each
        # root is watched individually and the set is rebuilt after each scan.
        self.watcher = QFileSystemWatcher(self)
        self.watcher.directoryChanged.connect(self._on_change)
        self.watcher.fileChanged.connect(self._on_change)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(700)   # editors write in bursts
        self._debounce.timeout.connect(self._rescan_from_watch)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)

        blurb = QLabel("Skills in the agentskills.io format, read from wherever they "
                       "already are — Hermes, Claude Code, or your own folder. Only the "
                       "names and descriptions go in the prompt; the body loads on demand.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Dim")
        lay.addWidget(blurb)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        lay.addWidget(self.list, 2)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        lay.addWidget(self.detail, 1)

        row = QHBoxLayout()
        new = QPushButton("New skill…")
        new.setToolTip("Create a skill folder from a template, ready to edit")
        new.clicked.connect(self.new_skill)
        add = QPushButton("Add folder…")
        add.clicked.connect(self.add_folder)
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self.rescan)
        for b in (new, add, rescan):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(b)
        lay.addLayout(row)
        self.rescan()

    def _on_change(self, _path: str) -> None:
        if self.cfg.watch_skills:
            self._debounce.start()

    def _rescan_from_watch(self) -> None:
        before = {s.name for s in self.skills}
        self.rescan()
        if {s.name for s in self.skills} != before:
            self.skillsChanged.emit()

    def _rewatch(self) -> None:
        if not self.cfg.watch_skills:
            return
        existing = self.watcher.directories() + self.watcher.files()
        if existing:
            self.watcher.removePaths(existing)
        paths: list[str] = []
        for root in self.cfg.skills_dirs:
            base = Path(root).expanduser()
            if not base.is_dir():
                # Watch the parent so the folder being created is itself noticed.
                if base.parent.is_dir():
                    paths.append(str(base.parent))
                continue
            paths.append(str(base))
            try:
                for sub in base.rglob("*"):
                    if sub.is_dir() and len(sub.relative_to(base).parts) <= 3:
                        paths.append(str(sub))
            except OSError:
                continue
        for skill in self.skills:
            paths.append(str(skill.path))
        if paths:
            self.watcher.addPaths(sorted(set(paths))[:200])

    def rescan(self) -> None:
        self.skills, self.rejected = skillmod.discover_detailed(self.cfg.skills_dirs)
        self._rewatch()
        self.list.clear()
        for s in self.skills:
            self.list.addItem(s.name)
        if self.skills:
            summary = (f"{len(self.skills)} skill(s) across "
                       f"{len(self.cfg.skills_dirs)} folder(s). New folders are "
                       "picked up automatically.")
        else:
            summary = ("No skills found yet. Drop a folder containing SKILL.md into "
                       "a watched directory and it will appear here — no restart.")
        if self.rejected:
            summary += f"\n\n{len(self.rejected)} file(s) skipped:"
            for path, reason in self.rejected[:6]:
                summary += f"\n  {Path(path).parent.name}/SKILL.md — {reason}"
        self.detail.setPlainText(summary)

    def new_skill(self) -> None:
        """Create a skill from a template in the first writable skills folder."""
        name, ok = QInputDialog.getText(
            self, "New skill", "Name (lowercase, hyphenated):", text="my-skill")
        if not ok or not name.strip():
            return
        description, ok = QInputDialog.getText(
            self, "New skill",
            "One line: what it does and when to use it.\n"
            "This is what the model sees in every prompt.")
        if not ok:
            return
        target = None
        for candidate in self.cfg.skills_dirs:
            try:
                Path(candidate).expanduser().mkdir(parents=True, exist_ok=True)
                target = candidate
                break
            except OSError:
                continue
        if target is None:
            QMessageBox.warning(self, "No writable folder",
                                "None of the skill folders can be written to. "
                                "Add one with Add folder…")
            return
        try:
            path = skillmod.create_skill(target, name, description)
        except OSError as e:
            QMessageBox.warning(self, "Could not create skill", str(e))
            return
        self.rescan()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        self.detail.setPlainText(
            f"Created {path}\n\nThe folder is open. Edit SKILL.md and it will be "
            "picked up automatically — no restart, no rescan.")

    def add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Skills folder", str(Path.home()))
        if not path:
            return
        if path not in self.cfg.skills_dirs:
            self.cfg.skills_dirs.append(path)
            self.cfg.save()
        self.rescan()

    def _show(self, row: int) -> None:
        if not (0 <= row < len(self.skills)):
            return
        s = self.skills[row]
        res = s.resources()
        lines = [s.description, "", f"folder: {s.root}"]
        if s.problem:
            lines.append(f"note: {s.problem}")
        if res:
            lines.append("bundled: " + ", ".join(res[:12]))
        self.detail.setPlainText("\n".join(lines))


# ------------------------------------------------------------ main window --
class MainWindow(QWidget):
    requestPrepare = Signal()
    requestSend = Signal(str)
    requestReset = Signal()
    requestRebind = Signal(str)
    requestForgetProject = Signal()
    requestNameSession = Signal(str)
    transcriptReady = Signal(str)
    statusReady = Signal(str)
    serverFailed = Signal(str)
    dictated = Signal(str, bool)
    probeFinished = Signal(str, bool, bool)
    backendLocated = Signal(str)
    takeoverDone = Signal(str, bool, str)

    def __init__(self, cfg: Config, progress=None):
        super().__init__()
        self.progress = progress or (lambda message: None)
        self.cfg = cfg
        self.busy = False
        self.speech = speechmod.Speech(cfg)
        self.speaker = self.speech.speaker()
        self.speaker.on_error = lambda msg: self.statusReady.emit(f"Speech: {msg}")
        self.dictation = None
        self._partial_len = 0
        self._ngl_retries = 0
        self._loading_path = ""
        self.downloads = None
        self.downloads_window = None
        self.attachments: list = []
        self._replies: dict[int, dict] = {}
        self._reply_no = 0
        self._pending_prompt = ""
        self._pending_mark = 0
        # Populated as their tabs are first shown.
        self.models_panel = self.params_panel = self.skills_panel = None
        self.memory_panel = self.persona_panel = self.speech_panel = None
        self.backend_panel = self.system_panel = self.tools_panel = None
        self.canvas_panel = None
        self._tool_list: list[dict] = []
        self._gen_start = None
        self._gen_last = 0.0
        self._gen_tokens = 0
        self.session = sessionmod.new_session()
        self.setWindowTitle("Kestrel")
        # The restored-down geometry is set first so that un-maximising gives a
        # sensible window rather than whatever the layout last demanded, and is
        # clamped to the screen so it can never open wider than the monitor.
        self.setMinimumSize(720, 480)
        self.resize(*self._default_size())

        self.chat = ChatView()
        self.activity = ActivityTree()
        self.gauge = ContextGauge()
        self.monitor_strip = MonitorStrip(sysmon.Monitor())
        self.monitor_timer = QTimer(self)
        self.monitor_timer.setInterval(2000)
        self.monitor_timer.timeout.connect(self.monitor_strip.refresh)
        self.monitor_timer.start()
        self.monitor_strip.refresh()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("Flush")
        self.log.setFont(mono_font(10))
        self.log.setMaximumBlockCount(4000)

        self.progress("building the interface")
        self._build()
        self._connect_signals()
        self.statusReady.connect(self._status)
        self._rate_timer = QTimer(self)
        self._rate_timer.setInterval(250)
        self._rate_timer.timeout.connect(self._tick_rate)
        self._rate_timer.start()
        self.serverFailed.connect(self.on_server_failed)
        self._start_worker()

    def _connect_signals(self) -> None:
        """Every window-owned signal, connected in one place.

        These carry results back from worker threads. Connecting them in
        scattered spots is how two of them ended up declared, emitted, and
        wired to nothing at all.
        """
        for signal, slot in (
            (self.transcriptReady, self.on_transcript),
            (self.statusReady, self._status),
            (self.serverFailed, self.on_server_failed),
            (self.dictated, self.on_dictated),
            (self.probeFinished, self._continue_load),
            (self.backendLocated, self._backend_located),
            (self.takeoverDone, self._takeover_done),
        ):
            signal.connect(slot)

    # -- layout ---------------------------------------------------------------
    def _build(self) -> None:
        # Built first: panels constructed below connect their log output to it.
        self.busy_overlay = BusyOverlay()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._top_bar())

        body = QWidget()
        root.addWidget(body, 1)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # left: model / cluster / skills
        left = QTabWidget()
        # Vertical icon rail: nine horizontal tabs never fitted a narrow panel,
        # and elided text ("S...", "M...") named nothing. Icons plus tooltips
        # stay legible at any width.
        self._icon_tabs = ["status", "models", "params", "persona", "cluster",
                           "tools", "skills", "memory", "speech", "backend"]
        # Order matters: the shape is applied to whichever bar is installed, so
        # the custom bar has to be in place before the position is set.
        left.setTabBar(IconTabBar(self._icon_tabs))
        left.setTabPosition(QTabWidget.West)

        self.progress("reading projects")
        self.projects_panel = ProjectsPanel(self.cfg)
        self.projects_panel.statusLine.connect(self._status)
        self.projects_panel.projectChosen.connect(self.open_project)
        self.projects_panel.sessionChosen.connect(self.open_session)
        self.projects_panel.newSession.connect(self.new_session)
        left.addTab(self._status_tab(), "Status")

        left.addTab(self._lazy(lambda: ModelsPanel(self.cfg), self._wire_models),
                    "Models")
        left.addTab(self._lazy(lambda: ParamsPanel(self.cfg), self._wire_params),
                    "Params")
        left.addTab(self._lazy(lambda: PersonaPanel(self.cfg), self._wire_persona),
                    "Persona")

        self.progress("checking the cluster")
        self.cluster = ClusterPanel(self.cfg)
        self.cluster.logLine.connect(self.log.appendPlainText)
        # llama.cpp narrates its own loading; the overlay shows the latest line
        # so a long load is visibly progressing.
        self.cluster.logLine.connect(self.busy_overlay.update_detail)
        left.addTab(self.cluster, "Cluster")

        left.addTab(self._lazy(ToolsPanel, self._wire_tools), "Tools")
        left.addTab(self._lazy(lambda: SkillsPanel(self.cfg), self._wire_skills),
                    "Skills")
        left.addTab(self._lazy(lambda: MemoryPanel(self.cfg), self._wire_memory),
                    "Memory")
        left.addTab(self._lazy(lambda: SpeechPanel(self.cfg), self._wire_speech),
                    "Speech")
        left.addTab(self._lazy(lambda: BackendPanel(self.cfg), self._wire_backend),
                    "Backend")

        # Ignored, not Expanding: a nine-tab QTabWidget reports a size hint wide
        # enough to show every tab, and inside a splitter that hint becomes a
        # floor. Summed across both side panels it pushed the window minimum
        # past the width of the screen, so the window opened larger than the
        # monitor and could not be shrunk.
        left.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        # No widget minimum: it would stop the panel shrinking to its rail. The
        # readable floor is applied when restoring instead.
        left.setMinimumWidth(0)
        self._label_tabs(left)
        self.left_panel = left
        splitter.addWidget(left)

        # centre: transcript and composer
        centre = QWidget()
        centre.setMinimumWidth(240)
        centre.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        clay = QVBoxLayout(centre)
        clay.setContentsMargins(8, 0, 8, 0)
        clay.addWidget(self.chat, 1)

        self.typing = TypingIndicator()
        clay.addWidget(self.typing)
        self.busy_overlay.attach(centre)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Give Kestrel a task.   Enter to send, Shift+Enter for a new line.")
        self.input.setMaximumHeight(96)
        self.input.installEventFilter(self)
        self.attach_bar = QLabel("")
        self.attach_bar.setObjectName("Dim")
        self.attach_bar.setWordWrap(True)
        self.attach_bar.setCursor(Qt.PointingHandCursor)
        self.attach_bar.mousePressEvent = lambda _e: self.clear_attachments()
        self.attach_bar.hide()
        clay.addWidget(self.attach_bar)
        clay.addWidget(self.input)

        # Only the two actions that belong to the message itself stay here; the
        # session-level ones live in the top bar, which always has room. At a
        # narrow window four buttons in this row simply clipped.
        row = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("Primary")
        self.send_btn.clicked.connect(self.send)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        self.attach_btn = QPushButton("+")
        self.attach_btn.setObjectName("Chip")
        self.attach_btn.setToolTip("Attach files — text, code, Word, Excel, "
                                   "PDF or an image")
        self.attach_btn.setMaximumWidth(34)
        self.attach_btn.clicked.connect(self.attach_files)
        row.addWidget(self.attach_btn)

        self.follow_btn = QPushButton("Following")
        self.follow_btn.setObjectName("Chip")
        self.follow_btn.setCheckable(True)
        self.follow_btn.setChecked(True)
        self.follow_btn.setToolTip("Keep the newest output in view. Scrolling up "
                                   "releases it automatically.")
        self.follow_btn.toggled.connect(self.chat.set_following)
        self.chat.followChanged.connect(self._follow_changed)
        self.chat.actionRequested.connect(self.on_reply_action)
        row.addWidget(self.follow_btn)

        self.continue_btn = QPushButton("Continue")
        self.continue_btn.setObjectName("Chip")
        self.continue_btn.setToolTip("Pick the task back up from where it stopped")
        self.continue_btn.clicked.connect(self.continue_task)
        row.addWidget(self.continue_btn)

        row.addStretch(1)

        # Speaking and dictating belong to the message being written, not to the
        # window, so they sit with the other controls that act on it.
        self.speak_btn = QPushButton("Speak")
        self.speak_btn.setObjectName("Chip")
        self.speak_btn.setCheckable(True)
        self.speak_btn.setChecked(self.cfg.speech.tts_enabled)
        self.speak_btn.setToolTip("Read replies aloud")
        self.speak_btn.toggled.connect(self._toggle_tts)
        row.addWidget(self.speak_btn)

        self.mic_btn = QPushButton("Dictate")
        self.mic_btn.setObjectName("Chip")
        self.mic_btn.setToolTip("Dictate into the composer. Words appear as they "
                                "are recognised; press again to stop.")
        self.mic_btn.clicked.connect(self.dictate)
        row.addWidget(self.mic_btn)

        row.addWidget(self.stop_btn)
        row.addWidget(self.send_btn)
        clay.addLayout(row)
        splitter.addWidget(centre)

        # right: activity and server log
        right = QTabWidget()
        # The right column is what the work produces: the project it belongs
        # to, the code being written, the plan, and the machinery underneath.
        self._right_tabs = ["projects", "canvas", "plan", "activity", "log",
                            "monitor"]
        right.setTabBar(IconTabBar(self._right_tabs))
        right.setTabPosition(QTabWidget.East)
        right.addTab(self.projects_panel, "Projects")
        right.addTab(self._lazy(lambda: CanvasPanel(self.cfg), self._wire_canvas),
                     "Canvas")
        self.plan_panel = PlanPanel()
        right.addTab(self.plan_panel, "Plan")
        right.addTab(self.activity, "Activity")
        right.addTab(self.log, "Server log")
        right.addTab(self._lazy(SystemPanel, self._wire_system), "System")
        right.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        right.setMinimumWidth(0)
        self._label_tabs(right, self._right_tabs)
        self.right_panel = right

        splitter.addWidget(right)

        # Left, centre, right — and therefore exactly two handles. The 1px
        # spacers that used to sit between them gave the splitter five children
        # and four handles, which is why each side appeared to have two bars.
        for index, stretch in ((0, 0), (1, 1), (2, 0)):
            splitter.setStretchFactor(index, stretch)
        splitter.setChildrenCollapsible(True)
        # A wider grip, and room either side of it: text running right up to the
        # divider reads as though it has been cut off.
        splitter.setHandleWidth(10)
        splitter.setSizes([340, 620, 300])
        self.splitter = splitter
        self._panel_widths = {"left": 360, "right": 300}
        self._collapsed = {"left": False, "right": False}
        self._content_mins: dict[int, int] = {}
        splitter.splitterMoved.connect(self._clamp_panels)
        left.currentChanged.connect(lambda _i: self._clamp_panels())
        left.currentChanged.connect(lambda _i: QTimer.singleShot(60, self._watch_subtabs))
        right.currentChanged.connect(lambda _i: self._clamp_panels())
        left.tabBar().tabBarClicked.connect(
            lambda i: self._tab_clicked("left", i))
        right.tabBar().tabBarClicked.connect(
            lambda i: self._tab_clicked("right", i))
        outer.addWidget(splitter, 1)
        outer.addWidget(self.gauge)
        # Under the gauge, in space that was empty. The gauge itself is not
        # moved or resized: it is the thing people actually watch.
        outer.addWidget(self.monitor_strip)

    @staticmethod
    def _default_size() -> tuple[int, int]:
        screen = QApplication.primaryScreen()
        if screen is None:
            return 1280, 820
        available = screen.availableGeometry()
        return (max(720, min(1440, int(available.width() * 0.85))),
                max(480, min(900, int(available.height() * 0.85))))

    def _top_bar(self) -> QWidget:
        """Persistent header. It sits above the tab stack rather than inside it,
        so the theme and voice controls are reachable from every page."""
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(46)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(8)

        mark = QLabel("KESTREL")
        mark.setObjectName("Wordmark")
        lay.addWidget(mark)

        self.bar_status = QLabel("")
        self.bar_status.setObjectName("Dim")
        self.bar_status.setMinimumWidth(0)
        self.bar_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.bar_status, 1)

        new_btn = QPushButton("New")
        new_btn.setObjectName("Chip")
        new_btn.setToolTip("Start a new session")
        new_btn.clicked.connect(self.new_session)
        lay.addWidget(new_btn)

        settings_btn = QPushButton("Settings")
        settings_btn.setObjectName("Chip")
        settings_btn.clicked.connect(self.open_settings)
        lay.addWidget(settings_btn)

        self.downloads_btn = QPushButton("Downloads")
        self.downloads_btn.setObjectName("Chip")
        self.downloads_btn.setToolTip("Search and download models in the background")
        self.downloads_btn.clicked.connect(self.open_downloads)
        lay.addWidget(self.downloads_btn)

        self.theme_btn = QPushButton()
        self.theme_btn.setObjectName("Chip")
        self.theme_btn.setToolTip("Switch between the dark and light palette")
        self.theme_btn.clicked.connect(self.toggle_theme)
        lay.addWidget(self.theme_btn)
        self._label_theme_button()
        return bar

    def _label_theme_button(self) -> None:
        self.theme_btn.setText("Light" if theme.current == "dark" else "Dark")

    def apply_appearance(self) -> None:
        """Re-apply palette and fonts, then rebuild anything holding either."""
        sheet = theme.apply(self.cfg.theme, ui=self.cfg.ui_font,
                        mono=self.cfg.mono_font, size=self.cfg.font_size,
                        tint_name=self.cfg.ui_tint,
                        accent_name=self.cfg.ui_accent)
        appl = QApplication.instance()
        if appl is not None:
            appl.setStyleSheet(sheet)
        self.left_panel.tabBar().update()
        clear_font_cache()
        self.chat.rerender()
        self.log.setFont(mono_font(10))
        self.activity.setFont(mono_font(10))
        self.gauge.update()
        if self.speech_panel is not None:
            self._with("speech_panel", lambda p: p.refresh())

    def toggle_theme(self) -> None:
        sheet = theme.toggle()
        self.cfg.theme = theme.current
        self.cfg.save()
        appl = QApplication.instance()
        if appl is not None:
            appl.setStyleSheet(sheet)
        self._label_theme_button()
        # Inline HTML in the transcript does not follow a stylesheet change, so
        # the history is rebuilt in the new palette rather than left stranded.
        self.chat.rerender()
        self.gauge.update()
        self._with("speech_panel", lambda p: p.refresh())
        self._status(f"{theme.current.capitalize()} palette")

    @Slot(bool)
    def _follow_changed(self, following: bool) -> None:
        self.follow_btn.blockSignals(True)
        self.follow_btn.setChecked(following)
        self.follow_btn.setText("Following" if following else "Jump to latest")
        self.follow_btn.blockSignals(False)

    def _toggle_tts(self, on: bool) -> None:
        self.cfg.speech.tts_enabled = on
        self.cfg.save()
        if not on:
            self._silence()
        if self.speech_panel is not None:
            self._with("speech_panel", lambda p: p.tts_on.setChecked(on))
        self._status("Replies will be read aloud" if on else "Speech output off")

    def dictate(self) -> None:
        """Toggle continuous dictation, writing into the box as words arrive."""
        if self.dictation is not None and self.dictation.running:
            self.dictation.stop()
            self.dictation = None
            self.mic_btn.setText("Dictate")
            self._partial_len = 0
            self._status("Dictation stopped")
            return
        if not speechmod.audio_available()[1]:
            QMessageBox.information(self, "No recorder",
                                    "No audio recorder was found. Install ffmpeg, or "
                                    "pip install sounddevice, then try again.")
            return
        self.mic_btn.setText("Stop")
        self._partial_len = 0
        self.dictation = self.speech.dictation(
            on_text=lambda text, partial: self.dictated.emit(text, partial),
            on_status=lambda message: self.statusReady.emit(f"Dictation: {message}"))
        self.dictation.start()
        self.input.setFocus()

    @Slot(str, bool)
    def on_dictated(self, text: str, partial: bool) -> None:
        """Place recognised speech in the composer as it arrives.

        A partial is the whole utterance so far, re-sent as it grows, so the
        previous partial is replaced rather than appended — otherwise every
        word would appear several times over.
        """
        existing = self.input.toPlainText()
        if self._partial_len:
            existing = existing[:-self._partial_len]
        if partial:
            self._partial_len = len(text)
            self.input.setPlainText(existing + text)
        else:
            self._partial_len = 0
            joined = (existing.rstrip() + " " + text.strip()).strip() if existing.strip() else text.strip()
            self.input.setPlainText(joined + " ")
        cursor = self.input.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.input.setTextCursor(cursor)

    @Slot(str)
    def on_transcript(self, text: str) -> None:
        self.mic_btn.setEnabled(True)
        self.mic_btn.setText("Dictate")
        if not text:
            return
        existing = self.input.toPlainText()
        self.input.setPlainText((existing + " " + text).strip() if existing else text)
        self.input.setFocus()
        self._status(f"Heard: {text[:80]}")

    @Slot(str)
    @Slot(int, int)
    def _label_step(self, step: int, total: int) -> None:
        """Say where the work is, in the plan's terms.

        The loop counter and the checklist are different things, and showing
        "step 8" beside a four-step plan invites the reading that they are the
        same one.
        """
        agent = self.worker.agent
        todo = agent.todo if agent is not None else None
        if todo is not None and todo.items and not todo.complete:
            done, count = todo.progress
            current = todo.current
            where = f" — {current.text[:38]}" if current else ""
            self.typing.set_label(f"step {done + 1} of {count}{where}")
            return
        self.typing.set_label("thinking" if step == 1 else f"working ({step})")

    def _speak_chunk(self, chunk: str) -> None:
        """Feed the reply to the speaker as it is written."""
        if self.cfg.speech.tts_enabled:
            self.speaker.push(chunk)

    def _reply_record(self, index: int) -> dict | None:
        """Locate the nth reply in the live history.

        Derived rather than remembered: a map built as replies arrive goes stale
        the moment a conversation is reopened, forked or retried, and the
        history is the thing that actually decides what a retry would re-ask.
        """
        agent = self.worker.agent
        if agent is None:
            return self._replies.get(index)
        seen = 0
        for position, message in enumerate(agent.history):
            if message.get("role") != "assistant" or message.get("tool_calls"):
                continue
            # In the text dialect a tool call is an assistant message too, but
            # it is never shown as a reply — so it must not be counted as one,
            # or the numbering here and in the transcript drift apart.
            visible = strip_calls(str(message.get("content") or "")).strip()
            if not visible:
                continue
            seen += 1
            if seen != index:
                continue
            # The prompt is the last thing the user said before this reply,
            # skipping the tool results that the text dialect files as user
            # messages.
            prompt, mark = "", position
            for back in range(position - 1, -1, -1):
                previous = agent.history[back]
                if previous.get("role") != "user":
                    continue
                text = str(previous.get("content") or "")
                if text.startswith("["):
                    continue
                prompt, mark = text, back
                break
            return {"text": visible, "prompt": prompt,
                    "mark": mark, "end": position + 1}
        return self._replies.get(index)

    @Slot(str, int)
    def on_reply_action(self, action: str, index: int) -> None:
        """Act on a reply: retry it, speak it, copy it, or branch from it."""
        record = self._reply_record(index)
        if record is None:
            self._status("That reply is no longer available")
            return
        if action == "copy":
            text = record.get("text") or self.chat.reply_text(index)
            if not text.strip():
                self._status("Nothing to copy from that reply")
                return
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            if clipboard.supportsSelection():
                clipboard.setText(text, clipboard.Mode.Selection)
            self._status(f"Copied {len(text)} characters")
        elif action == "speak":
            self.speaker.reset()
            self.speaker.push(record["text"])
            self.speaker.flush()
            self._status("Reading aloud")
        elif action == "retry":
            self.retry_reply(record)
        elif action == "fork":
            self.fork_from(record)

    def retry_reply(self, record: dict) -> None:
        """Ask again from the same point.

        The exchange is rewound to just before the prompt, so the model answers
        the original question rather than being asked to revise its own answer.
        """
        agent = self.worker.agent
        if agent is None or self.busy:
            return
        if not record.get("prompt"):
            self._status("Nothing to retry — no prompt precedes that reply")
            return
        agent.history = agent.history[:record["mark"]]
        preview = " ".join(record["prompt"].split())[:60]
        self.chat.add_note(f"Retrying: {preview}", theme.TEXT_DIM)
        self.busy = True
        self.speaker.reset()
        self.typing.start("thinking")
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.requestSend.emit(record["prompt"])

    def fork_from(self, record: dict) -> None:
        """Branch a new conversation containing everything up to this reply.

        The original is left untouched, which is the point: a fork is for trying
        a different direction without losing the one that got here.
        """
        agent = self.worker.agent
        if agent is None:
            return
        self._save_session()
        forked = sessionmod.new_session()
        forked.messages = [dict(m) for m in agent.history[:record["end"]]]
        forked.title = (sessionmod.title_from(forked.messages) + "  (fork)")[:80]
        if agent.todo is not None:
            forked.plan = agent.todo.to_state()
        sessionmod.save_session(forked, self.projects_panel.current_project())
        self.projects_panel.refresh_sessions()
        # Switch to the branch: the transcript is rebuilt from the forked
        # history, so what is on screen matches what the model now has.
        self.open_session(forked)
        self.chat.add_note("Forked. The original conversation is saved and "
                           "unchanged; this one continues from here.", theme.SIGNAL)
        self._status(f"Forked to “{forked.title}”")

    def _speak(self, text: str) -> None:
        if not self.cfg.speech.tts_enabled:
            return
        self.speaker.push(text)
        self.speaker.flush()

    def _with(self, name: str, action) -> None:
        """Run `action` on a panel only if its tab has been opened.

        Deferred panels are None until first shown, and dotting into one is an
        easy mistake to make in a dozen places — this makes the guard the only
        way to reach them.
        """
        panel = getattr(self, name, None)
        if panel is not None:
            try:
                action(panel)
            except Exception as e:
                self.log.appendPlainText(f"[ui] {name}: {e}")

    def _lazy(self, factory, on_ready):
        tab = LazyTab(factory)
        tab.built.connect(on_ready)
        # The tab is built during its show event, which is after currentChanged
        # has already fired — so measuring the page then finds it empty and the
        # panel keeps whatever minimum the previous tab had. Re-clamp once the
        # contents actually exist.
        tab.built.connect(lambda _p: self._clamp_panels())
        return tab

    def _wire_models(self, p) -> None:
        self.models_panel = p
        p.logLine.connect(self.log.appendPlainText)
        p.statusLine.connect(self._status)
        p.loadRequested.connect(self.load_model)
        p.unloadRequested.connect(self.unload_model)

    def _wire_params(self, p) -> None:
        self.params_panel = p
        p.statusLine.connect(self._status)
        p.appearanceChanged.connect(self.apply_appearance)

    def _wire_canvas(self, p) -> None:
        self.canvas_panel = p
        p.statusLine.connect(self._status)
        p.reviewRequested.connect(self.review_canvas)

    def _wire_tools(self, p) -> None:
        self.tools_panel = p
        p.update_tools(self._tool_list)

    def _wire_skills(self, p) -> None:
        self.skills_panel = p
        p.skillsChanged.connect(self._skills_changed)
        if self.worker.agent is not None:
            self.r_skills.set(str(len(p.skills)))

    def _wire_memory(self, p) -> None:
        self.memory_panel = p
        p.statusLine.connect(self._status)

    def _wire_persona(self, p) -> None:
        self.persona_panel = p
        p.statusLine.connect(self._status)
        p.personaChanged.connect(self._persona_changed)

    def _wire_speech(self, p) -> None:
        self.speech_panel = p
        p.statusLine.connect(self._status)
        p.transcribed.connect(self.on_transcript)
        p.tts_on.setChecked(self.cfg.speech.tts_enabled)

    def _wire_backend(self, p) -> None:
        self.backend_panel = p
        p.logLine.connect(self.log.appendPlainText)
        p.statusLine.connect(self._status)

    def _wire_system(self, p) -> None:
        self.system_panel = p

    def _label_tabs(self, tabs: QTabWidget, kinds: list[str] | None = None) -> None:
        """Icon-only tabs, with the name kept as a tooltip."""
        for index, kind in enumerate(kinds or self._icon_tabs):
            if index >= tabs.count():
                break
            name = tabs.tabText(index) or kind.capitalize()
            tabs.setTabToolTip(index, name)
            tabs.setTabText(index, "")

    def _toggle_panel(self, key: str, panel: QWidget, visible: bool) -> None:
        """Remember the width on the way out so restoring returns it to where
        the user had it, rather than to a default."""
        sizes = self.splitter.sizes()
        index = 0 if key == "left" else 2
        centre = 1
        rail = self._rail_width(panel)
        if not visible:
            if sizes[index] > rail + 20:
                self._panel_widths[key] = sizes[index]
            # Collapse to exactly the icon rail: the tabs stay reachable, and
            # the pane border is dropped so no empty sliver shows beside them.
            self._set_collapsed_style(panel, True)
            panel.setMinimumWidth(0)     # so it can shrink to the rail
            sizes[centre] += sizes[index] - rail
            sizes[index] = rail
        else:
            self._set_collapsed_style(panel, False)
            # A minimum Qt enforces during the drag, rather than a correction
            # applied after it: clamping in splitterMoved fights the drag and
            # loses, which is why the floor did not hold.
            floor = self._content_min(panel)
            panel.setMinimumWidth(floor)
            wanted = max(floor, self._panel_widths.get(key, 340))
            restored = min(wanted, max(floor, sizes[centre] - 240))
            sizes[index] = restored
            sizes[centre] = max(240, sizes[centre] - (restored - rail))
        self.splitter.setSizes(sizes)
        self._collapsed[key] = not visible

    def _content_min(self, panel) -> int:  # noqa: C901
        """The narrowest this panel can be drawn without cutting anything off.

        Measured from the page rather than assumed: a widget minimum would stop
        the panel collapsing to its rail, and a fixed number is wrong for a tab
        holding a wide form. The widest page seen so far is remembered, so
        dragging does not get narrower as tabs are opened.
        """
        page = panel.currentWidget()
        needed = 0
        if page is not None:
            # Walk down to whatever is visible: a panel may hold its own tabs,
            # and findChild would otherwise measure the first scroll area in
            # the tree rather than the page on screen. That is why Sampling —
            # the widest of the Params tabs — never set the floor.
            inner = page
            for _ in range(4):
                if isinstance(inner, QTabWidget) and inner.currentWidget():
                    inner = inner.currentWidget()
                    continue
                nested = inner.findChild(QTabWidget) if inner else None
                if nested is not None and nested.currentWidget():
                    inner = nested.currentWidget()
                    continue
                break
            if isinstance(inner, QScrollArea) and inner.widget() is not None:
                inner = inner.widget()
            elif inner is not None and inner.findChild(QScrollArea) is not None:
                area = inner.findChild(QScrollArea)
                inner = area.widget() or area
            # Whichever is larger: a preferred size can be smaller than the
            # width the contents actually refuse to go below, and it is the
            # latter that decides whether a row overflows.
            needed = max(inner.sizeHint().width(),
                         inner.minimumSizeHint().width()) \
                + self._rail_width(panel) + 26
        key = id(panel)
        remembered = self._content_mins.get(key, PANEL_MIN_WIDTH)
        # Capped, but generously: the cap exists to stop one runaway widget
        # dictating the layout, not to clip a page that genuinely needs room.
        needed = max(PANEL_MIN_WIDTH, min(620, needed), remembered)
        self._content_mins[key] = needed
        return needed

    def _watch_subtabs(self) -> None:
        """Re-measure when a panel's own tabs are switched.

        Params holds several pages of different widths; the floor has to follow
        whichever is showing, not whichever happened to be first.
        """
        page = self.left_panel.currentWidget()
        nested = page.findChild(QTabWidget) if page else None
        if nested is not None and not nested.property("watched"):
            nested.setProperty("watched", True)
            nested.currentChanged.connect(lambda _i: self._clamp_panels())
        self._clamp_panels()

    def _clamp_panels(self, *_args) -> None:
        """Keep each expanded panel's minimum in step with what it now holds."""
        for key, panel in (("left", self.left_panel), ("right", self.right_panel)):
            if not self._collapsed.get(key):
                panel.setMinimumWidth(self._content_min(panel))
        sizes = self.splitter.sizes()
        changed = False
        for key, index, panel in (("left", 0, self.left_panel),
                                  ("right", 2, self.right_panel)):
            if self._collapsed.get(key):
                continue
            floor = self._content_min(panel)
            if 0 < sizes[index] < floor and sizes[1] > 200:
                give = min(floor - sizes[index], sizes[1] - 200)
                sizes[index] += give
                sizes[1] -= give
                changed = True
        if changed:
            self.splitter.setSizes(sizes)

    @staticmethod
    def _rail_width(panel) -> int:
        return max(34, panel.tabBar().sizeHint().width() + 2)

    @staticmethod
    def _set_collapsed_style(panel, collapsed: bool) -> None:
        panel.setProperty("collapsed", collapsed)
        panel.style().unpolish(panel)
        panel.style().polish(panel)
        for child in panel.findChildren(QWidget):
            child.setVisible(not collapsed) if child is panel.currentWidget() else None

    def _tab_clicked(self, key: str, index: int) -> None:
        """Clicking the active tab closes the panel; any other opens it there.

        The same gesture both ways, so the icons behave like the collapse
        handle rather than needing it.
        """
        panel = self.left_panel if key == "left" else self.right_panel
        collapsed = self._collapsed.get(key, False)
        if collapsed:
            self._toggle_panel(key, panel, True)
        elif index == panel.currentIndex():
            self._toggle_panel(key, panel, False)

    def _status_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 12, 8)
        lay.setSpacing(6)
        self.r_endpoint = Readout("endpoint", self.cfg.server_url)
        self.r_model = Readout("model")
        self.r_ctx = Readout("context")
        self.r_profile = Readout("profile")
        self.r_dialect = Readout("dialect")
        self.r_tools = Readout("tools")
        self.r_skills = Readout("skills")
        self.r_speed = Readout("speed")
        self.r_step = Readout("step")
        self.r_persona = Readout("persona")
        self.r_think = Readout("thinking")
        self.r_memories = Readout("memories")
        for r in (self.r_endpoint, self.r_model, self.r_ctx, self.r_profile,
                  self.r_dialect, self.r_tools, self.r_skills, self.r_persona,
                  self.r_think,
                  self.r_memories, self.r_speed, self.r_step):
            lay.addWidget(r)

        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.clicked.connect(self.reconnect)
        self.restart_btn = QPushButton("Restart server")
        self.restart_btn.setToolTip("Stop whatever holds the port and start "
                                    "again, reloading the current model if "
                                    "one was loaded")
        self.restart_btn.clicked.connect(self.restart_server)
        lay.addWidget(_row(self.reconnect_btn, self.restart_btn))

        self.detail_box = QCheckBox("Show tool arguments and raw output")
        self.detail_box.setChecked(self.cfg.show_tool_detail)
        self.detail_box.setToolTip("Off: the transcript shows results only. "
                                   "Everything is always kept in Activity.")
        self.detail_box.toggled.connect(self._set_tool_detail)
        lay.addWidget(self.detail_box)

        self.approval_box = QComboBox()
        self.approval_box.addItems(["always", "safe", "never"])
        self.approval_box.setCurrentText(self.cfg.approval)
        self.approval_box.currentTextChanged.connect(self._set_approval)
        lay.addWidget(Field("ask before running", self.approval_box))

        self.status_msg = QLabel("")
        self.status_msg.setWordWrap(True)
        self.status_msg.setObjectName("Dim")
        lay.addWidget(self.status_msg)

        note = QLabel("The prompt is sized to whatever context the server loaded.")
        note.setWordWrap(True)
        note.setObjectName("Dim")
        lay.addWidget(note)
        lay.addStretch(1)
        return w

    def _set_tool_detail(self, on: bool) -> None:
        self.cfg.show_tool_detail = on
        self.cfg.save()

    def _set_approval(self, mode: str) -> None:
        self.cfg.approval = mode
        self.cfg.save()
        if self.worker.agent and self.worker.agent.registry:
            self.worker.agent.registry.approval_mode = mode

    # -- worker ---------------------------------------------------------------
    def _start_worker(self) -> None:
        self.thread = QThread(self)
        self.worker = AgentWorker(self.cfg)
        self.worker.moveToThread(self.thread)
        self.requestPrepare.connect(self.worker.prepare)
        self.requestSend.connect(self.worker.send)
        self.requestReset.connect(self.worker.reset)
        self.requestRebind.connect(self.worker.rebind)
        self.requestForgetProject.connect(self.worker.forget_project)
        self.requestNameSession.connect(self.worker.name_session)
        self.worker.ready.connect(self.on_ready)
        self.worker.failed.connect(self.on_failed)
        self.worker.token.connect(self.chat.stream)
        self.worker.token.connect(self._count_token)
        self.worker.token.connect(lambda _t: self.typing.set_label("writing"))
        self.worker.token.connect(self._speak_chunk)
        self.worker.thinking.connect(lambda _t: self.typing.set_label("reasoning"))
        # Reasoning tokens are tokens: the rate should not read zero through
        # the part of a turn that often takes longest.
        self.worker.thinking.connect(self._count_token)
        self.worker.toolCall.connect(
            lambda name, _a: self.typing.set_label(f"running {name}"))
        self.worker.stepped.connect(self._label_step)
        self.worker.toolCall.connect(self.on_tool_call)
        self.worker.toolResult.connect(self.on_tool_result)
        self.worker.assistantDone.connect(self.on_assistant)
        self.worker.contextUpdate.connect(self.gauge.update_usage)
        self.worker.genStats.connect(self.on_gen)
        self.worker.stepped.connect(lambda s, m: self.r_step.set(f"{s}/{m}"))
        self.worker.approvalNeeded.connect(self.on_approval)
        self.worker.thinking.connect(self.on_thinking)
        self.worker.thinkingDone.connect(self.on_thinking_done)
        self.worker.memoryRecall.connect(self.on_memory_recall)
        self.worker.memorySaved.connect(self.on_memory_saved)
        self.worker.todoUpdate.connect(self.plan_panel.update_todo)
        self.plan_panel.pauseToggled.connect(self.worker.set_paused)
        self.plan_panel.planEdited.connect(self._plan_edited)
        self.plan_panel.statusLine.connect(self._status)
        self.plan_panel.needTodo.connect(self._ensure_todo)
        self.worker.turnFinished.connect(self.on_turn_finished)
        self.thread.start()
        # Start-up is staged rather than attempted at once, and each stage runs
        # after the window is on screen: interface, then the backend, then the
        # connection to it. Nothing here blocks the event loop.
        QTimer.singleShot(120, self._boot)

    # -- slots ----------------------------------------------------------------
    @Slot(object)
    def on_ready(self, info: dict) -> None:
        self.busy_overlay.end()
        b = info["budget"]
        self.r_endpoint.set(self.cfg.server_url)
        self.r_model.set(info["model"] or "unknown")
        self.r_ctx.set(f"{info['n_ctx']:,}")
        self.r_profile.set(info["profile"])
        self.r_dialect.set(info["dialect"])
        self.r_tools.set(str(info["tools"]))
        self.r_skills.set(str(info["skills"]))
        self._tool_list = info.get("tool_list") or []
        self._with("tools_panel", lambda p: p.update_tools(self._tool_list))
        self.r_persona.set(info.get("persona") or "none")
        self.r_think.set(info.get("thinking", "off"))
        self.r_memories.set(str(info.get("memories", 0)))
        self.gauge.n_ctx = info["n_ctx"]
        self.gauge.reserve = b.output
        self.gauge.profile = b.profile
        self.gauge.update()
        self.chat.add_note(
            f"Ready. {info['model'] or 'model'} · {info['n_ctx']:,} tokens ({b.profile}) · "
            f"{info['dialect']} tool calls · {info['tools']} tools · {info['skills']} skills. "
            f"Prompt allowance {b.system:,}, transcript {b.history:,}, "
            f"memory {b.memory:,}, generation reserve {b.output:,}."
            + (" Reasoning reserve applied." if b.thinking else ""), theme.SIGNAL)
        if self.skills_panel is not None:
            self._with("skills_panel", lambda p: p.rescan())
        if self.memory_panel is not None:
            self._with("memory_panel", lambda p: p.refresh())
        self.plan_panel.update_todo(info.get("todo"))
        if info.get("persona"):
            self.chat.add_note(f"persona: {info['persona']}", theme.VIOLET)

    @Slot(str)
    def on_failed(self, message: str) -> None:
        self.chat.add_error(message)
        self.log.appendPlainText("[error] " + message)

    @Slot(str, object)
    def on_tool_call(self, name: str, args: dict) -> None:
        # By default the transcript shows what a tool did, not how. Running a
        # skill should read as an answer, not as a transcript of the script that
        # produced it — the arguments and raw output stay in the Activity tab.
        if self.cfg.show_tool_detail:
            preview = ", ".join(f"{k}={str(v)[:60]}" for k, v in args.items())
        else:
            preview = _summarise_call(name, args)
        if self.chat.streaming():
            # In the text dialect the streamed prose contains the tool block
            # itself; show only whatever the model said around it.
            said = strip_calls(self.chat.streamed_text()).strip()
            if said:
                self.chat.end_assistant(said)
            else:
                self.chat.discard_open()
        self.activity.add_call(name, args)
        if name == "finish":
            return  # the answer itself lands in the transcript a moment later
        self.chat.add_tool_call(name, preview)

    @Slot(str, bool, str)
    def on_tool_result(self, name: str, ok: bool, text: str) -> None:
        shown = text if self.cfg.show_tool_detail else _result_only(text, name)
        self.chat.add_tool_result(name, ok, shown,
                                  max_lines=8 if self.cfg.show_tool_detail else 5)
        self.activity.add_result(name, ok, text)
        self.chat.begin_assistant()

    @Slot(str)
    def on_assistant(self, text: str) -> None:
        self.chat.end_assistant(text)
        agent = self.worker.agent
        self._reply_no += 1
        self._replies[self._reply_no] = {
            "text": text,
            "prompt": self._pending_prompt,
            "mark": self._pending_mark,
            "end": len(agent.history) if agent is not None else 0,
        }
        if self.cfg.speech.tts_enabled:
            # Whatever streamed has already been spoken; this covers a reply
            # delivered whole, such as one arriving through finish().
            if not self.speaker.spoken_anything():
                self.speaker.push(text)
            self.speaker.flush()

    @Slot(str)
    def _count_token(self, _chunk: str) -> None:
        """Count streamed chunks for the live rate.

        llama.cpp emits roughly one chunk per token, which is close enough for a
        readout that exists to show whether generation is healthy.
        """
        now = time.monotonic()
        if self._gen_start is None or now - self._gen_last > 2.0:
            self._gen_start = now
            self._gen_tokens = 0
        self._gen_last = now
        self._gen_tokens += 1

    def _tick_rate(self) -> None:
        if self._gen_start is None:
            return
        now = time.monotonic()
        if now - self._gen_last > 1.5:          # generation has stopped
            self._gen_start = None
            return
        elapsed = now - self._gen_start
        if elapsed > 0.3:
            rate = self._gen_tokens / elapsed
            self.gauge.set_rate(rate)
            self.r_speed.set(f"{rate:.1f} tok/s")

    @Slot(float, int)
    def on_gen(self, tps: float, tokens: int) -> None:
        if tps > 0:
            self.r_speed.set(f"{tps:.1f} tok/s")
            self.gauge.set_rate(tps)

    @Slot(str, str)
    def on_approval(self, name: str, preview: str) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Approve tool call")
        box.setText(f"Run {name}?")
        box.setInformativeText(preview[:1500])
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        granted = box.exec() == QMessageBox.Yes
        self.worker.grant(granted)

    @Slot(str)
    def on_thinking(self, chunk: str) -> None:
        if self.cfg.thinking.show:
            self.chat.stream_thought(chunk)

    @Slot(str, int)
    def on_thinking_done(self, text: str, tokens: int) -> None:
        if self.cfg.thinking.show:
            self.chat.end_thought(text, tokens)
        self.activity.add_note("thinking", f"{tokens} tokens")

    @Slot(object)
    def on_memory_recall(self, memories) -> None:
        if memories:
            self.chat.add_note(
                "recalled " + "; ".join(m.text[:60] for m in memories[:3])
                + (f" (+{len(memories) - 3} more)" if len(memories) > 3 else ""),
                theme.VIOLET)

    @Slot(object)
    def on_memory_saved(self, items) -> None:
        for _mid, text in items:
            self.chat.add_note("remembered: " + text[:90], theme.VIOLET)
        self._with("memory_panel", lambda p: p.refresh())
        self.r_memories.set(str(self.worker.agent.memory.count())
                            if self.worker.agent and self.worker.agent.memory else "0")

    @Slot(str)
    def _status(self, message: str) -> None:
        self.status_msg.setText(message)
        self.bar_status.setText(message)
        self.log.appendPlainText("[ui] " + message)

    def _preflight(self, path: str) -> bool:
        """Check the model can fit before spending a minute finding out.

        Two models of the same size can need very different amounts of memory:
        the weights are only part of it, and the KV cache scales with layers,
        key/value heads and context length. A 16 GB model with a wide cache can
        need more than a 17 GB one with a narrow one, which is why the larger
        file loads and the smaller does not.
        """
        try:
            from .. import gguf as ggufmod
            info = ggufmod.read(path, want_template=False)
        except Exception:
            return True
        if not info.n_layer:
            return True
        ctx = self.cfg.runtime.ctx_size or 4096
        bits = 8 if self.cfg.runtime.cache_type_k.startswith("q8") else 16
        needed = ggufmod.estimate_vram_mb(info, ctx, bits)
        try:
            sample = sysmon.Monitor().sample()
            usable = max(0, sample.mem_total_mb - 4096)   # leave the OS room
        except Exception:
            return True
        if not usable or needed <= usable:
            return True

        fits = ggufmod.context_that_fits(info, usable, bits)
        kv = ggufmod.kv_bytes(info, ctx, bits) / 1024 ** 3
        box = QMessageBox(self)
        box.setWindowTitle("This may not fit")
        box.setText(f"{Path(path).name} needs about {needed / 1024:.1f} GB at "
                    f"{ctx:,} context, and this machine has roughly "
                    f"{usable / 1024:.1f} GB to give.")
        box.setInformativeText(
            f"The weights are {info.size_gb:.1f} GB; the KV cache accounts for "
            f"{kv:.1f} GB of the rest, and it grows with context length."
            + (f"\n\nAt {fits:,} context it would fit." if fits else ""))
        smaller = (box.addButton(f"Use {fits:,} context", QMessageBox.AcceptRole)
                   if fits else None)
        anyway = box.addButton("Load anyway", QMessageBox.ActionRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(smaller or anyway)
        box.exec()
        clicked = box.clickedButton()
        if smaller is not None and clicked is smaller:
            self.cfg.runtime.ctx_size = fits
            self.cfg.save()
            self._with("params_panel", lambda p: p.refresh_preview())
            return True
        return clicked is anyway

    def load_model(self, path: str) -> None:
        """Probe first, on a thread, then continue on the GUI thread."""
        server = self.cluster.server
        if server.running:
            self._continue_load(path, False, False)
            return
        self.busy_overlay.begin("Checking the endpoint", self.cfg.runtime.url())

        def probe():
            taken = clustermod.port_in_use(self.cfg.runtime.host, self.cfg.runtime.port)
            alive = clustermod.endpoint_alive(self.cfg.runtime.url(), timeout=2.0) if taken else False
            self.probeFinished.emit(path, alive, taken)

        threading.Thread(target=probe, daemon=True).start()

    @Slot(str, bool, bool)
    def _continue_load(self, path: str, alive: bool, taken: bool) -> None:
        url = self.cfg.runtime.url()
        server = self.cluster.server
        if not path:
            # An autostart probe: only act when nothing is answering.
            self.busy_overlay.end()
            if alive:
                self.chat.add_note("A server is already running — connecting to it.")
                self.requestPrepare.emit()
            else:
                self.start_backend()
            return

        if taken and not server.running:
            # Every route to a busy port ends at the same dialog, which always
            # offers to stop what is there — a choice between "use it" and
            # "give up" is not a choice worth presenting.
            self.busy_overlay.end()
            self.port_conflict(path, alive)
            return

        if not getattr(self, "_loading_path", "") == path:
            self._ngl_retries = 0        # a different model starts fresh
            self._loading_path = path
            if not self._preflight(path):
                self._status("Load cancelled")
                return
        self.cluster.stop_server()
        self._status(f"Loading {Path(path).name}…")
        self.chat.add_note(f"Loading {Path(path).name}…")
        self.busy_overlay.begin(
            f"Loading {Path(path).name}",
            "llama.cpp is reading the model into memory. A large model takes a "
            "while; progress appears below as the server reports it.")
        try:
            spawned = server.start(self.cfg)
        except Exception as e:
            self.busy_overlay.end()
            if clustermod.is_port_problem(str(e)):
                self.port_conflict(path)
            else:
                QMessageBox.warning(self, "Cannot start llama-server", str(e))
            self._status(str(e))
            return
        self.cluster.start_btn.setEnabled(False)
        self.cluster.stop_btn.setEnabled(True)

        def wait():
            ok = server.wait_healthy(url)
            # The process exiting is the authoritative answer, not the health
            # probe: a foreign server on the same port would answer for it.
            if ok and (server.adopted or server.running):
                self.cfg.server_url = url
                self._ngl_retries = 0
                # Remember whether this model can see, so an attached image is
                # sent rather than described.
                try:
                    from .. import gguf as ggufmod
                    info = ggufmod.read(path, want_template=False)
                    self.cfg.model_vision = bool(info.vision and info.projector)
                    if info.vision and not info.projector:
                        self.statusReady.emit(
                            f"{Path(path).name} can read images, but no mmproj "
                            "file was found beside it — download the projector "
                            "to enable that.")
                except Exception:
                    self.cfg.model_vision = False
                self.statusReady.emit(f"Loaded {Path(path).name}")
                self.requestPrepare.emit()
            else:
                self.serverFailed.emit(server.failure_summary())

        threading.Thread(target=wait, daemon=True).start()

    @Slot(str)
    def on_server_failed(self, summary: str) -> None:
        # An offload that will not fit is worth retrying with less of it before
        # reporting anything: the alternative is a model that simply refuses to
        # load on hardware that can run it.
        if self._retry_with_less_gpu(summary):
            return
        self.busy_overlay.end()
        if clustermod.is_port_problem(summary):
            self.cluster.start_btn.setEnabled(True)
            self.cluster.stop_btn.setEnabled(False)
            self.chat.add_error("llama-server did not start.\n" + summary)
            self.port_conflict(self.cfg.model_path)
            return
        self.cluster.start_btn.setEnabled(True)
        self.cluster.stop_btn.setEnabled(False)
        self._status("llama-server did not start")
        self.chat.add_error("llama-server did not start.\n" + summary)
        hint = ""
        if clustermod.compute_buffer_failure(summary):
            rt = self.cfg.runtime
            hint = ("\n\nThis is the compute buffer, not the weights or the KV "
                    "cache. It is sized by the batch, so a model can fail here "
                    "while comfortably fitting in memory — and a larger model "
                    "with a narrower batch will load where this one did not."
                    f"\n\nBatch is currently {rt.batch_size or 2048} with a "
                    f"micro-batch of {rt.ubatch_size or 512}. Try 512 and 128 "
                    "under Params \u2192 Runtime; it costs prompt-processing "
                    "speed and nothing else.")
        elif "unknown command" in summary:
            hint = ("\n\nThat message comes from the unified `llama` binary, which "
                    "expects a subcommand. Kestrel calls `llama serve` for it — if "
                    "you see this, the configured binary may be an older or "
                    "different build. Check the path under Settings \u2192 Model.")
        QMessageBox.warning(self, "llama-server did not start", summary + hint)

    def _boot(self) -> None:
        """Stage two: locate the backend, bring it up, then connect to it.

        Each stage runs on its own turn of the event loop and reports itself, so
        a slow one is visible rather than looking like a stall.
        """
        if not self.cfg.auto_start_server:
            self.chat.add_note("Connecting…")
            self.requestPrepare.emit()
            return

        self._status("Looking for llama.cpp")

        def locate():
            from .. import llamacpp
            found = llamacpp.find_server(self.cfg.llama_server_bin)
            self.backendLocated.emit(found or "")

        threading.Thread(target=locate, daemon=True).start()

    @Slot(str)
    def _backend_located(self, binary: str) -> None:
        if not binary:
            self.chat.add_error(
                "No llama-server found on this machine, so nothing was started. "
                "Open the Backend tab to install it, or set the path under "
                "Settings \u2192 Model.")
            self._status("llama.cpp not found")
            self.requestPrepare.emit()   # a remote endpoint may still answer
            return
        if binary != self.cfg.llama_server_bin:
            self.cfg.llama_server_bin = binary
            self.cfg.save()
        self.chat.add_note(f"Using {Path(binary).name}. Checking for a running "
                           "server…")
        self._status("Checking for a running server")
        self.maybe_autostart()

    def _recovery_ladder(self) -> list[tuple[str, dict]]:
        """What to try, in order, when a model will not load.

        Ordered by what it costs to give up. Moving the cache to system RAM
        costs a little speed and keeps every layer on the GPU; moving the idle
        experts of a mixture-of-experts model costs little because most of them
        are not used for any given token. Only after those does the offload
        itself come down, and only after that the context — the one change that
        alters what the model can actually do.
        """
        rt = self.cfg.runtime
        attempted = getattr(self.cluster.server, "attempted_ngl", -1)
        steps: list[tuple[str, dict]] = []

        # The compute buffer is sized by the batch, not by the model or the
        # context, and "failed to allocate compute pp buffers" is that buffer
        # rather than the weights or the cache. Shrinking it costs
        # prompt-processing speed and nothing else, so it comes first.
        batch = rt.batch_size or 2048
        ubatch = rt.ubatch_size or 512
        if ubatch > 128 or batch > 512:
            steps.append((f"reducing the batch to {min(512, batch)} and the "
                          f"micro-batch to {min(128, ubatch)}, which is what "
                          "sizes the compute buffer",
                          {"batch_size": min(512, batch),
                           "ubatch_size": min(128, ubatch)}))
        elif ubatch > 32:
            steps.append(("reducing the micro-batch further, to 32",
                          {"batch_size": 128, "ubatch_size": 32}))

        if not rt.no_kv_offload:
            steps.append(("keeping the KV cache in system RAM",
                          {"no_kv_offload": True}))
        if not rt.cpu_moe:
            steps.append(("keeping the mixture-of-experts weights on the CPU",
                          {"cpu_moe": True}))
        if not rt.cache_type_k.startswith("q8"):
            steps.append(("using an 8-bit KV cache, which halves it",
                          {"cache_type_k": "q8_0", "cache_type_v": "q8_0"}))
        if attempted > 8:
            steps.append((f"halving the GPU offload to {attempted // 2} layers",
                          {"n_gpu_layers": attempted // 2}))
        if attempted > 0:
            steps.append(("running entirely on the CPU", {"n_gpu_layers": 0}))
        current = rt.ctx_size or 4096
        if current > 4096:
            steps.append((f"halving the context to {max(4096, current // 2):,}",
                          {"ctx_size": max(4096, current // 2)}))
        return steps

    def _retry_with_less_context(self, _summary: str) -> bool:
        """Nothing on the GPU and it still will not fit: shrink the cache.

        The remaining lever is context length, and halving it halves the KV
        cache — which on a wide model is several gigabytes.
        """
        current = self.cfg.runtime.ctx_size or 4096
        if self._ngl_retries >= 3 or current <= 4096:
            return False
        self._ngl_retries += 1
        self.cfg.runtime.ctx_size = max(4096, current // 2)
        self.cfg.save()
        self.chat.add_note(
            f"Still would not fit with nothing on the GPU. Trying again at "
            f"{self.cfg.runtime.ctx_size:,} context — the KV cache is what is "
            "too large, and it scales with context.", theme.AMBER)
        self._with("params_panel", lambda p: p.refresh_preview())
        QTimer.singleShot(200, lambda: self.load_model(self.cfg.model_path))
        return True

    def _retry_with_less_gpu(self, summary: str) -> bool:
        """Halve the GPU offload and try again, down to running on the CPU."""
        if not clustermod.looks_like_oom(summary):
            return False
        if self._ngl_retries >= 8:
            self.chat.add_error(
                "It will not fit on this machine at any setting tried — a "
                "smaller batch, the cache in system RAM, experts on the CPU, an "
                "8-bit cache, no GPU offload, and a smaller context. A smaller "
                "quantisation is the next thing to try.")
            return False
        steps = self._recovery_ladder()
        if not steps:
            return False
        description, changes = steps[0]
        self._ngl_retries += 1
        for key, value in changes.items():
            setattr(self.cfg.runtime, key, value)
        self.cfg.save()
        self.chat.add_note(f"That did not fit. Trying again {description}.",
                           theme.AMBER)
        self._status(f"Retrying: {description}")
        self.busy_overlay.update_detail(description)
        self._with("params_panel", lambda p: p.refresh_preview())
        QTimer.singleShot(200, lambda: self.load_model(self.cfg.model_path))
        return True

    def maybe_autostart(self) -> None:
        """Start the backend if nothing is serving yet.

        The probing happens on a worker thread. Asking whether a port answers
        takes milliseconds when the answer is no and five seconds when a
        firewall drops the packets instead of refusing them, and doing that on
        the GUI thread is indistinguishable from the application hanging.
        """
        url = self.cfg.runtime.url()

        def probe():
            alive = (clustermod.endpoint_alive(url, timeout=1.5)
                     or clustermod.endpoint_alive(self.cfg.server_url, timeout=1.5))
            self.probeFinished.emit("", alive, False)

        threading.Thread(target=probe, daemon=True).start()

    def restart_server(self) -> None:
        """Stop the backend — ours or a stranger's — and start it again."""
        host, port = self.cfg.runtime.host, self.cfg.runtime.port
        model = self.cfg.model_path if self.cluster.server.running else ""
        self.busy_overlay.begin("Restarting llama-server", f"{host}:{port}")

        def work():
            self.cluster.server.stop()
            if clustermod.port_in_use(host, port):
                clustermod.stop_listener(host, port)
            self.takeoverDone.emit(model, True, "the running server")
        threading.Thread(target=work, daemon=True).start()

    def port_conflict(self, path: str = "", alive: bool = True) -> None:
        """Offer the same three ways out of a busy port, wherever it came from.

        A dialog with only Yes and No forces a choice between two things the
        user may not want; the useful third option — stop what is there and
        start again — is the one that actually resolves it.
        """
        host, port = self.cfg.runtime.host, self.cfg.runtime.port
        url = self.cfg.runtime.url()
        pid, name = clustermod.listener_on(host, port)
        who = f"{name} (pid {pid})" if pid else "another process"

        box = QMessageBox(self)
        box.setWindowTitle("That port is already in use")
        box.setText(f"{who} is already using {url}.")
        box.setInformativeText(
            "A model cannot be loaded into a server Kestrel did not start. "
            "Most often this is a llama-server left behind by a previous run."
            if alive else
            f"Port {port} is held but nothing is answering on it.")
        restart = box.addButton("Stop it and start fresh", QMessageBox.AcceptRole)
        reuse = box.addButton("Connect to it as it is", QMessageBox.ActionRole) \
            if alive else None
        change = box.addButton("Use a different port", QMessageBox.ActionRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(restart)
        box.exec()
        clicked = box.clickedButton()

        if clicked is restart:
            self._takeover(path or self.cfg.model_path)
        elif reuse is not None and clicked is reuse:
            self.cfg.server_url = url
            self._status(f"Using the server already running on {url}")
            self.requestPrepare.emit()
        elif clicked is change:
            self._choose_new_port(path)

    def _choose_new_port(self, path: str = "") -> None:
        """Move to a free port instead of fighting for this one."""
        suggestion = self.cfg.runtime.port
        while suggestion < 65000 and clustermod.port_in_use(self.cfg.runtime.host,
                                                            suggestion):
            suggestion += 1
        port, ok = QInputDialog.getInt(self, "Port", "Listen on port:",
                                       suggestion, 1024, 65535)
        if not ok:
            return
        self.cfg.runtime.port = port
        self.cfg.server_url = self.cfg.runtime.url()
        self.cfg.save()
        self._with("params_panel", lambda p: p.refresh_preview())
        self._status(f"Port set to {port}")
        if path:
            self.load_model(path)
        else:
            self.start_backend()

    def _takeover(self, path: str) -> None:
        """Stop the stranger on the port, then load as normal."""
        host, port = self.cfg.runtime.host, self.cfg.runtime.port
        self.busy_overlay.begin("Stopping the other server", f"{host}:{port}")

        def work():
            stopped, detail = clustermod.stop_listener(host, port)
            self.takeoverDone.emit(path, stopped, detail)
        threading.Thread(target=work, daemon=True).start()

    @Slot(str, bool, str)
    def _takeover_done(self, path: str, stopped: bool, detail: str) -> None:
        self.busy_overlay.end()
        if not stopped:
            QMessageBox.warning(
                self, "Could not stop it",
                f"{detail}\n\nStop it yourself, or change the port under "
                "Params \u2192 Runtime.")
            self._status("Port still held")
            return
        self.chat.add_note(f"Stopped {detail} and took over the port.",
                           theme.AMBER)
        if path:
            self.load_model(path)
        else:
            self.start_backend()

    def start_backend(self) -> None:
        """Bring llama.cpp up with no model loaded.

        The backend should be running when the window opens so that choosing a
        model is the only remaining step — but loading one takes minutes for a
        large file, and doing that unasked at launch is the wrong default.
        """
        server = self.cluster.server
        if server.running:
            return
        self._status("Starting llama-server (no model loaded)")
        self.busy_overlay.begin("Starting llama-server",
                                "No model is being loaded — choose one from the "
                                "Models tab when you are ready.")
        try:
            server.start(self.cfg, with_model=False)
        except Exception as e:
            self.busy_overlay.end()
            self._status(f"Could not start llama-server: {e}")
            if clustermod.is_port_problem(str(e)):
                self.port_conflict()
            else:
                self.chat.add_error(f"Could not start llama-server: {e}")
            return
        self.log.appendPlainText("$ " + " ".join(server.command))
        self.cluster.start_btn.setEnabled(False)
        self.cluster.stop_btn.setEnabled(True)
        url = self.cfg.runtime.url()

        def wait():
            ok = server.wait_healthy(url, timeout=120)
            if ok and (server.adopted or server.running):
                self.cfg.server_url = url
                self.statusReady.emit("llama-server ready — no model loaded yet")
                self.requestPrepare.emit()
            else:
                self.serverFailed.emit(server.failure_summary())

        threading.Thread(target=wait, daemon=True).start()

    def unload_model(self) -> None:
        """Drop the model but leave the backend running and ready."""
        self.cluster.stop_server()
        self._status("Model unloaded")
        self.chat.add_note("Model unloaded. Restarting the server without one.")
        self.start_backend()

    @Slot()
    def _skills_changed(self) -> None:
        """A skill appeared or changed on disk. Reload it into the running agent
        rather than waiting for a reconnect — dropping a folder in and having it
        work is the point."""
        agent = self.worker.agent
        if agent is None:
            return
        before = len(agent.skills)
        agent.reload_skills()
        after = len(agent.skills)
        if after != before:
            self._status(f"Skills reloaded — {after} available")
            self.chat.add_note(f"Skills reloaded: {after} available", theme.SIGNAL)
        self.r_skills.set(str(after))

    @Slot()
    def _ensure_todo(self) -> None:
        """Hand the panel the running checklist so steps can be added by hand
        before the model has produced one of its own."""
        agent = self.worker.agent
        if agent is not None and agent.todo is not None:
            self.plan_panel.todo = agent.todo

    @Slot()
    def _persona_changed(self) -> None:
        agent = self.worker.agent
        if agent is None:
            return
        persona = agent.reload_persona()
        name = persona.name if persona and persona.any_content() else ""
        self.r_persona.set(name or "none")
        self.chat.add_note(
            f"Persona: {name}. It applies from the next message." if name
            else "Persona cleared.", theme.VIOLET)

    def _plan_edited(self) -> None:
        agent = self.worker.agent
        if agent is not None and agent.todo is not None:
            agent.todo.save()
        self._status("Plan updated — the model sees it on its next step")

    def _save_session(self) -> None:
        """Persist after every turn rather than on exit: a crash or a closed
        window should not cost the conversation."""
        agent = self.worker.agent
        if agent is None or not agent.history:
            return
        self.session.messages = [dict(m) for m in agent.history]
        self.session.model = self.r_model.value.text()
        if agent.todo is not None:
            self.session.plan = agent.todo.to_state()
        if agent.ctx is not None:
            self.session.digest = agent.ctx.digest
        try:
            sessionmod.save_session(self.session, self.projects_panel.current_project())
            self.projects_panel.refresh_sessions()
        except Exception as e:
            self.log.appendPlainText(f"[sessions] could not save: {e}")

    @Slot(str)
    def open_project(self, path: str) -> None:
        self.cfg.workspace = path
        self.cfg.save()
        self.session = sessionmod.new_session()
        # A different folder is a different workspace: its own checklist is
        # loaded from disk when the agent reconnects.
        self.plan_panel.update_todo(None)
        self.chat.clear()
        self.chat.clear_log()
        self.activity.clear()
        self.attachments = []
        self._replies.clear()
        # Not just a reset: the checklist, thinking log, memory scope and file
        # sandbox all belong to a folder and have to move with it.
        self.requestRebind.emit(path)
        self.projects_panel.refresh()
        if self.memory_panel is not None:
            self._with("memory_panel", lambda p: p.reopen())
        self._status(f"Project: {Path(path).name}")
        self.chat.add_note(f"Switched to project {Path(path).name}. Files, memory, "
                           "checklist and conversations are all scoped to it.",
                           theme.SIGNAL)
        self.reconnect()

    @Slot(object)
    def open_session(self, session) -> None:
        agent = self.worker.agent
        if agent is None:
            self._status("Not connected yet")
            return
        self.session = session
        self.requestNameSession.emit(session.id)
        agent.load_history(session.messages, session.digest)
        if agent.todo is not None:
            # The checklist belongs to this conversation, not to whatever was
            # open a moment ago.
            agent.todo.load_state(session.plan)
            self.plan_panel.update_todo(agent.todo)
        self.chat.clear()
        self.chat.clear_log()
        self.activity.clear()
        # Replayed rather than restored verbatim: the transcript is the model's
        # history, and this is the reader's view of it.
        for message in session.messages:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user" and not content.startswith("["):
                self.chat.add_user(content)
            elif role == "assistant" and content:
                self.chat.begin_assistant()
                self.chat.end_assistant(strip_calls(content) or content)
        self.chat.add_note(f"Reopened: {session.title}", theme.SIGNAL)
        self._status(f"Reopened conversation from {session.when()}")

    def _ring(self) -> None:
        """A short chime when a task ends.

        Long tasks are worth walking away from, and a finished one is otherwise
        indistinguishable from a stalled one at a glance.
        """
        if not self.cfg.bell_on_finish:
            return
        chosen = (self.cfg.bell_sound or "").strip()
        bell = (Path(chosen) if chosen else
                Path(__file__).resolve().parent.parent.parent / "assets" / "bell.wav")
        if not bell.exists():
            return

        def work():
            try:
                speechmod.play(bell, blocking=False)
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    @Slot()
    def on_turn_finished(self) -> None:
        self.typing.stop()
        self.chat.finish_turn()
        self._ring()
        agent = self.worker.agent
        unfinished = bool(agent is not None and agent.todo is not None
                          and agent.todo.items and not agent.todo.complete)
        self.continue_btn.setEnabled(True)
        self.continue_btn.setText("Continue" + (" ▸" if unfinished else ""))
        self._save_session()
        self.plan_panel.set_running(False)
        self.busy = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.r_step.set("idle")

    # -- actions --------------------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802
        from PySide6.QtCore import QEvent
        if obj is self.input and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
                    event.modifiers() & Qt.ShiftModifier):
                self.send()
                return True
        return super().eventFilter(obj, event)

    def send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or self.busy:
            return
        self.input.clear()
        payload = text
        if self.attachments:
            # Pictures travel as pictures when the model can read them, and as
            # a description when it cannot.
            payload = attachmod.message_content(self.attachments, text,
                                                vision=self.cfg.model_vision)
            if isinstance(payload, str):
                text = payload
            self.clear_attachments()
        agent = self.worker.agent
        self._pending_prompt = text
        self._pending_mark = len(agent.history) if agent is not None else 0
        self.chat.add_user(text)
        self.chat.begin_assistant()
        self.busy = True
        self.speaker.reset()
        self.typing.start("thinking")
        self.plan_panel.set_running(True)
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.continue_btn.setEnabled(False)
        self.requestSend.emit(text)

    def attach_files(self) -> None:
        """Read files into the next message.

        The text goes with the message rather than into a file the model has to
        be told to open: an attachment is context for what is being asked, not
        a task in itself.
        """
        chosen, _ = QFileDialog.getOpenFileNames(
            self, "Attach files", str(Path(self.cfg.workspace).expanduser()),
            "All files (*);;Text and code (*.txt *.md *.py *.js *.json *.csv);;"
            "Documents (*.docx *.xlsx *.pptx *.pdf *.odt);;"
            "Images (*.png *.jpg *.jpeg *.gif *.webp)")
        if not chosen:
            return
        for path in chosen:
            item = attachmod.read(path)
            self.attachments.append(item)
            if item.kind == "image":
                self._status(f"{item.name} attached — a text model cannot read "
                             "its contents, only that it is there")
        self._show_attachments()

    def _show_attachments(self) -> None:
        if not self.attachments:
            self.attach_bar.hide()
            return
        names = ", ".join(a.label for a in self.attachments)
        self.attach_bar.setText(f"Attached: {names}    (clear)")
        self.attach_bar.show()

    def clear_attachments(self) -> None:
        self.attachments = []
        self._show_attachments()

    def continue_task(self) -> None:
        """Resume: the open step if there is one, otherwise where it left off.

        A turn can end with work outstanding — stopped by hand, cut short by the
        step limit, or a reply that trailed away. Retyping the request loses the
        thread; this hands the model back its own place in the plan.
        """
        agent = self.worker.agent
        if agent is None or self.busy:
            return
        if not agent.history:
            self._status("Nothing to continue yet")
            return
        prompt = "Continue from where you stopped. Use your tools; do not repeat "\
                 "work that is already done."
        todo = agent.todo
        if todo is not None and todo.items and not todo.complete:
            done, total = todo.progress
            current = todo.current
            prompt = (f"Continue. {done} of {total} steps are done"
                      + (f"; step {current.id} is open: {current.text}"
                         if current else "")
                      + ". Carry on with your tools, update the checklist as you "
                        "go, and call finish only when every step is closed.")
        self.chat.add_note("Continuing…", theme.TEXT_DIM)
        self.busy = True
        self.speaker.reset()
        self.typing.start("thinking")
        self.plan_panel.set_running(True)
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._pending_prompt = prompt
        self._pending_mark = len(agent.history)
        self.requestSend.emit(prompt)

    def stop(self) -> None:
        """Stop generation and speech together.

        A reply that has been cancelled should not keep being read out; the
        speaker is killed rather than allowed to finish its queue.
        """
        self._silence()
        self.worker.cancel()
        self.chat.add_note("Stopping…")

    def _silence(self) -> None:
        self.speaker.stop()
        self.speaker = self.speech.speaker()
        self.speaker.on_error = lambda msg: self.statusReady.emit(f"Speech: {msg}")

    def new_session(self) -> None:
        self._save_session()
        self.session = sessionmod.new_session()
        self.requestNameSession.emit(self.session.id)
        agent = self.worker.agent
        if agent is not None and agent.todo is not None:
            agent.todo.clear()
            self.plan_panel.update_todo(agent.todo)
        # A new session starts the project's memory over too. What was learnt
        # about the machine and about the person is kept: neither stopped being
        # true because a conversation ended.
        self.requestForgetProject.emit()
        self.attachments = []
        self.requestReset.emit()
        self.chat.clear()
        self.chat.clear_log()
        self.activity.clear()
        self._replies.clear()
        self._reply_no = 0
        self.chat.add_note("New conversation.")
        self.projects_panel.refresh_sessions()

    def reconnect(self) -> None:
        self.worker.cfg = self.cfg
        self.chat.add_note("Reconnecting…")
        self.requestPrepare.emit()

    def open_downloads(self) -> None:
        """Open the download window, creating it the first time.

        The manager lives on the window rather than in the dialog, so transfers
        carry on when it is closed and reopening shows them still running.
        """
        if self.downloads is None:
            from ..config import config_dir
            self.downloads = dlmod.DownloadManager(
                max_concurrent=2, token=self.cfg.hf_token,
                state_path=config_dir() / "downloads.json")
        if self.downloads_window is None:
            self.downloads_window = DownloadsWindow(self.cfg, self.downloads, self)
            self.downloads_window.statusLine.connect(self._status)
            self.downloads_window.finished.connect(self._downloaded)
        self.downloads_window.show()
        self.downloads_window.raise_()
        self.downloads_window.activateWindow()

    @Slot(str)
    def _downloaded(self, path: str) -> None:
        self.log.appendPlainText(f"[download] {path}")
        self._with("models_panel", lambda p: p.rescan())

    @Slot(str, str)
    def review_canvas(self, text: str, language: str) -> None:
        """Send the canvas to the model for review.

        The code travels in the message rather than being written to disk
        first: reviewing what is on screen is the point, and saving a
        half-finished file only to read it back is a detour.
        """
        if self.busy:
            self._status("Busy — wait for the current turn to finish")
            return
        name = self.canvas_panel.filename.text().strip() if self.canvas_panel else ""
        prompt = (
            f"Review this {language} from the canvas"
            + (f" ({name})" if name else "") + ".\n\n"
            "Point out anything that is wrong or will break, then anything "
            "worth improving. Be specific about lines. If it is fine, say so "
            "briefly rather than inventing problems.\n\n"
            f"```{language}\n{text.rstrip()}\n```")
        self.chat.add_note(f"Reviewing the canvas ({len(text):,} characters)…",
                           theme.TEXT_DIM)
        self.busy = True
        self.speaker.reset()
        self.typing.start("reading the code")
        self.plan_panel.set_running(True)
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.continue_btn.setEnabled(False)
        agent = self.worker.agent
        self._pending_prompt = prompt
        self._pending_mark = len(agent.history) if agent is not None else 0
        self.requestSend.emit(prompt)

    def open_settings(self) -> None:
        """Application settings. Model settings live in the Params panel."""
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() == QDialog.Accepted:
            self.cfg = dlg.apply()
            self.worker.cfg = self.cfg
            self.cluster.cfg = self.cfg
            self.cluster.refresh()
            self.projects_panel.cfg = self.cfg
            self.projects_panel.refresh()
            for name in ("skills_panel", "memory_panel", "models_panel",
                         "params_panel", "persona_panel", "speech_panel",
                         "backend_panel"):
                self._with(name, lambda p: setattr(p, "cfg", self.cfg))
            self._with("skills_panel", lambda p: p.rescan())
            self._with("memory_panel", lambda p: p.reopen())
            self._with("persona_panel", lambda p: p.rescan())
            self.apply_appearance()
            self.reconnect()

    def shutdown_backend(self) -> None:
        """Stop the server Kestrel started, once, from wherever exit came."""
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True
        try:
            if self.cluster.server.running:
                self.cluster.server.stop(wait_for_port=False)
        except Exception:
            pass

    def closeEvent(self, event):  # noqa: N802
        try:
            if self.downloads is not None:
                self.downloads.stop_all()
            if self.dictation is not None:
                self.dictation.stop()
            self.speaker.stop()
            # A server Kestrel started belongs to Kestrel: leaving it running
            # holds the port and the model's memory after the window is gone.
            self.log.appendPlainText("[shutdown] stopping llama-server")
            self.shutdown_backend()
            self.worker.cancel()
            self.cluster.server.stop()
            self.thread.quit()
            self.thread.wait(2000)
        except Exception:
            pass
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else [sys.argv[0]] + list(argv))
    app.setApplicationName("Kestrel")
    cfg = Config.load()
    app.setStyleSheet(theme.apply(cfg.theme, ui=cfg.ui_font, mono=cfg.mono_font,
                                  size=cfg.font_size, tint_name=cfg.ui_tint,
                                  accent_name=cfg.ui_accent))
    install_wheel_guard(app)
    window_holder = {}

    def shutdown():
        # Belt and braces alongside the job object: a clean exit should not rely
        # on closeEvent being delivered, and this runs on any route out of the
        # event loop.
        win = window_holder.get("win")
        if win is not None:
            try:
                win.shutdown_backend()
            except Exception:
                pass

    app.aboutToQuit.connect(shutdown)
    atexit.register(shutdown)

    splash = Splash()
    splash.begin()
    splash.message("reading settings")
    try:
        win = MainWindow(cfg, progress=splash.message)
        window_holder["win"] = win
        splash.message("ready")
        win.showMaximized()
    finally:
        splash.finish(locals().get("win"))
    return app.exec()
