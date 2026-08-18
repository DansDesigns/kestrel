"""Model browser, backend controls and the memory browser."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
import re

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QFrame, QPlainTextEdit, QTabBar,
                               QSizePolicy,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSpinBox, QTabWidget, QTextEdit,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import models as modelsmod
from ..gguf import estimate_vram_mb, kv_bytes as gguf_kv_bytes
from .. import attach as attachmod
from .. import canvas as canvasmod
from .. import (llamacpp, persona as personamod, sessions as sessionmod,
                speech as speechmod, sysmon)
from .. import memory as memorymod
from ..memory import KINDS, MemoryStore
from ..todo import BLOCKED, DISPLAY_MARKS, DOING, DONE, MARKS, TODO
from ..runtime import CACHE_TYPES, REASONING_FORMATS, ROPE_SCALING, SPLIT_MODES
from . import theme
from PySide6.QtGui import QFont, QFontDatabase

from .widgets import (Field, GpuSplit, Readout, WrappingDelegate, mono_font,
                      stretch_columns)

# Windows ships bitmap-only faces with no outlines for Qt to instantiate;
# listing them logs a DirectWrite failure each and yields nothing usable.
LEGACY_RASTER = {"8514oem", "courier", "fixedsys", "modern", "ms sans serif",
                 "ms serif", "roman", "script", "small fonts", "system", "terminal"}


def _usable_font(family: str) -> bool:
    return family.lower() not in LEGACY_RASTER and not family.startswith("@")


def _row(*widgets, stretch_last: bool = False) -> QWidget:
    """Controls side by side, each free to shrink with the panel."""
    w = QWidget()
    w.setMinimumWidth(0)
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for i, x in enumerate(widgets):
        # Controls share the row evenly and shrink with it, rather than each
        # holding its preferred width and overflowing the panel.
        if x.sizePolicy().horizontalPolicy() != QSizePolicy.Fixed:
            x.setMinimumWidth(0)
            x.setSizePolicy(QSizePolicy.Ignored, x.sizePolicy().verticalPolicy())
        lay.addWidget(x, 1 if (stretch_last and i == len(widgets) - 1) else 1)
    return w


class UiThread:
    """Mixin giving a panel a way to run a callable on the GUI thread.

    Every panel here does slow work — scanning disks, HTTP, subprocesses — on a
    background thread. Touching a widget from that thread is undefined
    behaviour in Qt and shows up as an occasional segfault rather than an
    exception, so results come back through this instead.
    """

    def _init_ui_thread(self) -> None:
        self.uiCall.connect(self._run_on_ui)

    @staticmethod
    def _run_on_ui(fn) -> None:
        try:
            fn()
        except Exception:
            pass

    def ui(self, fn) -> None:
        self.uiCall.emit(fn)


def _heading(text: str) -> QLabel:
    """A section title, written as words rather than a shouted keyword.

    The eyebrow style upper-cases and letter-spaces its text, which suits a
    two-word field label and makes a section heading look like a warning label.
    """
    label = QLabel(text)
    label.setObjectName("Section")
    return label


def _form() -> QFormLayout:
    """Labels above their fields.

    Side panels are narrow and the labels are long ("Repeat penalty / window"),
    so a two-column form sets a floor well past what the panel can give and the
    controls run off the edge. Wrapping every row removes the floor entirely.
    """
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.WrapAllRows)
    form.setHorizontalSpacing(6)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setLabelAlignment(Qt.AlignLeft)
    form.setContentsMargins(0, 0, 0, 0)
    form.setSpacing(4)
    return form


def _scroll(inner: QWidget) -> QScrollArea:
    """Wrap a panel body so it can be narrower than its contents want to be.

    Without this the widest control in a tab sets a floor for the whole panel,
    and the panel sets a floor for the window."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameStyle(0)
    area.setWidget(inner)
    area.setMinimumWidth(0)
    area.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    inner.setContentsMargins(10, 8, 12, 10)
    return area


# =========================================================== models panel ===
class ModelsPanel(UiThread, QWidget):
    uiCall = Signal(object)
    loadRequested = Signal(str)     # model path
    unloadRequested = Signal()
    logLine = Signal(str)
    statusLine = Signal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._init_ui_thread()
        self.catalog = modelsmod.Catalog(
            cfg.model_dirs, cache_path=Path(cfg.memory.db_path).parent / "models.json")
        self._dl_cancel = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)
        # Local models only: downloading has its own window, because it runs
        # for an hour and is not something to watch inside a settings panel.
        lay.addWidget(_scroll(self._local_tab()))
        self.rescan()

    # -- local ---------------------------------------------------------------
    def _local_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(6)

        self.filter = QLineEdit()
        self.filter.setClearButtonEnabled(True)
        self.filter.setPlaceholderText("Filter models…")
        self.filter.textChanged.connect(self._refilter)
        lay.addWidget(self.filter)
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self.rescan)
        folder = QPushButton("Add folder…")
        folder.clicked.connect(self._add_folder)
        lay.addWidget(_row(rescan, folder))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["", "Model", "Quant", "Size", "Ctx"])
        self.tree.setFont(mono_font(10))
        # The star column is a marker: fixed, no handle, no share of the width.
        stretch_columns(self.tree, first_stretch=1, fixed={0: 22})
        # Flush to the panel: a framed, rounded box inside an already-framed
        # panel wastes a margin on each side and reads as a box within a box.
        self.tree.setObjectName("Flush")
        self.tree.setFrameShape(QFrame.NoFrame)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(0)
        self.tree.setRootIsDecorated(False)
        self.tree.header().setStretchLastSection(False)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.currentItemChanged.connect(self._show_detail)
        self.tree.itemDoubleClicked.connect(lambda *_: self.load_selected())
        lay.addWidget(self.tree, 2)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setFont(mono_font(10))
        self.detail.setObjectName("Flush")
        self.detail.setFrameShape(QFrame.NoFrame)
        self.detail.setMaximumHeight(168)
        lay.addWidget(self.detail)

        self.star_btn = QPushButton("Favourite  ★")
        self.star_btn.setToolTip("Keep this model at the top of the list")
        self.star_btn.clicked.connect(self.toggle_favourite)
        lay.addWidget(self.star_btn)

        self.load_btn = QPushButton("Load model")
        self.load_btn.setObjectName("Primary")
        self.load_btn.clicked.connect(self.load_selected)
        self.unload_btn = QPushButton("Unload")
        self.unload_btn.clicked.connect(self.unloadRequested.emit)
        lay.addWidget(_row(self.load_btn, self.unload_btn))
        self.delete_btn = QPushButton("Delete from disk…")
        self.delete_btn.setObjectName("Danger")
        self.delete_btn.setToolTip("Permanently remove this model file, and its "
                                   "other parts if it is sharded")
        self.delete_btn.clicked.connect(self.delete_selected)
        lay.addWidget(self.delete_btn)
        return w

    def rescan(self) -> None:
        self.detail.setPlainText("Scanning…")

        def work():
            self.catalog.dirs = self.cfg.model_dirs
            self.catalog.scan(progress=lambda m: self.statusLine.emit(m))
            self.ui(self._refilter)
            self.statusLine.emit(f"{len(self.catalog.entries)} model(s) found")
            if not self.catalog.entries:
                self.ui(lambda: self.detail.setPlainText(
                    "No GGUF files found.\n\nAdd the folder your models live in, or "
                    "fetch one from the Download tab. Kestrel also reads LM Studio's "
                    "and huggingface-cli's usual locations automatically."))

        threading.Thread(target=work, daemon=True).start()

    def is_favourite(self, path) -> bool:
        return str(path) in (self.cfg.favourite_models or [])

    def toggle_favourite(self) -> None:
        """Star a model, and keep the starred ones at the top.

        A model library grows quickly and the two or three actually in use are
        otherwise lost among the rest.
        """
        entry = self._entry_for(self.tree.currentItem())
        if entry is None:
            return
        path = str(entry.path)
        favourites = list(self.cfg.favourite_models or [])
        if path in favourites:
            favourites.remove(path)
            self.statusLine.emit(f"Removed {entry.name} from favourites")
        else:
            favourites.append(path)
            self.statusLine.emit(f"{entry.name} added to favourites")
        self.cfg.favourite_models = favourites
        self.cfg.save()
        self._refilter()

    def _refilter(self) -> None:
        entries = list(self.catalog.find(self.filter.text()))
        # Starred first, then whatever order the catalogue gave.
        entries.sort(key=lambda e: (not self.is_favourite(e.path),))
        self.tree.clear()
        for e in entries:
            starred = self.is_favourite(e.path)
            item = QTreeWidgetItem(["★" if starred else ""] + list(e.row()))
            item.setData(0, Qt.UserRole, str(e.path))
            item.setToolTip(1, str(e.path))
            if starred:
                item.setForeground(0, QColor(theme.AMBER))
            if e.info.error:
                item.setForeground(1, QColor(theme.ALERT))
            elif str(e.path) == self.cfg.model_path:
                item.setForeground(1, QColor(theme.AMBER))
            self.tree.addTopLevelItem(item)
        for column in range(1, self.tree.columnCount()):
            self.tree.resizeColumnToContents(column)
        if self.tree.topLevelItemCount() and self.tree.currentItem() is None:
            match = next((i for i in range(self.tree.topLevelItemCount())
                          if self.tree.topLevelItem(i).data(0, Qt.UserRole)
                          == self.cfg.model_path), 0)
            self.tree.setCurrentItem(self.tree.topLevelItem(match))

    def _entry_for(self, item):
        if item is None:
            return None
        path = item.data(0, Qt.UserRole)
        return next((e for e in self.catalog.entries if str(e.path) == path), None)

    def _show_detail(self, item, _prev=None) -> None:
        e = self._entry_for(item)
        if e is None:
            return
        ctx = self.cfg.runtime.ctx_size or 8192
        need = estimate_vram_mb(e.info, ctx)
        pooled = sum(n.mem_mb for n in self.cfg.active_nodes())
        lines = [e.info.summary(), "", f"path           {e.path}"]
        if e.repo:
            lines.append(f"repo           {e.repo}")
        kv = gguf_kv_bytes(e.info, ctx) / 1024 ** 3
        lines.append(f"needs ~{need / 1024:.1f} GB at {ctx:,} ctx "
                     f"({e.info.size_gb:.1f} GB weights + {kv:.1f} GB KV cache)")
        try:
            from ..runtime import resolve_gpu_layers
            if self.cfg.runtime.n_gpu_layers < 0 and e.info.n_layer:
                fits = resolve_gpu_layers(self.cfg, str(e.path))
                where = (f"{fits} of {e.info.n_layer} layers on the GPU, the rest "
                         "in system RAM" if fits else
                         "entirely in system RAM — no GPU memory detected")
                lines.append(f"split          {where}")
        except Exception:
            pass
        if pooled:
            lines.append(f"cluster pool   {pooled:,} MB across {len(self.cfg.active_nodes())} "
                         f"worker(s) — {'fits' if need <= pooled else 'too large, add memory'}")
        if e.info.n_ctx_train and ctx > e.info.n_ctx_train:
            lines.append(f"warning: {ctx:,} exceeds the trained context "
                         f"({e.info.n_ctx_train:,}); set rope scaling or lower it")
        self.detail.setPlainText("\n".join(lines))

    def load_selected(self) -> None:
        e = self._entry_for(self.tree.currentItem())
        if e is None:
            QMessageBox.information(self, "No model", "Pick a model from the list first.")
            return
        if e.info.parts > 1:
            self.logLine.emit(f"[models] {e.name} is part 1 of {e.info.parts}; "
                              "llama.cpp will pick up the rest automatically")
        self.cfg.model_path = str(e.path)
        self.cfg.save()
        self._refilter()
        self.loadRequested.emit(str(e.path))

    def delete_selected(self) -> None:
        entry = self._entry_for(self.tree.currentItem())
        if entry is None:
            QMessageBox.information(self, "No model", "Pick a model from the list.")
            return
        parts = (f"\n\nThis model is in {entry.info.parts} parts; all of them will "
                 "be deleted." if entry.info.parts > 1 else "")
        if QMessageBox.question(
                self, "Delete model",
                f"Permanently delete this file?\n\n{entry.path}\n"
                f"{entry.info.size_gb:.1f} GB{parts}\n\nThis cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if str(entry.path) == self.cfg.model_path:
            # Deleting the file underneath a running server leaves it serving a
            # model that no longer exists, so the reference goes too.
            self.cfg.model_path = ""
            self.cfg.save()
            self.unloadRequested.emit()
        removed, failures = modelsmod.delete_model(entry)
        if failures:
            QMessageBox.warning(self, "Could not delete everything",
                                "\n".join(failures[:6]))
        self.statusLine.emit(f"Deleted {len(removed)} file(s)")
        self.logLine.emit(f"[models] deleted {entry.path}")
        self.rescan()

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Models folder", str(Path.home()))
        if d and d not in self.cfg.model_dirs:
            self.cfg.model_dirs.append(d)
            self.cfg.save()
            self.rescan()

# =========================================================== params panel ===
class ParamsPanel(QWidget):
    """Everything adjustable, in one place.

    Settings used to be split between a dialog and the side panels, which meant
    knowing which half a given option lived in. They are all here now, grouped
    by when they take effect: sampling and reasoning apply to the next message,
    runtime needs the model reloaded, agent and appearance are immediate.
    """

    statusLine = Signal(str)
    appearanceChanged = Signal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        tabs = QTabWidget()
        tabs.addTab(_scroll(self._sampling_tab()), "Sampling")
        tabs.addTab(_scroll(self._thinking_tab()), "Thinking")
        tabs.addTab(_scroll(self._runtime_tab()), "Runtime")
        outer.addWidget(tabs)

    # -- runtime (load-time; needs a reload to take effect) -------------------
    def _runtime_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        rt = self.cfg.runtime

        note = QLabel("Load-time settings. Changing any of these needs the model "
                      "reloaded before it takes effect.")
        note.setWordWrap(True)
        note.setObjectName("Dim")
        lay.addWidget(note)

        form = _form()

        self.rt_ctx = QSpinBox(); self.rt_ctx.setRange(0, 4_000_000)
        self.rt_ctx.setSingleStep(1024); self.rt_ctx.setValue(rt.ctx_size)
        form.addRow("Context length", self.rt_ctx)

        self.rt_budget = QSpinBox()
        self.rt_budget.setRange(0, 262144)
        self.rt_budget.setSingleStep(512)
        self.rt_budget.setSpecialValueText("auto")
        self.rt_budget.setToolTip("auto asks the device how much memory it has")
        self.rt_budget.setValue(rt.gpu_budget_mb)
        self.rt_budget.setSuffix(" MB")
        form.addRow("GPU memory budget", self.rt_budget)
        self.budget_note = QLabel("")
        self.budget_note.setObjectName("Dim")
        self.budget_note.setWordWrap(True)
        form.addRow("", self.budget_note)

        self.rt_nkvo = QCheckBox("Keep the KV cache in system RAM")
        self.rt_nkvo.setToolTip("Frees graphics memory for the weights. The "
                                "cache is read every token, so this costs some "
                                "speed — but a model that fits is faster than "
                                "one that does not run.")
        self.rt_nkvo.setChecked(rt.no_kv_offload)
        form.addRow("", self.rt_nkvo)

        self.rt_cpumoe = QCheckBox("Keep mixture-of-experts weights on the CPU")
        self.rt_cpumoe.setToolTip("Only useful for MoE models. A fraction of "
                                  "the experts run per token, so the idle ones "
                                  "are the cheapest thing to move off the GPU.")
        self.rt_cpumoe.setChecked(rt.cpu_moe)
        form.addRow("", self.rt_cpumoe)

        self.split = GpuSplit()
        self.split.changed.connect(lambda v: self.rt_ngl.setValue(v))
        form.addRow("Split between GPU and system RAM", self.split)

        self.rt_ngl = QSpinBox()
        self.rt_ngl.setRange(-1, 999)
        self.rt_ngl.setSpecialValueText("auto")
        self.rt_ngl.setToolTip("auto offloads as many layers as the device holds")
        self.rt_ngl.setValue(rt.n_gpu_layers)
        self.rt_ngl.valueChanged.connect(
            lambda v: self.split.set_value(v) if v >= 0 else None)
        auto = QPushButton("Auto")
        auto.setToolTip("Offload as many layers as the device can hold")
        auto.clicked.connect(self._auto_split)
        refresh = QPushButton("Recalculate")
        refresh.setToolTip("Re-read the selected model and the device memory")
        refresh.clicked.connect(self.refresh_split)
        form.addRow("GPU layers  (0 = CPU only)", _row(self.rt_ngl, auto, refresh))

        self.rt_threads = QSpinBox(); self.rt_threads.setRange(0, 256); self.rt_threads.setValue(rt.threads)
        form.addRow("Threads", self.rt_threads)

        self.rt_batch = QSpinBox(); self.rt_batch.setRange(0, 65536)
        self.rt_batch.setSingleStep(128); self.rt_batch.setValue(rt.batch_size)
        self.rt_ubatch = QSpinBox(); self.rt_ubatch.setRange(0, 65536)
        self.rt_ubatch.setSingleStep(128); self.rt_ubatch.setValue(rt.ubatch_size)
        form.addRow("Batch / micro-batch", _row(self.rt_batch, self.rt_ubatch))

        self.rt_fa = QComboBox(); self.rt_fa.addItems(["auto", "on", "off"])
        self.rt_fa.setCurrentText(rt.flash_attn)
        form.addRow("Flash attention", self.rt_fa)

        self.rt_ck = QComboBox(); self.rt_ck.addItems(CACHE_TYPES); self.rt_ck.setCurrentText(rt.cache_type_k)
        self.rt_cv = QComboBox(); self.rt_cv.addItems(CACHE_TYPES); self.rt_cv.setCurrentText(rt.cache_type_v)
        form.addRow("KV cache K / V", _row(self.rt_ck, self.rt_cv))

        self.rt_mmap = QCheckBox("Disable mmap"); self.rt_mmap.setChecked(rt.no_mmap)
        self.rt_mlock = QCheckBox("Lock in RAM"); self.rt_mlock.setChecked(rt.mlock)
        self.rt_nkvo = QCheckBox("Keep KV on CPU"); self.rt_nkvo.setChecked(rt.no_kv_offload)
        form.addRow("Memory", _row(self.rt_mmap, self.rt_mlock, self.rt_nkvo))

        self.rt_parallel = QSpinBox(); self.rt_parallel.setRange(0, 64); self.rt_parallel.setValue(rt.parallel)
        form.addRow("Parallel slots", self.rt_parallel)

        self.rt_sm = QComboBox(); self.rt_sm.addItems(SPLIT_MODES); self.rt_sm.setCurrentText(rt.split_mode)
        self.rt_mg = QSpinBox(); self.rt_mg.setRange(0, 16); self.rt_mg.setValue(rt.main_gpu)
        form.addRow("Split mode / main GPU", _row(self.rt_sm, self.rt_mg))

        self.rt_ts = QLineEdit(rt.tensor_split)
        self.rt_ts.setPlaceholderText("blank = derived from cluster node memory")
        form.addRow("Tensor split", self.rt_ts)

        self.rt_rope = QComboBox(); self.rt_rope.addItems(ROPE_SCALING); self.rt_rope.setCurrentText(rt.rope_scaling)
        self.rt_rope_base = QDoubleSpinBox(); self.rt_rope_base.setRange(0, 10_000_000)
        self.rt_rope_base.setDecimals(0); self.rt_rope_base.setValue(rt.rope_freq_base)
        self.rt_rope_scale = QDoubleSpinBox(); self.rt_rope_scale.setRange(0, 100)
        self.rt_rope_scale.setSingleStep(0.1); self.rt_rope_scale.setValue(rt.rope_freq_scale)
        form.addRow("RoPE scale / base / factor",
                    _row(self.rt_rope, self.rt_rope_base, self.rt_rope_scale))

        self.rt_tpl = QLineEdit(rt.chat_template)
        self.rt_tpl.setPlaceholderText("blank = use the template in the GGUF")
        form.addRow("Chat template", self.rt_tpl)

        self.rt_jinja = QCheckBox("Use --jinja (needed for native tool calls)")
        self.rt_jinja.setChecked(rt.jinja)
        form.addRow("", self.rt_jinja)

        self.rt_host = QLineEdit(rt.host)
        self.rt_port = QSpinBox(); self.rt_port.setRange(1, 65535); self.rt_port.setValue(rt.port)
        form.addRow("Host / port", _row(self.rt_host, self.rt_port))

        self.rt_extra = QLineEdit(rt.extra_args)
        self.rt_extra.setPlaceholderText("any other llama-server flags")
        form.addRow("Extra arguments", self.rt_extra)

        self.rt_dl = QLineEdit(self.cfg.download_dir)
        pick = QPushButton("…")
        pick.setMaximumWidth(34)
        pick.clicked.connect(self._pick_download_dir)
        form.addRow("Download folder", _row(self.rt_dl, pick, stretch_last=False))

        lay.addLayout(form)
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply_runtime)
        lay.addWidget(apply_btn)
        self.cmd_preview = QTextEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setFont(mono_font(9))
        self.cmd_preview.setMaximumHeight(90)
        lay.addWidget(Field("command preview", self.cmd_preview))
        lay.addStretch(1)
        self.refresh_preview()
        self.refresh_split()
        return w

    def _auto_split(self) -> None:
        self.rt_ngl.setValue(-1)
        self.refresh_split()
        self.statusLine.emit(f"Auto: {self.split.fits} of {self.split.n_layer} "
                             "layers on the GPU")

    def refresh_split(self) -> None:
        """Re-read the model and the device so the bar reflects both.

        Done on demand rather than continuously: it parses a GGUF header and
        asks the driver how much memory it has, neither of which is worth doing
        on every repaint.
        """
        from .. import gguf, sysmon
        from ..runtime import resolve_gpu_layers

        info = None
        if self.cfg.model_path:
            try:
                info = gguf.read(self.cfg.model_path, want_template=False)
            except Exception:
                info = None
        vram = system = 0
        integrated = False
        try:
            monitor = sysmon.Monitor()
            gpus = monitor.gpus()
            if gpus:
                best = max(gpus, key=lambda g: g.budget_mb)
                vram, integrated = best.budget_mb, best.integrated
            system = monitor.sample().mem_total_mb
        except Exception:
            pass
        if gpus:
            self.budget_note.setText(
                "  ·  ".join(f"{g.name}: {g.memory_summary()}" for g in gpus[:2]))
        else:
            self.budget_note.setText("No GPU counters available.")
        vram = self.cfg.runtime.gpu_budget_mb or vram
        current = self.cfg.runtime.n_gpu_layers
        if current < 0:
            current = resolve_gpu_layers(self.cfg) if self.cfg.model_path else 0
        self.split.set_model(info, self.cfg.runtime.ctx_size or 4096,
                             vram, system, integrated, current)

    def _pick_download_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Download folder", self.rt_dl.text())
        if d:
            self.rt_dl.setText(d)

    def apply_runtime(self) -> None:
        rt = self.cfg.runtime
        rt.ctx_size = self.rt_ctx.value()
        rt.n_gpu_layers = self.rt_ngl.value()
        rt.threads = self.rt_threads.value()
        rt.batch_size = self.rt_batch.value()
        rt.ubatch_size = self.rt_ubatch.value()
        rt.flash_attn = self.rt_fa.currentText()
        rt.cache_type_k = self.rt_ck.currentText()
        rt.cache_type_v = self.rt_cv.currentText()
        rt.no_mmap = self.rt_mmap.isChecked()
        rt.mlock = self.rt_mlock.isChecked()
        rt.no_kv_offload = self.rt_nkvo.isChecked()
        rt.parallel = self.rt_parallel.value()
        rt.split_mode = self.rt_sm.currentText()
        rt.main_gpu = self.rt_mg.value()
        rt.tensor_split = self.rt_ts.text().strip()
        rt.gpu_budget_mb = self.rt_budget.value()
        rt.no_kv_offload = self.rt_nkvo.isChecked()
        rt.cpu_moe = self.rt_cpumoe.isChecked()
        rt.rope_scaling = self.rt_rope.currentText()
        rt.rope_freq_base = self.rt_rope_base.value()
        rt.rope_freq_scale = self.rt_rope_scale.value()
        rt.chat_template = self.rt_tpl.text().strip()
        rt.jinja = self.rt_jinja.isChecked()
        rt.host = self.rt_host.text().strip() or "127.0.0.1"
        rt.port = self.rt_port.value()
        rt.extra_args = self.rt_extra.text()
        self.cfg.download_dir = self.rt_dl.text().strip()
        self.cfg.server_url = rt.url()
        self.cfg.save()
        self.refresh_preview()
        self.statusLine.emit("Runtime settings saved — reload the model to apply")

    def refresh_preview(self) -> None:
        try:
            from ..cluster import build_command
            self.cmd_preview.setPlainText(" ".join(build_command(self.cfg)))
        except Exception as e:
            self.cmd_preview.setPlainText(f"({e})")

    def _sampling_tab(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        s, t = self.cfg.sampling, self.cfg.thinking

        note = QLabel("Sampling applies to the next message — no reload needed.")
        note.setWordWrap(True)
        note.setObjectName("Dim")
        lay.addWidget(note)

        presets = QHBoxLayout()
        for name in ("deterministic", "precise", "balanced", "creative"):
            b = QPushButton(name.capitalize())
            b.clicked.connect(lambda _=False, n=name: self._preset(n))
            presets.addWidget(b)
        lay.addLayout(presets)

        form = _form()

        def dspin(val, lo, hi, step=0.05, dec=2):
            x = QDoubleSpinBox(); x.setRange(lo, hi); x.setSingleStep(step)
            x.setDecimals(dec); x.setValue(val); return x

        def ispin(val, lo, hi):
            x = QSpinBox(); x.setRange(lo, hi); x.setValue(val); return x

        self.temp = dspin(s.temperature, 0, 5)
        self.top_k = ispin(s.top_k, 0, 1000)
        form.addRow("Temperature / top-k", _row(self.temp, self.top_k))
        self.top_p = dspin(s.top_p, 0, 1)
        self.min_p = dspin(s.min_p, 0, 1, 0.01, 3)
        form.addRow("Top-p / min-p", _row(self.top_p, self.min_p))
        self.typical = dspin(s.typical_p, 0, 1)
        form.addRow("Typical-p", self.typical)
        self.rep = dspin(s.repeat_penalty, 0.5, 3, 0.01)
        self.rep_n = ispin(s.repeat_last_n, -1, 8192)
        form.addRow("Repeat penalty / window", _row(self.rep, self.rep_n))
        self.presence = dspin(s.presence_penalty, -2, 2)
        self.frequency = dspin(s.frequency_penalty, -2, 2)
        form.addRow("Presence / frequency", _row(self.presence, self.frequency))
        self.mirostat = QComboBox(); self.mirostat.addItems(["off", "v1", "v2"])
        self.mirostat.setCurrentIndex(s.mirostat)
        self.mtau = dspin(s.mirostat_tau, 0, 20, 0.1)
        self.meta = dspin(s.mirostat_eta, 0, 1, 0.01)
        form.addRow("Mirostat / tau / eta", _row(self.mirostat, self.mtau, self.meta))
        self.dry = dspin(s.dry_multiplier, 0, 5, 0.1)
        self.dry_base = dspin(s.dry_base, 0, 5, 0.05)
        self.dry_len = ispin(s.dry_allowed_length, 0, 64)
        form.addRow("DRY mult / base / length", _row(self.dry, self.dry_base, self.dry_len))
        self.xtc_p = dspin(s.xtc_probability, 0, 1, 0.05)
        self.xtc_t = dspin(s.xtc_threshold, 0, 1, 0.01)
        form.addRow("XTC probability / threshold", _row(self.xtc_p, self.xtc_t))
        self.seed = ispin(s.seed, -1, 2_000_000_000)
        form.addRow("Seed (-1 random)", self.seed)
        lay.addLayout(form)

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply)
        lay.addWidget(apply_btn)
        lay.addStretch(1)
        return inner

    def _thinking_tab(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        t = self.cfg.thinking

        def ispin(val, lo, hi):
            x = QSpinBox(); x.setRange(lo, hi); x.setValue(val); return x

        tnote = QLabel("A reasoning model can spend thousands of tokens before its "
                       "first tool call. Kestrel reserves room for that, and never "
                       "sends an old trace back — so capping it here directly buys "
                       "transcript space.")
        tnote.setWordWrap(True)
        tnote.setObjectName("Dim")
        lay.addWidget(tnote)

        tform = _form()
        self.th_mode = QComboBox(); self.th_mode.addItems(["auto", "on", "off"])
        self.th_mode.setCurrentText(t.mode)
        tform.addRow("Mode", self.th_mode)
        self.th_budget = ispin(t.budget, 0, 100_000)
        tform.addRow("Token budget (0 = uncapped)", self.th_budget)
        self.th_fmt = QComboBox(); self.th_fmt.addItems(REASONING_FORMATS)
        self.th_fmt.setCurrentText(t.reasoning_format)
        tform.addRow("Reasoning format", self.th_fmt)
        self.th_effort = QComboBox(); self.th_effort.addItems(["", "low", "medium", "high"])
        self.th_effort.setCurrentText(t.effort)
        tform.addRow("Effort", self.th_effort)
        self.th_show = QCheckBox("Show the trace"); self.th_show.setChecked(t.show)
        self.th_keep = QCheckBox("Resend past traces"); self.th_keep.setChecked(t.keep_in_history)
        tform.addRow("", _row(self.th_show, self.th_keep))
        lay.addLayout(tform)

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("Primary")
        apply_btn.clicked.connect(self.apply)
        lay.addWidget(apply_btn)
        lay.addStretch(1)
        return inner

    def _preset(self, name: str) -> None:
        self.cfg.sampling.preset(name)
        s = self.cfg.sampling
        self.temp.setValue(s.temperature)
        self.top_k.setValue(s.top_k)
        self.top_p.setValue(s.top_p)
        self.min_p.setValue(s.min_p)
        self.rep.setValue(s.repeat_penalty)
        self.apply()

    def apply(self) -> None:
        s, t = self.cfg.sampling, self.cfg.thinking
        s.temperature = self.temp.value(); s.top_k = self.top_k.value()
        s.top_p = self.top_p.value(); s.min_p = self.min_p.value()
        s.typical_p = self.typical.value()
        s.repeat_penalty = self.rep.value(); s.repeat_last_n = self.rep_n.value()
        s.presence_penalty = self.presence.value()
        s.frequency_penalty = self.frequency.value()
        s.mirostat = self.mirostat.currentIndex()
        s.mirostat_tau = self.mtau.value(); s.mirostat_eta = self.meta.value()
        s.dry_multiplier = self.dry.value(); s.dry_base = self.dry_base.value()
        s.dry_allowed_length = self.dry_len.value()
        s.xtc_probability = self.xtc_p.value(); s.xtc_threshold = self.xtc_t.value()
        s.seed = self.seed.value()
        t.mode = self.th_mode.currentText(); t.budget = self.th_budget.value()
        t.reasoning_format = self.th_fmt.currentText()
        t.effort = self.th_effort.currentText()
        t.show = self.th_show.isChecked(); t.keep_in_history = self.th_keep.isChecked()
        self.cfg.save()
        self.statusLine.emit("Parameters applied")


# =========================================================== memory panel ===
class MemoryPanel(QWidget):
    statusLine = Signal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.store: MemoryStore | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)

        blurb = QLabel("Carried between sessions. Relevant entries are injected "
                       "each turn; pinned ones always are.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Dim")
        lay.addWidget(blurb)

        # Three kinds of memory, kept apart because they have different lives:
        # a project fact dies with the project, what Kestrel knows about the
        # machine does not, and what it knows about you belongs to you.
        self.tier_tabs = QTabBar()
        self.tier_tabs.setExpanding(False)
        for label, key in (("Project", memorymod.PROJECT),
                           ("Global", memorymod.GLOBAL),
                           ("Personal", memorymod.PERSONAL)):
            index = self.tier_tabs.addTab(label)
            self.tier_tabs.setTabData(index, key)
            self.tier_tabs.setTabToolTip(index, memorymod.TIER_HELP[key])
        self.tier_tabs.currentChanged.connect(lambda _i: self.refresh())
        lay.addWidget(self.tier_tabs)

        self.tier_note = QLabel("")
        self.tier_note.setObjectName("Dim")
        self.tier_note.setWordWrap(True)
        lay.addWidget(self.tier_note)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search memories…")
        self.search.textChanged.connect(self.refresh)
        lay.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["", "Kind", "Memory"])
        self.tree.setObjectName("Flush")
        self.tree.setFont(mono_font(10))
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        stretch_columns(self.tree, first_stretch=2)
        self.tree.itemDoubleClicked.connect(lambda *_: self.edit())
        lay.addWidget(self.tree, 1)

        r1 = QHBoxLayout()
        for text, slot in (("Add…", self.add), ("Edit…", self.edit),
                           ("Pin", self.toggle_pin), ("Delete", self.delete)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            r1.addWidget(b)
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        for text, slot in (("Export…", self.export), ("Import…", self.do_import),
                           ("Clear all", self.clear)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            r2.addWidget(b)
        lay.addLayout(r2)

        self.enabled = QCheckBox("Long-term memory enabled")
        self.enabled.setChecked(cfg.memory.enabled)
        self.enabled.toggled.connect(self._toggle_enabled)
        self.capture = QCheckBox("Capture facts automatically after each task")
        self.capture.setChecked(cfg.memory.auto_capture)
        self.capture.toggled.connect(self._toggle_capture)
        self.global_scope = QCheckBox("Share one pool across all workspaces")
        self.global_scope.setChecked(cfg.memory.global_scope)
        self.global_scope.toggled.connect(self._toggle_scope)
        for c in (self.enabled, self.capture, self.global_scope):
            lay.addWidget(c)

        self.count_label = QLabel("")
        self.count_label.setObjectName("Dim")
        lay.addWidget(self.count_label)
        self.reopen()

    # -- store ---------------------------------------------------------------
    def purge(self) -> None:
        """Remove memories the durability filter would refuse today."""
        if self.store is None:
            return
        removed = self.store.purge_ephemeral(
            scope=None if self.cfg.memory.scope == "global" else self.cfg.workspace)
        self.refresh()
        if removed:
            QMessageBox.information(
                self, "Cleaned up",
                f"Removed {len(removed)} memory(ies) that would not be stored "
                "today:\n\n" + "\n".join(f"· {t[:80]}" for t in removed[:12]))
        self.statusLine.emit(f"Removed {len(removed)} stale memory(ies)")

    def reopen(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        if self.cfg.memory.enabled:
            try:
                self.store = MemoryStore(self.cfg.memory.db_path, self.cfg.memory_scope())
            except Exception as e:
                self.statusLine.emit(f"Could not open memory store: {e}")
        self.refresh()

    def current_tier(self) -> str:
        return self.tier_tabs.tabData(self.tier_tabs.currentIndex()) or ""

    def refresh(self) -> None:
        self.tree.clear()
        if self.store is None:
            self.count_label.setText("Memory is switched off.")
            return
        tier = self.current_tier()
        self.tier_note.setText(memorymod.TIER_HELP.get(tier, ""))
        query = self.search.text().strip()
        items = (self.store.search(query, limit=200, include_pinned=False)
                 if query else self.store.all(tier=tier, limit=400))
        if query and tier:
            items = [m for m in items if m.tier == tier]
        for m in items:
            it = QTreeWidgetItem(["*" if m.pinned else "", m.kind, m.text])
            it.setData(0, Qt.UserRole, m.id)
            it.setToolTip(2, f"#{m.id}  importance {m.importance}  used {m.uses}x\n{m.text}")
            if m.pinned:
                it.setForeground(0, QColor(theme.AMBER))
            self.tree.addTopLevelItem(it)
        total = self.store.count()
        self.count_label.setText(
            f"{len(items)} shown of {total} · "
            f"{'shared pool' if self.cfg.memory.global_scope else 'this workspace'} · "
            f"{self.cfg.memory.db_path}")

    def _selected_id(self) -> int | None:
        it = self.tree.currentItem()
        return it.data(0, Qt.UserRole) if it else None

    # -- actions -------------------------------------------------------------
    def add(self) -> None:
        if self.store is None:
            return
        text, ok = QInputDialog.getText(self, "New memory", "Something worth remembering:")
        if not ok or not text.strip():
            return
        kind, ok = QInputDialog.getItem(self, "Kind", "Type of memory:", KINDS, 0, False)
        if not ok:
            return
        # Added while looking at a tier, so that is where it goes: the tab is
        # the answer to "which kind of memory is this".
        self.store.remember(text.strip(), kind, importance=4, source="user",
                            tier=self.current_tier(), enforce=False)
        self.refresh()

    def edit(self) -> None:
        mid = self._selected_id()
        if mid is None or self.store is None:
            return
        current = next((m for m in self.store.all(limit=10000) if m.id == mid), None)
        if current is None:
            return
        text, ok = QInputDialog.getText(self, "Edit memory", "Memory:", text=current.text)
        if ok and text.strip():
            self.store.update(mid, text=text.strip())
            self.refresh()

    def toggle_pin(self) -> None:
        mid = self._selected_id()
        if mid is None or self.store is None:
            return
        current = next((m for m in self.store.all(limit=10000) if m.id == mid), None)
        if current:
            self.store.set_pinned(mid, not current.pinned)
            self.refresh()

    def delete(self) -> None:
        mid = self._selected_id()
        if mid is None or self.store is None:
            return
        self.store.forget(int(mid))
        self.refresh()

    def clear(self) -> None:
        if self.store is None:
            return
        if QMessageBox.question(self, "Clear memory",
                                "Delete every memory in this scope?") == QMessageBox.Yes:
            n = self.store.clear()
            self.statusLine.emit(f"Deleted {n} memories")
            self.refresh()

    def export(self) -> None:
        if self.store is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export memories",
                                              "memories.jsonl", "JSON Lines (*.jsonl)")
        if path:
            Path(path).write_text(self.store.export(), "utf-8")
            self.statusLine.emit(f"Exported to {path}")

    def do_import(self) -> None:
        if self.store is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import memories", "",
                                              "JSON Lines (*.jsonl);;All files (*)")
        if path:
            n = self.store.import_jsonl(Path(path).read_text("utf-8", errors="replace"))
            self.statusLine.emit(f"Imported {n} new memories")
            self.refresh()

    def _toggle_enabled(self, on: bool) -> None:
        self.cfg.memory.enabled = on
        self.cfg.save()
        self.reopen()

    def _toggle_capture(self, on: bool) -> None:
        self.cfg.memory.auto_capture = on
        self.cfg.save()

    def _toggle_scope(self, on: bool) -> None:
        self.cfg.memory.global_scope = on
        self.cfg.save()
        self.reopen()


# ============================================================= plan panel ===
class PlanPanel(QWidget):
    """Live view of the model's checklist, and the controls for editing it.

    This is not a progress bar Kestrel invents — it is exactly the block being
    sent to the model each turn, so editing it here changes what the model is
    working from on the next step.
    """

    pauseToggled = Signal(bool)
    planEdited = Signal()
    statusLine = Signal(str)
    needTodo = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.todo = None
        self._paused = False
        self._running = False
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)

        self.head = QLabel("No plan yet.")
        self.head.setObjectName("Readout")
        self.head.setWordWrap(True)
        lay.addWidget(self.head)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setMaximumHeight(6)
        lay.addWidget(self.bar)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["", "Step"])
        self.tree.setFont(mono_font(10))
        self.tree.setRootIsDecorated(False)
        self.tree.setWordWrap(True)
        self.tree.setUniformRowHeights(False)
        self.tree.setItemDelegateForColumn(1, WrappingDelegate(self.tree))
        self.tree.header().sectionResized.connect(
            lambda *_: self.tree.scheduleDelayedItemsLayout())
        # Steps are edited where they are read: double-click the text, or press
        # F2. A plan is a working document, and retyping it through a dialog to
        # fix a word is friction the model does not have.
        self.tree.setEditTriggers(QTreeWidget.DoubleClicked | QTreeWidget.EditKeyPressed)
        self.tree.itemChanged.connect(self._step_edited)
        stretch_columns(self.tree, first_stretch=1)
        lay.addWidget(self.tree, 1)

        self.done_box = QWidget()
        done_lay = QVBoxLayout(self.done_box)
        done_lay.setContentsMargins(0, 0, 0, 0)
        finished = QLabel("Every step is closed. The next task starts a new plan, "
                          "or write one now with Add step…")
        finished.setWordWrap(True)
        finished.setObjectName("Dim")
        done_lay.addWidget(finished)
        show_btn = QPushButton("Show the finished steps")
        show_btn.clicked.connect(lambda: (self.tree.setVisible(True),
                                          self.done_box.setVisible(False)))
        done_lay.addWidget(show_btn)
        clear_btn = QPushButton("Clear and start fresh")
        clear_btn.setObjectName("Primary")
        clear_btn.clicked.connect(self.clear_all_now)
        done_lay.addWidget(clear_btn)
        done_lay.addStretch(1)
        self.done_box.hide()
        lay.addWidget(self.done_box, 1)

        self.pause_btn = QPushButton("Pause after this step")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setToolTip("Hold at the next step boundary so the plan can "
                                  "be edited safely")
        self.pause_btn.toggled.connect(self._on_pause)
        lay.addWidget(self.pause_btn)

        # Two rows rather than four across: this panel is often narrow, and a
        # single row clips the labels at any sensible width.
        buttons = [("Add step…", self.add_steps, "Write steps yourself, one per line"),
                   ("Working", lambda: self.set_status(DOING),
                    "Mark the selected step as being worked on"),
                   ("Done", lambda: self.set_status(DONE),
                    "Mark the selected step finished"),
                   ("To do", lambda: self.set_status(TODO),
                    "Put the selected step back to not started"),
                   ("Move up", lambda: self.move_step(-1),
                    "Move the selected step earlier"),
                   ("Move down", lambda: self.move_step(1),
                    "Move the selected step later"),
                   ("Remove", self.remove_step, "Delete the selected step"),
                   ("Clear all", self.clear_all, "Discard the whole plan")]
        for pair in (buttons[:3], buttons[3:6], buttons[6:]):
            row = QHBoxLayout()
            row.setSpacing(6)
            for text, slot, tip in pair:
                b = QPushButton(text)
                b.setToolTip(tip)
                b.setMinimumWidth(0)
                b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                b.clicked.connect(slot)
                row.addWidget(b)
            lay.addLayout(row)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setObjectName("Dim")
        lay.addWidget(self.hint)

        note = QLabel("Re-sent with every prompt, so it survives compaction.")
        note.setWordWrap(True)
        note.setObjectName("Dim")
        lay.addWidget(note)

    def set_running(self, running: bool) -> None:
        self._running = running
        self._refresh_hint()

    def _on_pause(self, paused: bool) -> None:
        self._paused = paused
        self.pause_btn.setText("Paused — resume" if paused else "Pause after this step")
        self.pauseToggled.emit(paused)
        self._refresh_hint()

    def _refresh_hint(self) -> None:
        if self._paused:
            self.hint.setText("Paused at the next step boundary. Edit the plan, "
                              "then resume.")
        elif self._running:
            self.hint.setText("Running. Edits take effect from the next step.")
        else:
            self.hint.setText("")

    def _selected(self):
        item = self.tree.currentItem()
        return item.data(0, Qt.UserRole) if item else None

    def add_steps(self) -> None:
        """Write a plan by hand, one step per line.

        A model that will not decompose a task — small ones often will not —
        should not leave the checklist unusable. Steps added here behave exactly
        like generated ones: the model sees them in its next prompt and closes
        them as it goes.
        """
        if self.todo is None:
            self.needTodo.emit()
        if self.todo is None:
            QMessageBox.information(self, "Not connected",
                                    "Connect to a model first; the checklist "
                                    "belongs to the running session.")
            return
        text, ok = QInputDialog.getMultiLineText(
            self, "Add steps", "One step per line:", "")
        if not ok or not text.strip():
            return
        added = 0
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*+]|\[.\])\s*", "", line).strip()
            if cleaned:
                self.todo.add(cleaned)
                added += 1
        if added:
            self.update_todo(self.todo)
            self.planEdited.emit()
            self.statusLine.emit(f"Added {added} step(s)")

    def move_step(self, delta: int) -> None:
        if self.todo is None:
            return
        item_id = self._selected()
        if item_id is None:
            return
        if self.todo.move(int(item_id), delta):
            self.update_todo(self.todo)
            self.planEdited.emit()
            # Keep the moved step selected, or the next press moves its neighbour.
            for row in range(self.tree.topLevelItemCount()):
                node = self.tree.topLevelItem(row)
                if node.data(0, Qt.UserRole) == item_id:
                    self.tree.setCurrentItem(node)
                    break

    def set_status(self, status: str) -> None:
        """Move the selected step between to do, working and done.

        The model sets these as it goes; being able to set them by hand matters
        when it gets one wrong, or when the work happened outside Kestrel.
        """
        if self.todo is None:
            return
        item_id = self._selected()
        if item_id is None:
            return
        if self.todo.update(int(item_id), status):
            self.update_todo(self.todo)
            self.planEdited.emit()
            self.statusLine.emit(f"Step {item_id}: {status}")

    def remove_step(self) -> None:
        if self.todo is None:
            return
        item_id = self._selected()
        if item_id is not None and self.todo.remove(int(item_id)):
            self.update_todo(self.todo)
            self.planEdited.emit()

    def skip_step(self) -> None:
        if self.todo is None:
            return
        item_id = self._selected()
        if item_id is not None and self.todo.update(int(item_id), "done", "skipped"):
            self.update_todo(self.todo)
            self.planEdited.emit()

    def clear_done(self) -> None:
        if self.todo is None:
            return
        if self.todo.clear_done():
            self.update_todo(self.todo)
            self.planEdited.emit()

    def clear_all_now(self) -> None:
        """Clear without confirming: the plan is finished, so nothing is lost."""
        if self.todo is not None:
            self.todo.clear()
            self.update_todo(self.todo)
            self.planEdited.emit()

    def clear_all(self) -> None:
        if self.todo is None or not self.todo.items:
            return
        if QMessageBox.question(self, "Clear plan",
                                "Discard the whole checklist?") == QMessageBox.Yes:
            self.todo.clear()
            self.update_todo(self.todo)
            self.planEdited.emit()

    def _step_edited(self, item, column: int) -> None:
        """Write an edited step back to the checklist."""
        if column != 1 or self._loading or self.todo is None:
            return
        item_id = item.data(0, Qt.UserRole)
        if item_id is None:
            return
        text = item.text(1).split("  —")[0].strip()
        # The row shows "2a  Write the loop"; only the words are the step.
        text = re.sub(r"^\d+[a-z]?(?:\.\d+)?\s+", "", text).strip()
        if not text:
            self.update_todo(self.todo)      # refuse to blank a step
            return
        if self.todo.update(int(item_id), note="") and text:
            step = self.todo.get(int(item_id))
            if step is not None and step.text != text:
                step.text = text
                self.todo.save()
                self.planEdited.emit()
                self.statusLine.emit(f"Step {item_id} updated")

    def update_todo(self, todo) -> None:
        self.todo = todo
        self._loading = True
        self.tree.clear()
        if todo is None or not todo.items:
            self.head.setText("No plan yet — Add step… to write one.")
            self.bar.setMaximum(1)
            self.bar.setValue(0)
            self._loading = False
            return
        done, total = todo.progress
        self.bar.setMaximum(max(1, total))
        self.bar.setValue(done)

        # A finished plan collapses to one line. Leaving a wall of ticked steps
        # on screen implies there is still something to do, and the next task
        # starts from a clean panel.
        complete = todo.complete
        self.tree.setVisible(not complete)
        self.done_box.setVisible(complete)
        if complete:
            blocked = sum(1 for i in todo.items if i.status == BLOCKED)
            summary = f"Plan complete — {done} of {total} done"
            if blocked:
                summary += f", {blocked} blocked"
            self.head.setText(summary + (f"\n{todo.title}" if todo.title else ""))
            self.pause_btn.setChecked(False)
        else:
            self.head.setText(f"{done}/{total} done"
                              + (f" — {todo.title}" if todo.title else ""))
        current = todo.current
        for label, item, depth in todo.outline():
            indent = "      " * depth
            row = QTreeWidgetItem([DISPLAY_MARKS.get(item.status, "○"),
                                   f"{indent}{label}  {item.text}"
                                   + (f"  — {item.note}" if item.note else "")])
            row.setData(0, Qt.UserRole, item.id)
            if depth:
                row.setForeground(1, QColor(theme.TEXT_DIM))
            # The mark belongs beside the first line of a wrapped step, not
            # floating in the middle of the space the text needs.
            row.setTextAlignment(0, Qt.AlignTop | Qt.AlignHCenter)
            row.setTextAlignment(1, Qt.AlignTop | Qt.AlignLeft)
            row.setFlags(row.flags() | Qt.ItemIsEditable)
            row.setToolTip(1, f"step {label} · {item.status} — double-click to edit")
            colour = {DONE: theme.SIGNAL, DOING: theme.AMBER,
                      BLOCKED: theme.ALERT}.get(item.status, theme.TEXT_DIM)
            row.setForeground(0, QColor(colour))
            row.setForeground(1, QColor(theme.TEXT if item is current else theme.TEXT_DIM))
            self.tree.addTopLevelItem(row)
        # Rebuilding the tree fires itemChanged for every row; the guard stops
        # those being mistaken for the user typing.
        self._loading = False
        # Row heights are measured against the column width, which is not known
        # until the tree has been laid out — so ask for the measurement again
        # once it has been.
        self.tree.scheduleDelayedItemsLayout()


# ========================================================== persona panel ===
class PersonaPanel(QWidget):
    statusLine = Signal(str)
    personaChanged = Signal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.personas: list[personamod.Persona] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)

        blurb = QLabel("Compiled into tiers. Voice ships in every prompt; the "
                       "full background stays on disk until asked for.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Dim")
        lay.addWidget(blurb)

        self.list = QListWidget()
        self.list.setObjectName("Flush")
        self.list.currentRowChanged.connect(self._show)
        lay.addWidget(self.list, 1)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(mono_font(10))
        lay.addWidget(self.preview, 1)

        self.level = QComboBox()
        self.level.addItems(["follow context profile", "0 — one line", "1 — voice + 2 rules",
                             "2 — voice, traits, rules", "3 — everything distilled"])
        self.level.setCurrentIndex(0 if cfg.persona_level < 0 else cfg.persona_level + 1)
        self.level.currentIndexChanged.connect(self._set_level)
        lay.addWidget(Field("injected detail", self.level))

        row = QHBoxLayout()
        use = QPushButton("Use this")
        use.setObjectName("Primary")
        use.clicked.connect(self.use_selected)
        clear = QPushButton("None")
        clear.clicked.connect(self.clear_persona)
        add = QPushButton("Load file…")
        add.clicked.connect(self.load_file)
        for b in (use, clear, add):
            row.addWidget(b)
        lay.addLayout(row)
        self.rescan()

    def rescan(self) -> None:
        self.personas = personamod.discover(self.cfg.persona_dirs)
        current = personamod.load_file(self.cfg.persona_file) if self.cfg.persona_file else None
        if current and not any(p.path == current.path for p in self.personas):
            self.personas.insert(0, current)
        self.list.clear()
        for p in self.personas:
            self.list.addItem(p.name or (p.path.stem if p.path else "unnamed"))
        if not self.personas:
            self.preview.setPlainText(
                "No personas found.\n\nDrop a .md file into one of the persona folders, "
                "or load an existing SOUL.md — Kestrel will distil it into voice, traits "
                "and rules rather than sending the whole thing every turn.")
        active = self.cfg.persona_file
        for i, p in enumerate(self.personas):
            if p.path and str(p.path) == active:
                self.list.setCurrentRow(i)
                break

    def _show(self, row: int) -> None:
        if not (0 <= row < len(self.personas)):
            return
        p = self.personas[row]
        level = self.cfg.persona_level if self.cfg.persona_level >= 0 else 2
        lines = [p.summary(), "", f"--- injected at tier {level} ---", p.compile(level)]
        self.preview.setPlainText("\n".join(lines))

    def _set_level(self, idx: int) -> None:
        self.cfg.persona_level = idx - 1
        self.cfg.save()
        self._show(self.list.currentRow())

    def use_selected(self) -> None:
        row = self.list.currentRow()
        if not (0 <= row < len(self.personas)):
            return
        p = self.personas[row]
        self.cfg.persona_file = str(p.path) if p.path else ""
        self.cfg.save()
        self.personaChanged.emit()
        self.statusLine.emit(f"Persona: {p.name}")

    def clear_persona(self) -> None:
        self.cfg.persona_file = ""
        self.cfg.persona = ""
        self.cfg.save()
        self.personaChanged.emit()
        self.statusLine.emit("Persona cleared")

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Persona or SOUL.md", str(Path.home()),
                                              "Markdown (*.md);;All files (*)")
        if not path:
            return
        self.cfg.persona_file = path
        self.cfg.save()
        self.rescan()
        self.personaChanged.emit()
        self.statusLine.emit(f"Persona: {Path(path).name}")


# ========================================================== backend panel ===
class BackendPanel(UiThread, QWidget):
    """Find llama.cpp, or fetch it."""

    uiCall = Signal(object)
    logLine = Signal(str)
    statusLine = Signal(str)
    foundBinary = Signal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._init_ui_thread()
        self.cfg = cfg

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 0)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 4, 12, 8)
        lay.setSpacing(8)

        lay.addWidget(_heading("Where Kestrel connects"))
        self.url = QLineEdit(cfg.server_url)
        lay.addWidget(Field("Server URL", self.url,
                            "llama-server, LM Studio, llamafile, vLLM, or "
                            "Ollama's /v1."))
        self.api_key = QLineEdit(cfg.api_key)
        self.api_key.setEchoMode(QLineEdit.Password)
        lay.addWidget(Field("API key", self.api_key, "Blank for a local server."))
        self.hf_token = QLineEdit(cfg.hf_token)
        self.hf_token.setEchoMode(QLineEdit.Password)
        lay.addWidget(Field("Hugging Face token", self.hf_token,
                            "Only for gated repositories."))
        self.server_bin = QLineEdit(cfg.llama_server_bin)
        pick = QPushButton("Browse…")
        pick.clicked.connect(self._pick_binary)
        lay.addWidget(Field("llama-server binary", self.server_bin))
        save = QPushButton("Save endpoint")
        save.clicked.connect(self.save_endpoint)
        lay.addWidget(_row(pick, save))

        lay.addWidget(_heading("What was found on this machine"))
        self.r_server = Readout("llama-server", "not found")
        self.r_rpc = Readout("rpc-server", "not found")
        self.r_version = Readout("version", "—")
        for readout in (self.r_server, self.r_rpc, self.r_version):
            lay.addWidget(readout)
        scan = QPushButton("Scan again")
        scan.clicked.connect(self.scan)
        lay.addWidget(scan)

        lay.addWidget(_heading("Install llama.cpp"))
        blurb = QLabel("If llama.cpp is missing, Kestrel can fetch an official "
                       "build or compile one with RPC support so this machine "
                       "can also act as a cluster worker.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Dim")
        lay.addWidget(blurb)

        self.backend = QComboBox()
        self.backend.addItems(llamacpp.BACKENDS)
        self.backend.setCurrentText(cfg.llama_backend)
        self.backend.currentTextChanged.connect(self._set_backend)
        lay.addWidget(Field("Accelerator", self.backend,
                            f"auto detects: {llamacpp.detect_backend()}"))

        self.with_rpc = QCheckBox("Include the RPC backend (needed for clustering)")
        self.with_rpc.setChecked(cfg.llama_with_rpc)
        self.with_rpc.toggled.connect(self._set_rpc)
        lay.addWidget(self.with_rpc)
        self.reinstall = QCheckBox("Remove Kestrel's existing copy first")
        self.reinstall.setToolTip("Only removes what Kestrel installed; a system "
                                  "or package-manager build is left alone")
        lay.addWidget(self.reinstall)

        self.install_btn = QPushButton("Download official build")
        self.install_btn.setObjectName("Primary")
        self.install_btn.clicked.connect(lambda: self.install(source=False))
        self.build_btn = QPushButton("Build from source (with RPC)")
        self.build_btn.clicked.connect(lambda: self.install(source=True))
        lay.addWidget(self.install_btn)
        lay.addWidget(self.build_btn)
        self.remove_btn = QPushButton("Remove Kestrel's copy")
        self.remove_btn.setObjectName("Danger")
        self.remove_btn.clicked.connect(self.remove)
        lay.addWidget(self.remove_btn)

        self.output = QTextEdit()
        self.output.setObjectName("Flush")
        self.output.setReadOnly(True)
        self.output.setFont(mono_font(9))
        self.output.setMinimumHeight(120)
        lay.addWidget(self.output, 1)

        outer.addWidget(_scroll(inner))
        self.scan()

    def _pick_binary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "llama-server binary",
                                              str(Path.home()))
        if path:
            self.server_bin.setText(path)

    def save_endpoint(self) -> None:
        self.cfg.server_url = self.url.text().strip() or self.cfg.server_url
        self.cfg.api_key = self.api_key.text()
        self.cfg.hf_token = self.hf_token.text()
        self.cfg.llama_server_bin = self.server_bin.text().strip()
        self.cfg.save()
        self.statusLine.emit("Endpoint saved — reconnect to apply")
        self.scan()

    def _set_backend(self, value: str) -> None:
        self.cfg.llama_backend = value
        self.cfg.save()

    def _set_rpc(self, on: bool) -> None:
        self.cfg.llama_with_rpc = on
        self.cfg.save()

    def _say(self, message: str) -> None:
        self.ui(lambda: self.output.append(message))
        self.logLine.emit("[llama.cpp] " + message)

    def scan(self) -> None:
        def work():
            found = llamacpp.ensure(self.cfg, progress=self._say, allow_install=False)
            self.ui(lambda: self._apply(found))
        threading.Thread(target=work, daemon=True).start()

    def _apply(self, found) -> None:
        self.r_server.set(found.server or "not found")
        self.r_rpc.set(found.rpc or "not found")
        self.r_version.set(found.version or "—")
        self.remove_btn.setEnabled(bool(found.managed))
        if found.present and not found.working:
            # An installed-but-broken binary is the failure that looks like
            # nothing happened, so it is stated rather than left to be inferred.
            self.r_version.set("does not run")
            self._say(f"!! {found.server} does not run: {found.problem}")
            self._say("Remove Kestrel's copy and reinstall, or pick a different "
                      "accelerator — a CPU build always runs."
                      if found.managed else
                      "Kestrel did not install this one, so it will be left alone. "
                      "Install a fresh copy and Kestrel will prefer it.")
            self.reinstall.setChecked(bool(found.managed))
            self.statusLine.emit("llama.cpp is installed but does not run")
        elif found.ok:
            self.foundBinary.emit(found.server)
            self.statusLine.emit("llama.cpp ready")

    def remove(self) -> None:
        if QMessageBox.question(
                self, "Remove llama.cpp",
                "Remove the copy of llama.cpp that Kestrel installed?\n\n"
                "Anything installed by a package manager, Homebrew, or your own "
                "build is left untouched.") != QMessageBox.Yes:
            return
        self.output.clear()

        def work():
            removed = llamacpp.uninstall(progress=self._say)
            if removed:
                self.cfg.llama_server_bin = ""
                self.cfg.rpc_bin = ""
                self.cfg.save()
                self.statusLine.emit(f"Removed {len(removed)} item(s)")
            self.ui(self.scan)
        threading.Thread(target=work, daemon=True).start()

    def install(self, source: bool) -> None:
        self.install_btn.setEnabled(False)
        self.build_btn.setEnabled(False)
        self.output.clear()
        remove_first = self.reinstall.isChecked()

        def work():
            try:
                found = llamacpp.ensure(self.cfg, progress=self._say, allow_install=True,
                                        backend=self.cfg.llama_backend, source=source,
                                        with_rpc=self.cfg.llama_with_rpc,
                                        remove_first=remove_first)
                self.ui(lambda: self._apply(found))
            except Exception as e:
                self._say(f"failed: {e}")
                self.statusLine.emit(f"Install failed: {e}")
            finally:
                self.ui(lambda: (self.install_btn.setEnabled(True),
                                 self.build_btn.setEnabled(True)))
        threading.Thread(target=work, daemon=True).start()


# =========================================================== speech panel ===
class SpeechPanel(UiThread, QWidget):
    """Voice input and output. Local engines are the default and network
    engines stay unselectable until they are explicitly permitted."""

    uiCall = Signal(object)
    statusLine = Signal(str)
    transcribed = Signal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._init_ui_thread()
        self.cfg = cfg
        self.speech = speechmod.Speech(cfg)
        self._dl_cancel = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        tabs = QTabWidget()
        tabs.addTab(_scroll(self._output_tab()), "Voice out")
        tabs.addTab(_scroll(self._input_tab()), "Voice in")
        tabs.addTab(_scroll(self._services_tab()), "Engines")
        outer.addWidget(tabs)
        # Probing engines shells out — `espeak-ng --voices` and pyttsx3's
        # initialisation both take real time — so it happens off the GUI thread
        # rather than holding the window closed.
        threading.Thread(target=self._refresh_async, daemon=True).start()

    def _refresh_async(self) -> None:
        try:
            tts = self.speech.tts_status()
            stt = self.speech.stt_status()
            voices = self.speech.voices(self.cfg.speech.tts_engine)
            models = self.speech.stt_models(self.cfg.speech.stt_engine)
        except Exception:
            return
        self.ui(lambda: self._apply_engines(tts, stt, voices, models))

    # -- output ---------------------------------------------------------------
    def _output_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        s = self.cfg.speech

        self.tts_on = QCheckBox("Read replies aloud")
        self.tts_on.setChecked(s.tts_enabled)
        self.tts_on.toggled.connect(self._save)
        lay.addWidget(self.tts_on)

        self.auto_speak = QCheckBox("Only the final answer, not tool activity")
        self.auto_speak.setChecked(not s.speak_tool_calls)
        self.auto_speak.toggled.connect(self._save)
        lay.addWidget(self.auto_speak)

        self.tts_engine = QComboBox()
        self.tts_engine.currentIndexChanged.connect(self._engine_changed)
        lay.addWidget(Field("engine", self.tts_engine))

        self.voice = QComboBox()
        self.voice.currentIndexChanged.connect(self._save)
        lay.addWidget(Field("voice", self.voice))

        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.3, 3.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(s.tts_speed)
        self.speed.valueChanged.connect(self._save)
        lay.addWidget(Field("speed", self.speed))

        test = QPushButton("Test voice")
        test.clicked.connect(self.test_voice)
        lay.addWidget(test)

        speed_note = QLabel("Voices come in qualities: low is several times "
                            "faster to synthesise than medium and noticeably "
                            "quicker to start, which matters most on a laptop.")
        speed_note.setWordWrap(True)
        speed_note.setObjectName("Dim")
        lay.addWidget(speed_note)

        head = QLabel("Download a voice")
        head.setObjectName("Section")
        lay.addWidget(head)
        self.piper_list = QTreeWidget()
        self.piper_list.setHeaderLabels(["Voice", "Description"])
        self.piper_list.setObjectName("Flush")
        self.piper_list.setFont(mono_font(10))
        stretch_columns(self.piper_list, first_stretch=1)
        for locale, name, quality, desc in speechmod.PIPER_CATALOGUE:
            item = QTreeWidgetItem([f"{locale}-{name}-{quality}", desc])
            item.setData(0, Qt.UserRole, (locale, name, quality))
            self.piper_list.addTopLevelItem(item)
        lay.addWidget(self.piper_list, 1)

        self.dl_progress = QProgressBar()
        self.dl_progress.hide()
        lay.addWidget(self.dl_progress)
        grab = QPushButton("Download selected voice")
        grab.clicked.connect(self.download_voice)
        lay.addWidget(grab)
        lay.addStretch(1)
        return w

    # -- input ----------------------------------------------------------------
    def _input_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        s = self.cfg.speech

        self.stt_on = QCheckBox("Enable dictation")
        self.stt_on.setChecked(s.stt_enabled)
        self.stt_on.toggled.connect(self._save)
        lay.addWidget(self.stt_on)

        self.stt_engine = QComboBox()
        self.stt_engine.currentIndexChanged.connect(self._engine_changed)
        lay.addWidget(Field("engine", self.stt_engine))

        self.stt_model = QComboBox()
        self.stt_model.currentIndexChanged.connect(self._save)
        lay.addWidget(Field("model", self.stt_model))

        self.language = QLineEdit(s.stt_language)
        self.language.setPlaceholderText("auto, or a code such as en, de, fr")
        self.language.editingFinished.connect(self._save)
        lay.addWidget(Field("language", self.language))

        self.record_secs = QSpinBox()
        self.record_secs.setRange(2, 300)
        self.record_secs.setValue(s.record_seconds)
        self.record_secs.valueChanged.connect(self._save)
        lay.addWidget(Field("recording length (seconds)", self.record_secs))

        self.audio_note = QLabel("")
        self.audio_note.setWordWrap(True)
        self.audio_note.setObjectName("Dim")
        lay.addWidget(self.audio_note)

        test = QPushButton("Record and transcribe")
        test.clicked.connect(self.test_dictation)
        lay.addWidget(test)

        head = QLabel("Download a transcription model")
        head.setObjectName("Section")
        lay.addWidget(head)
        self.whisper_list = QTreeWidget()
        self.whisper_list.setHeaderLabels(["Model", "Size", "Notes"])
        self.whisper_list.setObjectName("Flush")
        self.whisper_list.setFont(mono_font(10))
        stretch_columns(self.whisper_list, first_stretch=2)
        for filename, size, desc in speechmod.WHISPER_CATALOGUE:
            item = QTreeWidgetItem([filename.replace("ggml-", "").replace(".bin", ""),
                                    size, desc])
            item.setData(0, Qt.UserRole, filename)
            self.whisper_list.addTopLevelItem(item)
        lay.addWidget(self.whisper_list, 1)

        grab = QPushButton("Download selected model")
        grab.clicked.connect(self.download_model)
        lay.addWidget(grab)
        lay.addStretch(1)
        return w

    # -- engines / services ---------------------------------------------------
    def _services_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        blurb = QLabel("Kestrel speaks and listens locally by default. Network "
                       "services are listed but cannot be selected until you "
                       "permit them below — until then nothing you say or hear "
                       "leaves this machine.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Dim")
        lay.addWidget(blurb)

        self.engine_tree = QTreeWidget()
        self.engine_tree.setHeaderLabels(["Engine", "Where", "State"])
        self.engine_tree.setColumnWidth(0, 170)
        self.engine_tree.setColumnWidth(1, 70)
        self.engine_tree.setObjectName("Flush")
        self.engine_tree.setFont(mono_font(10))
        lay.addWidget(self.engine_tree, 1)

        self.allow_net = QCheckBox("Permit network speech services")
        self.allow_net.setChecked(self.cfg.speech.allow_network)
        self.allow_net.toggled.connect(self._toggle_network)
        lay.addWidget(self.allow_net)

        self.api_base = QLineEdit(self.cfg.speech.api_base)
        self.api_base.setPlaceholderText("https://api.openai.com")
        self.api_base.editingFinished.connect(self._save)
        lay.addWidget(Field("OpenAI-compatible endpoint", self.api_base))

        self.api_key = QLineEdit(self.cfg.speech.api_key)
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.editingFinished.connect(self._save)
        lay.addWidget(Field("API key", self.api_key))

        head = QLabel("Install the offline engines")
        head.setObjectName("Section")
        lay.addWidget(head)
        hint = QLabel("Piper for speech out, faster-whisper for dictation, "
                      "sounddevice for the microphone. Installed into Kestrel's "
                      "own environment; about 150 MB.")
        hint.setWordWrap(True)
        hint.setObjectName("Dim")
        lay.addWidget(hint)
        self.install_btn = QPushButton("Install Piper and faster-whisper")
        self.install_btn.clicked.connect(self.install_engines)
        lay.addWidget(self.install_btn)
        self.install_log = QTextEdit()
        self.install_log.setReadOnly(True)
        self.install_log.setFont(mono_font(9))
        self.install_log.setMaximumHeight(110)
        self.install_log.hide()
        lay.addWidget(self.install_log)

        self.eleven_key = QLineEdit(self.cfg.speech.elevenlabs_key)
        self.eleven_key.setEchoMode(QLineEdit.Password)
        self.eleven_key.editingFinished.connect(self._save)
        lay.addWidget(Field("ElevenLabs key", self.eleven_key))
        lay.addStretch(1)
        return w

    # -- state ----------------------------------------------------------------
    def refresh(self) -> None:
        threading.Thread(target=self._refresh_async, daemon=True).start()

    def _apply_engines(self, tts, stt, voices, models) -> None:
        s = self.cfg.speech
        self.engine_tree.clear()
        for label, group in (("speech out", tts), ("speech in", stt)):
            head = QTreeWidgetItem([label.upper(), "", ""])
            head.setForeground(0, QColor(theme.TEXT_DIM))
            self.engine_tree.addTopLevelItem(head)
            for st in group:
                item = QTreeWidgetItem([st.name, st.badge,
                                        st.detail if st.available else st.install_hint])
                item.setForeground(1, QColor(theme.SIGNAL if st.available
                                             else theme.TEXT_DIM))
                head.addChild(item)
            head.setExpanded(True)

        self._fill(self.tts_engine, [("auto", "Automatic (best local)")]
                   + [(st.id, st.name) for st in tts if st.available], s.tts_engine)
        self._fill(self.stt_engine, [("auto", "Automatic (best local)")]
                   + [(st.id, st.name) for st in stt if st.available], s.stt_engine)

        self._fill(self.voice, [(v.id, v.label) for v in voices], s.tts_voice)
        if not voices:
            self.voice.addItem("no voices found — download one below", "")

        self._fill(self.stt_model, [(m.id, f"{m.name}  {m.quality}".strip())
                                    for m in models], s.stt_model)
        if not models:
            self.stt_model.addItem("no models found — download one below", "")

        can_play, can_rec = speechmod.audio_available()
        notes = []
        if not can_play:
            notes.append("No audio player detected; install ffmpeg, sox or alsa-utils.")
        if not can_rec:
            notes.append("No recorder detected; pip install sounddevice, or install ffmpeg.")
        self.audio_note.setText(" ".join(notes) or "Audio input and output are available.")

    @staticmethod
    def _fill(combo: QComboBox, pairs, selected: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        for value, label in pairs:
            combo.addItem(label, value)
        idx = combo.findData(selected)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _engine_changed(self) -> None:
        self._save()
        self.speech.close_stream()      # a different voice needs a new process
        self.refresh()

    def _toggle_network(self, on: bool) -> None:
        self.cfg.speech.allow_network = on
        self.cfg.save()
        self.statusLine.emit("Network speech services permitted" if on
                             else "Network speech services disabled")
        self.refresh()

    def _save(self) -> None:
        s = self.cfg.speech
        s.tts_enabled = self.tts_on.isChecked()
        s.speak_tool_calls = not self.auto_speak.isChecked()
        s.tts_engine = self.tts_engine.currentData() or "auto"
        s.tts_voice = self.voice.currentData() or ""
        s.tts_speed = self.speed.value()
        s.stt_enabled = self.stt_on.isChecked()
        s.stt_engine = self.stt_engine.currentData() or "auto"
        s.stt_model = self.stt_model.currentData() or ""
        s.stt_language = self.language.text().strip() or "auto"
        s.record_seconds = self.record_secs.value()
        s.api_base = self.api_base.text().strip()
        s.api_key = self.api_key.text()
        s.elevenlabs_key = self.eleven_key.text()
        self.cfg.save()

    # -- actions --------------------------------------------------------------
    def install_engines(self) -> None:
        self.install_btn.setEnabled(False)
        self.install_log.clear()
        self.install_log.show()

        def say(line: str) -> None:
            self.ui(lambda: self.install_log.append(line))

        def work():
            ok = speechmod.install_packages(["piper", "fasterwhisper", "audio"], say)
            self.statusLine.emit("Speech engines installed" if ok
                                 else "Installation failed — see the log")
            say("done." if ok else "failed.")
            self.ui(lambda: self.install_btn.setEnabled(True))
            self.refresh()

        threading.Thread(target=work, daemon=True).start()

    def test_voice(self) -> None:
        self.statusLine.emit("Loading the voice…")

        def work():
            sample = "Kestrel is ready. This is the selected voice."
            try:
                if self.speech.speak_now(sample):
                    self.statusLine.emit("Voice test complete")
                    return
                self.speech.speak(sample, blocking=True)
                self.statusLine.emit("Voice test complete")
            except Exception as e:
                self.statusLine.emit(f"Voice test failed: {e}")
        threading.Thread(target=work, daemon=True).start()

    def test_dictation(self) -> None:
        secs = self.record_secs.value()
        self.statusLine.emit(f"Recording for {secs}s…")

        def work():
            try:
                text = self.speech.listen(secs)
                self.statusLine.emit(f"Heard: {text[:120]}" if text else "Heard nothing")
                if text:
                    self.transcribed.emit(text)
            except Exception as e:
                self.statusLine.emit(f"Dictation failed: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _progress(self, done: int, total: int) -> None:
        if not total:
            return
        pct = int(1000 * done / total)
        label = f"{modelsmod.human_size(done)} / {modelsmod.human_size(total)}  %p%"

        def apply():
            self.dl_progress.setMaximum(1000)
            self.dl_progress.setValue(pct)
            self.dl_progress.setFormat(label)
        self.ui(apply)

    def download_voice(self) -> None:
        item = self.piper_list.currentItem()
        if item is None:
            QMessageBox.information(self, "No voice", "Pick a voice from the list.")
            return
        locale, name, quality = item.data(0, Qt.UserRole)
        dest = self.cfg.speech.voice_dirs[0]
        self.dl_progress.show()
        self._dl_cancel = False

        def work():
            try:
                path = speechmod.download_piper_voice(
                    locale, name, quality, dest, self._progress,
                    lambda: self._dl_cancel)
                self.cfg.speech.tts_voice = str(path)
                self.cfg.speech.tts_engine = "piper"
                self.cfg.save()
                self.statusLine.emit(f"Voice installed: {Path(path).name}")
                self.ui(self.refresh)
            except Exception as e:
                self.statusLine.emit(f"Voice download failed: {e}")
            finally:
                self.ui(self.dl_progress.hide)
        threading.Thread(target=work, daemon=True).start()

    def download_model(self) -> None:
        item = self.whisper_list.currentItem()
        if item is None:
            QMessageBox.information(self, "No model", "Pick a model from the list.")
            return
        filename = item.data(0, Qt.UserRole)
        dest = self.cfg.speech.model_dirs[0]
        self.dl_progress.show()
        self._dl_cancel = False

        def work():
            try:
                path = speechmod.download_whisper_model(
                    filename, dest, self._progress, lambda: self._dl_cancel)
                self.cfg.speech.stt_model = str(path)
                self.cfg.save()
                self.statusLine.emit(f"Model installed: {Path(path).name}")
                self.ui(self.refresh)
            except Exception as e:
                self.statusLine.emit(f"Model download failed: {e}")
            finally:
                self.ui(self.dl_progress.hide)
        threading.Thread(target=work, daemon=True).start()


# =========================================================== system panel ===
class SystemPanel(QWidget):
    """CPU, memory and GPU, sampled while the window is visible.

    Running a model locally is a resource negotiation, and the questions that
    matter — is the GPU being used at all, is memory about to run out — are
    otherwise in another window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.monitor = sysmon.Monitor()
        self.history: list[float] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)

        self.cpu = _Meter("CPU")
        self.mem = _Meter("Memory")
        lay.addWidget(self.cpu)
        lay.addWidget(self.mem)

        self.gpu_box = QVBoxLayout()
        lay.addLayout(self.gpu_box)
        self.gpu_meters: list[_Meter] = []

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setFont(mono_font(10))
        lay.addWidget(self.detail, 1)

        note = QLabel(sysmon.describe())
        note.setWordWrap(True)
        note.setObjectName("Dim")
        lay.addWidget(note)
        if not sysmon.HAVE_PSUTIL:
            hint = QLabel("Install psutil for per-core detail and swap: "
                          "pip install psutil")
            hint.setWordWrap(True)
            hint.setObjectName("Dim")
            lay.addWidget(hint)

        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.refresh)

    def showEvent(self, event):  # noqa: N802
        # Only sample while on screen: polling a GPU tool in the background
        # spawns a process every couple of seconds for a panel nobody is looking
        # at.
        self.timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event):  # noqa: N802
        self.timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        try:
            s = self.monitor.sample()
        except Exception as e:
            self.detail.setPlainText(f"Could not read system counters: {e}")
            return
        self.cpu.set_value(s.cpu_percent, f"{s.cpu_percent:.0f}%")
        self.mem.set_value(s.mem_percent,
                           f"{s.mem_used_mb / 1024:.1f} / {s.mem_total_mb / 1024:.1f} GB")

        # Two bars per device: how busy it is, and how much of its memory the
        # model is holding. The second is the one that decides whether a larger
        # model will fit, so it belongs on screen next to the others.
        while len(self.gpu_meters) < len(s.gpus) * 2:
            meter = _Meter("GPU")
            self.gpu_meters.append(meter)
            self.gpu_box.addWidget(meter)
        if not s.gpus and not s.gpus_scanned and not self.gpu_meters:
            meter = _Meter("GPU")
            meter.set_value(0, "looking…")
            self.gpu_meters.append(meter)
            self.gpu_box.addWidget(meter)
        for index, gpu in enumerate(s.gpus):
            busy, memory = self.gpu_meters[index * 2], self.gpu_meters[index * 2 + 1]
            busy.show()
            memory.show()
            busy.label.setText((gpu.name or "GPU")[:26])
            if gpu.utilisation >= 0:
                busy.set_value(gpu.utilisation, f"{gpu.utilisation:.0f}%")
            else:
                busy.set_value(0, "usage unavailable")

            memory.label.setText("GPU memory")
            budget = gpu.budget_mb
            if budget:
                used = gpu.mem_used_mb
                memory.set_value(100.0 * used / budget,
                                 f"{used / 1024:.1f} / {budget / 1024:.1f} GB"
                                 + (" shared" if gpu.integrated else ""))
            else:
                memory.set_value(0, "unknown")
        for meter in self.gpu_meters[len(s.gpus) * 2:]:
            meter.hide()

        lines = []
        if s.per_core:
            cores = "  ".join(f"{c:>3.0f}" for c in s.per_core[:16])
            lines.append(f"cores  {cores}")
        if s.swap_total_mb:
            lines.append(f"swap   {s.swap_used_mb / 1024:.1f} / "
                         f"{s.swap_total_mb / 1024:.1f} GB")
        for gpu in s.gpus:
            lines.append(gpu.name or "GPU")
            lines.append(f"  {gpu.memory_summary()}")
            extra = []
            if gpu.mem_used_mb:
                extra.append(f"{gpu.mem_used_mb} MB in use")
            if gpu.temperature >= 0:
                extra.append(f"{gpu.temperature:.0f}C")
            if extra:
                lines.append("  " + "  ".join(extra))
        if not s.gpus:
            # Reading GPU counters spawns a process and takes a second or two.
            # Saying there is none while still looking is simply wrong.
            lines.append("Looking for a GPU…" if not s.gpus_scanned else
                         "No GPU counters available. nvidia-smi, rocm-smi or the "
                         "Windows GPU counters provide them when a driver is "
                         "installed.")
        self.detail.setPlainText("\n".join(lines))


class _Meter(QWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        row = QHBoxLayout()
        self.label = QLabel(name)
        self.label.setObjectName("Eyebrow")
        self.value = QLabel("—")
        self.value.setObjectName("Readout")
        self.value.setAlignment(Qt.AlignRight)
        row.addWidget(self.label)
        row.addWidget(self.value, 1)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setMaximumHeight(6)
        self.bar.setRange(0, 100)
        lay.addWidget(self.bar)

    def set_value(self, percent: float, text: str) -> None:
        self.bar.setValue(int(max(0, min(100, percent))))
        self.value.setText(text)


# ========================================================= projects panel ===
class ProjectsPanel(UiThread, QWidget):
    """Projects and their conversations.

    A project is a folder under the workspace root with its own files, memory
    scope, checklist and saved conversations. Switching project switches all of
    them together — an agent that recalls one project's decisions while working
    on another is worse than one that recalls nothing.
    """

    uiCall = Signal(object)
    projectChosen = Signal(str)     # path
    sessionChosen = Signal(object)  # sessions.Session
    newSession = Signal()
    statusLine = Signal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._init_ui_thread()
        self.cfg = cfg
        self.projects: list[sessionmod.Project] = []
        self.sessions: list[sessionmod.Session] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)

        head = QLabel("Working folder")
        head.setObjectName("Section")
        lay.addWidget(head)

        # The path is the long thing here, so it gets the full width and the
        # buttons sit beneath it rather than squeezing it into a third of a
        # narrow panel.
        self.location = QLineEdit(str(Path(self.cfg.workspace).expanduser()))
        self.location.setToolTip("Where the agent's files live. Its file tools are "
                                 "confined to this folder.")
        self.location.returnPressed.connect(self.apply_location)
        lay.addWidget(self.location)

        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_location)
        show = QPushButton("Show in files")
        show.setToolTip("Open this folder in the file manager")
        show.clicked.connect(self.reveal)
        lay.addWidget(_row(browse, show))

        use = QPushButton("Use this folder")
        use.setObjectName("Primary")
        use.clicked.connect(self.apply_location)
        lay.addWidget(use)

        head2 = QLabel("Saved conversations")
        head2.setObjectName("Section")
        lay.addWidget(head2)

        self.session_list = QListWidget()
        self.session_list.itemDoubleClicked.connect(lambda *_: self.open_session())
        lay.addWidget(self.session_list, 2)

        row2 = QHBoxLayout()
        for text, slot in (("Open", self.open_session), ("New", self.newSession.emit),
                           ("Delete", self.delete_session)):
            b = QPushButton(text)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(slot)
            row2.addWidget(b)
        lay.addLayout(row2)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setObjectName("Dim")
        lay.addWidget(self.note)
        self.refresh()

    # -- state ---------------------------------------------------------------
    def root(self) -> Path:
        return Path(self.cfg.workspace_root or Path.home() / "kestrel-workspace")

    def current_project(self) -> sessionmod.Project:
        path = Path(self.cfg.workspace).expanduser()
        return sessionmod.Project(name=path.name, path=path)

    def refresh(self) -> None:
        self.location.setText(str(Path(self.cfg.workspace).expanduser()))
        self.refresh_sessions()

    def browse_location(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Workspace folder", self.location.text() or str(Path.home()))
        if chosen:
            self.location.setText(chosen)
            self.apply_location()

    def apply_location(self) -> None:
        """Switch to the folder in the field, creating it if it does not exist."""
        path = Path(self.location.text().strip()).expanduser()
        if not path.name:
            return
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "Cannot use that folder", str(e))
            return
        self.projectChosen.emit(str(path))

    def refresh_sessions(self) -> None:
        def work():
            active = self.current_project()
            sessions = sessionmod.list_sessions(active)
            stats = active.stats()
            self.ui(lambda: self._show_sessions(active, sessions, stats))
        threading.Thread(target=work, daemon=True).start()

    def _show_sessions(self, active, sessions, stats: str) -> None:
        self.sessions = sessions
        self.session_list.clear()
        for s in self.sessions:
            self.session_list.addItem(f"{s.title}\n    {s.summary()}")
        self.note.setText(f"Active: {active.name} — {stats}\n{active.path}")

    # -- actions -------------------------------------------------------------
    def reveal(self) -> None:
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.location.text() or str(Path.home())))

    def open_session(self) -> None:
        row = self.session_list.currentRow()
        if 0 <= row < len(self.sessions):
            self.sessionChosen.emit(self.sessions[row])

    def delete_session(self) -> None:
        row = self.session_list.currentRow()
        if not (0 <= row < len(self.sessions)):
            return
        if QMessageBox.question(self, "Delete conversation",
                                "Delete this saved conversation?") == QMessageBox.Yes:
            sessionmod.delete_session(self.sessions[row])
            self.refresh_sessions()


# ============================================================ tools panel ===
class ToolsPanel(QWidget):
    """What the model can actually do.

    The system prompt lists these to the model but not to the reader, so the
    only way to know what an agent was able to reach was to read the source.
    Danger is shown because it decides what the approval setting will stop.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(7)

        blurb = QLabel("Every tool the model can call. Writes and shell commands "
                       "follow the approval setting.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Dim")
        lay.addWidget(blurb)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Tool", "Access"])
        self.tree.setObjectName("Flush")
        self.tree.setFont(mono_font(10))
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        stretch_columns(self.tree)
        self.tree.currentItemChanged.connect(self._show)
        lay.addWidget(self.tree, 2)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setObjectName("Flush")
        self.detail.setFont(mono_font(10))
        lay.addWidget(self.detail, 1)

        self.count = QLabel("")
        self.count.setObjectName("Dim")
        lay.addWidget(self.count)
        self.tools: list[dict] = []

    def update_tools(self, tools: list[dict]) -> None:
        self.tools = list(tools or [])
        self.tree.clear()
        access = {"safe": "read", "write": "writes", "exec": "runs commands"}
        for tool in self.tools:
            item = QTreeWidgetItem([tool["name"], access.get(tool["danger"], "")])
            colour = {"write": theme.AMBER, "exec": theme.ALERT}.get(tool["danger"])
            if colour:
                item.setForeground(1, QColor(colour))
            item.setToolTip(0, tool["summary"])
            self.tree.addTopLevelItem(item)
        self.count.setText(f"{len(self.tools)} tools available")
        if self.tools:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def _show(self, item, _previous=None) -> None:
        if item is None:
            return
        tool = next((t for t in self.tools if t["name"] == item.text(0)), None)
        if tool is None:
            return
        lines = [tool["signature"], "", tool["summary"]]
        if tool["detail"]:
            lines += ["", tool["detail"]]
        if tool["params"]:
            lines.append("")
            for name, kind, required, desc in tool["params"]:
                mark = "required" if required else "optional"
                lines.append(f"  {name} ({kind}, {mark})")
                if desc:
                    lines.append(f"      {desc}")
        self.detail.setPlainText("\n".join(lines))

# ================================================================= canvas ===
class CanvasSurface(QWidget):
    """One editable buffer with its controls.

    Used twice: once for what the model writes, once for what you load. They
    behave identically except that the model's is the one its tools write to.
    """

    statusLine = Signal(str)
    reviewRequested = Signal(str, str)
    bufferChanged = Signal(str)

    def __init__(self, cfg, buffer, allow_import: bool = False, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.buffer = buffer
        self.path: Path | None = None
        self._syncing = False
        self.bufferChanged.connect(self._buffer_changed)
        buffer.listen(lambda text: self.bufferChanged.emit(text))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(6)

        row = QHBoxLayout()
        self.filename = QLineEdit(buffer.name)
        row.addWidget(self.filename, 1)
        self.language = QComboBox()
        self.language.addItems(["python", "javascript", "shell", "markdown",
                                "json", "yaml", "text", "c", "cpp", "rust",
                                "go", "html", "css", "sql", "other"])
        row.addWidget(self.language)
        lay.addLayout(row)

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("Flush")
        self.editor.setFont(mono_font(10))
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setPlaceholderText(
            "Files you load appear here for the model to read."
            if allow_import else
            "The model writes code here. You can edit it, then Save.")
        self.editor.textChanged.connect(self._touched)
        lay.addWidget(self.editor, 1)

        first = QHBoxLayout()
        actions = [("Check", self.review, "Ask the model to review this"),
                   ("Save", self.save, "Write it into the project folder"),
                   ("Copy", self.copy, "Copy the whole buffer")]
        for text, slot, tip in actions:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(slot)
            first.addWidget(b)
        lay.addLayout(first)

        second = QHBoxLayout()
        more = [("Import…" if allow_import else "Open…", self.open_file,
                 "Load a file — text, code, Word, Excel, PDF or an image"),
                ("Save as…", self.save_as, "Write it somewhere else"),
                ("Clear", self.clear, "Empty this canvas")]
        for text, slot, tip in more:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(slot)
            second.addWidget(b)
        lay.addLayout(second)

        self.note = QLabel("Empty.")
        self.note.setObjectName("Dim")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)

    # -- state ------------------------------------------------------------
    def _buffer_changed(self, text: str) -> None:
        if text == self.editor.toPlainText():
            return
        at = self.editor.textCursor().position()
        self._syncing = True
        self.editor.setPlainText(text)
        self._syncing = False
        cursor = self.editor.textCursor()
        cursor.setPosition(min(at, len(text)))
        self.editor.setTextCursor(cursor)

    def _touched(self) -> None:
        if not self._syncing:
            self.buffer.set(self.editor.toPlainText(),
                            language=self.language.currentText(),
                            name=self.filename.text().strip())
        lines = self.editor.document().blockCount()
        chars = len(self.editor.toPlainText())
        where = str(self.path) if self.path else "not saved yet"
        self.note.setText(f"{lines} line(s), {chars:,} characters · {where}")

    def target(self) -> Path:
        name = self.filename.text().strip() or "scratch.txt"
        candidate = Path(name).expanduser()
        if candidate.is_absolute():
            return candidate
        return Path(self.cfg.workspace).expanduser() / candidate

    # -- actions ----------------------------------------------------------
    def review(self) -> None:
        text = self.editor.toPlainText()
        if not text.strip():
            self.statusLine.emit("Nothing here to check.")
            return
        self.reviewRequested.emit(text, self.language.currentText())

    def save(self) -> None:
        target = self.target()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.editor.toPlainText(), "utf-8")
        except OSError as e:
            QMessageBox.warning(self, "Could not save", str(e))
            return
        self.path = target
        self._touched()
        self.statusLine.emit(f"Saved {target}")

    def save_as(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(self, "Save as", str(self.target()))
        if chosen:
            self.filename.setText(chosen)
            self.save()

    def open_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Load a file", str(Path(self.cfg.workspace).expanduser()),
            "All files (*);;Text and code (*.txt *.md *.py *.js *.json *.csv);;"
            "Documents (*.docx *.xlsx *.pptx *.pdf *.odt);;"
            "Images (*.png *.jpg *.jpeg *.gif *.webp)")
        if not chosen:
            return
        item = attachmod.read(chosen)
        if item.kind == "image":
            QMessageBox.information(
                self, "Image loaded",
                f"{item.name} is {item.meta.get('dimensions', 'an image')}.\n\n"
                "Its contents cannot be read by a text model, so a description "
                "of it has been placed here instead.")
            self.editor.setPlainText(item.block())
        elif not item.text.strip():
            QMessageBox.information(self, "Nothing to read",
                                    item.note or "No readable text in that file.")
            return
        else:
            self.editor.setPlainText(item.text)
        self.filename.setText(item.name)
        self.path = Path(chosen)
        suffix = self.path.suffix.lstrip(".").lower()
        known = {"py": "python", "js": "javascript", "sh": "shell",
                 "md": "markdown", "json": "json", "yml": "yaml",
                 "yaml": "yaml", "rs": "rust", "go": "go", "sql": "sql"}
        self.language.setCurrentText(known.get(suffix, "text"))
        self._touched()
        extra = f" — {item.note}" if item.note else ""
        if item.truncated:
            extra += " (truncated)"
        self.statusLine.emit(f"Loaded {item.label}{extra}")

    def copy(self) -> None:
        text = self.editor.toPlainText()
        if not text:
            self.statusLine.emit("Nothing to copy.")
            return
        QApplication.clipboard().setText(text)
        self.statusLine.emit(f"Copied {len(text):,} characters")

    def clear(self) -> None:
        if self.editor.toPlainText().strip() and QMessageBox.question(
                self, "Clear", "Discard what is here?") != QMessageBox.Yes:
            return
        self.editor.clear()
        self.path = None
        self._touched()

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(text)
        self._touched()


class CanvasPanel(QWidget):
    """Two surfaces: what the model writes, and what you brought it.

    They are separate because they have different owners. An import cannot
    destroy what the model is midway through writing, and the model cannot
    overwrite the file you loaded for it to look at — it can read that one
    with canvas_read(source="user").
    """

    statusLine = Signal(str)
    reviewRequested = Signal(str, str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(6)

        tabs = QTabWidget()
        self.model_surface = CanvasSurface(cfg, canvasmod.BUFFER)
        self.user_surface = CanvasSurface(cfg, canvasmod.USER_BUFFER,
                                          allow_import=True)
        tabs.addTab(self.model_surface, "Model")
        tabs.addTab(self.user_surface, "Your files")
        for surface in (self.model_surface, self.user_surface):
            surface.statusLine.connect(self.statusLine.emit)
            surface.reviewRequested.connect(self.reviewRequested.emit)
        lay.addWidget(tabs, 1)
        self.tabs = tabs

    # kept so existing callers still work
    @property
    def editor(self):
        return self.model_surface.editor

    @property
    def filename(self):
        return self.model_surface.filename

    def set_text(self, text: str) -> None:
        self.model_surface.set_text(text)

    def load_file(self, path: str) -> None:
        """Put a file into the user surface and show it."""
        item = attachmod.read(path)
        self.user_surface.set_text(item.text or item.block())
        self.user_surface.filename.setText(item.name)
        self.tabs.setCurrentWidget(self.user_surface)
        self.statusLine.emit(f"Loaded {item.label}")
