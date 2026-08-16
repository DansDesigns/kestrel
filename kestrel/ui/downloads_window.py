"""The download window.

Separate from the Models panel because downloading is not something you watch:
it runs for an hour, several files at once, while the rest of the application is
used for something else. Closing this window does not stop anything — the
manager owns the transfers, and reopening rebuilds the view from it.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSplitter, QTabWidget, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import downloads as dl
from .. import models as modelsmod
from . import theme
from .widgets import mono_font, stretch_columns


class DownloadsWindow(QWidget):
    statusLine = Signal(str)
    finished = Signal(str)          # path of a completed file
    # Results come back from worker threads. A queued signal is the only way to
    # cross that boundary: QTimer.singleShot from another thread has no event
    # loop to attach to and silently never fires.
    uiCall = Signal(object)

    def __init__(self, cfg, manager: dl.DownloadManager, parent=None):
        super().__init__(parent, Qt.Window)
        self.cfg = cfg
        self.manager = manager
        self.repos: list = []
        self.files: list = []
        self._seen_done: set[int] = set()
        self._installed: list = []

        self.uiCall.connect(self._run_on_ui)
        self.setWindowTitle("Download models")
        self.resize(880, 620)
        self.setMinimumSize(560, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        tabs = QTabWidget()
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._search_side())
        splitter.addWidget(self._queue_side())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        tabs.addTab(splitter, "Find and download")
        tabs.addTab(self._installed_side(), "Installed")
        tabs.currentChanged.connect(
            lambda i: self.refresh_installed() if i == 1 else None)
        root.addWidget(tabs, 1)

        self.status = QLabel("")
        self.status.setObjectName("Dim")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        # Polled rather than signalled: transfers report from worker threads,
        # and a timer on the GUI thread avoids marshalling every chunk.
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh_queue)
        self.timer.start()
        self.refresh_queue()

    # -- searching -----------------------------------------------------------
    def _search_side(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.query = QLineEdit()
        self.query.setPlaceholderText("Search Hugging Face for GGUF models…")
        self.query.returnPressed.connect(self.search)
        self.sort = QComboBox()
        for label, key in (("most downloaded", "downloads"),
                           ("most liked", "likes"),
                           ("recently updated", "modified"),
                           ("recently created", "created")):
            self.sort.addItem(label, key)
        find = QPushButton("Search")
        find.setObjectName("Primary")
        find.clicked.connect(self.search)
        row = QHBoxLayout()
        row.addWidget(self.query, 1)
        row.addWidget(self.sort)
        row.addWidget(find)
        lay.addLayout(row)

        lists = QSplitter(Qt.Horizontal)
        self.repo_list = QTreeWidget()
        self.repo_list.setHeaderLabels(["Repository", "Downloads"])
        self.repo_list.setObjectName("Flush")
        self.repo_list.setFont(mono_font(10))
        self.repo_list.setRootIsDecorated(False)
        self.repo_list.currentItemChanged.connect(self._repo_chosen)
        stretch_columns(self.repo_list)
        lists.addWidget(self.repo_list)

        self.file_list = QTreeWidget()
        self.file_list.setHeaderLabels(["File", "Size"])
        self.file_list.setObjectName("Flush")
        self.file_list.setFont(mono_font(10))
        self.file_list.setRootIsDecorated(False)
        self.file_list.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.file_list.itemDoubleClicked.connect(lambda *_: self.queue_selected())
        stretch_columns(self.file_list)
        lists.addWidget(self.file_list)
        lists.setSizes([420, 420])
        lay.addWidget(lists, 1)

        self.dest = QLineEdit(self.cfg.model_dirs[0] if self.cfg.model_dirs else "")
        browse = QPushButton("…")
        browse.setMaximumWidth(36)
        browse.clicked.connect(self._pick_dest)
        queue = QPushButton("Download selected")
        queue.setObjectName("Primary")
        queue.clicked.connect(self.queue_selected)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Save to"))
        row2.addWidget(self.dest, 1)
        row2.addWidget(browse)
        row2.addWidget(queue)
        lay.addLayout(row2)
        return w

    def _pick_dest(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Save models to",
                                                  self.dest.text() or str(Path.home()))
        if chosen:
            self.dest.setText(chosen)

    def _error(self, message: str) -> None:
        """Show a failure in the results, not only in the status line."""
        self.status.setText(message)
        self.status.setStyleSheet(f"color: {theme.ALERT};")
        self.repo_list.clear()
        item = QTreeWidgetItem([message[:120], ""])
        item.setForeground(0, QColor(theme.ALERT))
        self.repo_list.addTopLevelItem(item)
        self.statusLine.emit(message[:120])

    def search(self) -> None:
        term = self.query.text().strip()
        if not term:
            self.status.setText("Type something to search for.")
            return
        self.status.setStyleSheet("")
        self.status.setText(f"Searching Hugging Face for “{term}”…")
        sort = self.sort.currentData() or "downloads"
        token = self.cfg.hf_token

        def work():
            try:
                found = modelsmod.search_repos(term, sort=sort, token=token)
            except Exception as e:
                detail = str(e)
                hint = ""
                if "403" in detail or "401" in detail:
                    hint = ("  A token may be required, or a proxy is blocking "
                            "huggingface.co.")
                elif "SSL" in detail or "certificate" in detail.lower():
                    hint = "  This looks like a TLS or proxy problem."
                elif "Name or service" in detail or "getaddrinfo" in detail:
                    hint = "  The name huggingface.co did not resolve."
                self._ui(lambda: self._error(f"Search failed: {detail[:160]}{hint}"))
                return
            self._ui(lambda: self._show_repos(found))
        threading.Thread(target=work, daemon=True).start()

    def _show_repos(self, found) -> None:
        self.repos = found
        self.status.setStyleSheet("")
        self.repo_list.clear()
        if not found:
            self.status.setText("No repositories matched. Try a shorter term, "
                                "such as the model family alone.")
            return
        for repo in found:
            item = QTreeWidgetItem([repo.id, f"{repo.downloads:,}"])
            item.setData(0, Qt.UserRole, repo.id)
            self.repo_list.addTopLevelItem(item)
        self.status.setText(f"{len(found)} repositories. Pick one to list its files.")

    def _repo_chosen(self, item, _previous=None) -> None:
        if item is None:
            return
        repo = item.data(0, Qt.UserRole)
        self.file_list.clear()
        self.status.setText(f"Listing {repo}…")

        def work():
            try:
                files = modelsmod.list_repo_files(repo, token=self.cfg.hf_token)
            except Exception as e:
                message = str(e)
                self._ui(lambda: self.status.setText(f"Could not list files: {message}"))
                return
            self._ui(lambda: self._show_files(repo, files))
        threading.Thread(target=work, daemon=True).start()

    def _show_files(self, repo: str, files) -> None:
        self.files = files
        self.file_list.clear()
        for f in files:
            item = QTreeWidgetItem([f.name, modelsmod.human_size(f.size)])
            item.setData(0, Qt.UserRole, (repo, f.name))
            self.file_list.addTopLevelItem(item)
        self.status.setText(f"{len(files)} file(s) in {repo}. "
                            "Select one or more, or double-click to start.")

    def queue_selected(self) -> None:
        chosen = self.file_list.selectedItems()
        if not chosen:
            self.status.setText("Select a file first.")
            return
        destination = self.dest.text().strip()
        if not destination:
            self.status.setText("Choose a folder to save into.")
            return
        for item in chosen:
            repo, filename = item.data(0, Qt.UserRole)
            self.manager.add(repo, filename, destination)
        self.statusLine.emit(f"Queued {len(chosen)} download(s)")
        self.refresh_queue()

    # -- what is already here ------------------------------------------------
    def _installed_side(self) -> QWidget:
        """The models on disk, in the same window as the ones being fetched.

        Downloading without seeing what you already have is how a 20 GB file
        gets fetched twice, so the two live together.
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(6)

        self.installed = QTreeWidget()
        self.installed.setHeaderLabels(["Model", "Quant", "Size", "Folder"])
        self.installed.setObjectName("Flush")
        self.installed.setFont(mono_font(10))
        self.installed.setRootIsDecorated(False)
        self.installed.setAlternatingRowColors(True)
        self.installed.setUniformRowHeights(True)
        stretch_columns(self.installed)
        lay.addWidget(self.installed, 1)

        self.installed_note = QLabel("")
        self.installed_note.setObjectName("Dim")
        self.installed_note.setWordWrap(True)
        lay.addWidget(self.installed_note)

        row = QHBoxLayout()
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self.refresh_installed)
        reveal = QPushButton("Show in files")
        reveal.clicked.connect(self._reveal_installed)
        delete = QPushButton("Delete from disk…")
        delete.setObjectName("Danger")
        delete.clicked.connect(self._delete_installed)
        for b in (rescan, reveal, delete):
            row.addWidget(b)
        lay.addLayout(row)
        self.refresh_installed()
        return w

    def refresh_installed(self) -> None:
        self.installed_note.setText("Scanning…")

        def work():
            catalog = modelsmod.Catalog(self.cfg.model_dirs)
            catalog.scan()
            entries = list(catalog.entries)
            self._ui(lambda: self._show_installed(entries))
        threading.Thread(target=work, daemon=True).start()

    def _show_installed(self, entries) -> None:
        self._installed = entries
        self.installed.clear()
        total = 0
        for entry in entries:
            total += entry.info.file_size
            item = QTreeWidgetItem([entry.name, entry.info.quant or "—",
                                    modelsmod.human_size(entry.info.file_size),
                                    str(Path(entry.path).parent)])
            item.setData(0, Qt.UserRole, str(entry.path))
            item.setToolTip(0, str(entry.path))
            self.installed.addTopLevelItem(item)
        for column in (1, 2):
            self.installed.resizeColumnToContents(column)
        self.installed_note.setText(
            f"{len(entries)} model(s), {modelsmod.human_size(total)} on disk."
            if entries else
            "No GGUF files found in the model folders.")

    def _selected_installed(self):
        item = self.installed.currentItem()
        if item is None:
            return None
        path = item.data(0, Qt.UserRole)
        return next((e for e in getattr(self, "_installed", [])
                     if str(e.path) == path), None)

    def _reveal_installed(self) -> None:
        entry = self._selected_installed()
        if entry is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(entry.path).parent)))

    def _delete_installed(self) -> None:
        entry = self._selected_installed()
        if entry is None:
            self.status.setText("Pick a model first.")
            return
        parts = (f"\n\nThis model is in {entry.info.parts} parts; all will be "
                 "deleted." if entry.info.parts > 1 else "")
        if QMessageBox.question(
                self, "Delete model",
                f"Permanently delete this file?\n\n{entry.path}\n"
                f"{entry.info.size_gb:.1f} GB{parts}\n\nThis cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        removed, failures = modelsmod.delete_model(entry)
        if failures:
            QMessageBox.warning(self, "Could not delete everything",
                                "\n".join(failures[:6]))
        self.statusLine.emit(f"Deleted {len(removed)} file(s)")
        self.refresh_installed()

    # -- the queue -----------------------------------------------------------
    def _queue_side(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.queue = QTreeWidget()
        self.queue.setHeaderLabels(["File", "Progress", "State", ""])
        self.queue.setObjectName("Flush")
        self.queue.setFont(mono_font(10))
        self.queue.setRootIsDecorated(False)
        self.queue.setUniformRowHeights(True)
        header = self.queue.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.queue.currentItemChanged.connect(lambda *_: self._sync_buttons())
        lay.addWidget(self.queue, 1)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_selected)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("Danger")
        cancel.clicked.connect(self._cancel_selected)
        clear = QPushButton("Clear finished")
        clear.clicked.connect(lambda: (self.manager.clear_finished(),
                                       self.refresh_queue()))
        self.concurrent = QComboBox()
        self.concurrent.addItems(["1 at a time", "2 at a time", "3 at a time",
                                  "4 at a time"])
        self.concurrent.setCurrentIndex(self.manager.max_concurrent - 1)
        self.concurrent.currentIndexChanged.connect(self._set_concurrency)
        row = QHBoxLayout()
        for widget in (self.pause_btn, cancel, clear, self.concurrent):
            row.addWidget(widget)
        lay.addLayout(row)
        return w

    def _set_concurrency(self, index: int) -> None:
        self.manager.max_concurrent = index + 1
        self.manager._schedule()

    def _selected_job(self) -> dl.Job | None:
        item = self.queue.currentItem()
        if item is None:
            return None
        return self.manager.get(item.data(0, Qt.UserRole))

    def _toggle_selected(self) -> None:
        job = self._selected_job()
        if job is None:
            return
        if job.state == dl.PAUSED or job.state == dl.FAILED:
            self.manager.resume(job.id)
        else:
            self.manager.pause(job.id)
        self.refresh_queue()

    def _cancel_selected(self) -> None:
        job = self._selected_job()
        if job is not None:
            self.manager.cancel(job.id)
            self.refresh_queue()

    def _sync_buttons(self) -> None:
        job = self._selected_job()
        resumable = job is not None and job.state in (dl.PAUSED, dl.FAILED)
        self.pause_btn.setText("Resume" if resumable else "Pause")
        self.pause_btn.setEnabled(job is not None and job.state in dl.ACTIVE
                                  or resumable)

    def refresh_queue(self) -> None:
        jobs = self.manager.jobs
        current = self.queue.currentItem()
        current_id = current.data(0, Qt.UserRole) if current else None

        if self.queue.topLevelItemCount() != len(jobs):
            self.queue.clear()
            for job in jobs:
                item = QTreeWidgetItem(["", "", "", ""])
                item.setData(0, Qt.UserRole, job.id)
                self.queue.addTopLevelItem(item)
                bar = QProgressBar()
                bar.setMaximum(1000)
                bar.setTextVisible(False)
                bar.setMaximumHeight(10)
                self.queue.setItemWidget(item, 1, bar)

        colours = {dl.DONE: theme.SIGNAL, dl.FAILED: theme.ALERT,
                   dl.PAUSED: theme.AMBER, dl.CANCELLED: theme.TEXT_DIM}
        for index, job in enumerate(jobs):
            item = self.queue.topLevelItem(index)
            if item is None:
                continue
            item.setData(0, Qt.UserRole, job.id)
            item.setText(0, job.name)
            item.setToolTip(0, f"{job.repo}/{job.filename}\n→ {job.dest}")
            item.setText(2, job.state)
            item.setText(3, job.summary())
            colour = colours.get(job.state)
            item.setForeground(2, QColor(colour or theme.TEXT))
            bar = self.queue.itemWidget(item, 1)
            if bar is not None:
                bar.setValue(int(job.percent * 10))
            if job.state == dl.DONE and job.id not in self._seen_done:
                self._seen_done.add(job.id)
                self.finished.emit(str(job.final_path))
                self.statusLine.emit(f"Downloaded {job.name}")
                self.refresh_installed()      # it is now one of the local ones

        if current_id is not None:
            for index in range(self.queue.topLevelItemCount()):
                item = self.queue.topLevelItem(index)
                if item.data(0, Qt.UserRole) == current_id:
                    self.queue.setCurrentItem(item)
                    break
        active = len(self.manager.active())
        self.setWindowTitle(f"Download models — {active} active" if active
                            else "Download models")
        self._sync_buttons()

    def _ui(self, fn) -> None:
        self.uiCall.emit(fn)

    def _run_on_ui(self, fn) -> None:
        try:
            fn()
        except Exception as e:
            # Never silently: a swallowed exception here looks exactly like a
            # search that found nothing, which is the hardest kind of bug to
            # report.
            self._error(f"{type(e).__name__}: {e}")

    def closeEvent(self, event):  # noqa: N802
        # The window closes; the transfers do not. Stopping them here would
        # make closing the window an act of destruction rather than tidying.
        self.timer.stop()
        super().closeEvent(event)

    def showEvent(self, event):  # noqa: N802
        self.timer.start()
        self.refresh_queue()
        super().showEvent(event)
