"""Custom widgets.

The context gauge is the centrepiece. Every other agent UI shows you a token
count after the fact; this one shows you the whole window at all times, split
into what the system prompt costs, what the transcript costs, what is free, and
the generation reserve that never gets spent on input. When the amber band
creeps right you can see a compaction coming before it happens.
"""
from __future__ import annotations

import html
import re

from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetrics, QIcon,
                           QPainter, QPen,
                           QPixmap, QTextCharFormat, QTextCursor)
from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSpinBox,
                               QApplication, QComboBox, QHBoxLayout, QHeaderView,
                               QLabel, QPlainTextEdit, QSizePolicy, QSlider,
                               QStyle, QStyledItemDelegate, QTabBar, QTextBrowser,
                               QTextEdit, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from . import theme


_FAMILY_CACHE: dict[str, set] = {}
_ICON_CACHE: dict[tuple, QIcon] = {}


def _families() -> set:
    """Cached: enumerating fonts is cheap on Linux and decidedly not on a
    Windows machine with a large font set, and construction asks a dozen times."""
    if "all" not in _FAMILY_CACHE:
        # Legacy raster faces and the vertical "@" variants cannot be
        # instantiated; keeping them out avoids DirectWrite failures at runtime.
        _FAMILY_CACHE["all"] = {
            f for f in QFontDatabase.families()
            if not f.startswith("@")
            and f.lower() not in {"8514oem", "fixedsys", "modern", "ms sans serif",
                                  "ms serif", "roman", "script", "small fonts",
                                  "system", "terminal"}}
    return _FAMILY_CACHE["all"]


def clear_font_cache() -> None:
    _FAMILY_CACHE.clear()
    _ICON_CACHE.clear()


def mono_font(size: int = 11) -> QFont:
    families = _families()
    for name in theme.mono_fonts():
        if name in families:
            candidate = QFont(name)
            candidate.setPointSize(size)
            return candidate
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    f.setPointSize(size)
    return f


# ------------------------------------------------------------------- gauge --
class ContextGauge(QWidget):
    """The whole context window as one bar, with a key to what the bands mean.

    A coloured bar nobody can read is decoration. The legend below it names each
    band and gives its current size, so the picture and the numbers agree.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(62)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.n_ctx = 4096
        self.system = 0
        self.memory = 0
        self.plan = 0
        self.history = 0
        self.reserve = 0
        self.profile = "—"
        self.compactions = 0
        self.rate = 0.0
        self.setToolTip("The context window: prompt, recalled memory, checklist, "
                        "transcript, free space, and the reserved generation headroom.")

    def update_usage(self, usage, budget, compactions: int = 0) -> None:
        self.n_ctx = max(1, usage.n_ctx)
        self.system = usage.system
        self.memory = getattr(usage, "memory", 0)
        self.plan = getattr(usage, "plan", 0)
        self.history = usage.history
        self.reserve = usage.output or budget.output
        self.profile = budget.profile
        self.compactions = compactions
        self.update()

    def set_rate(self, tokens_per_second: float) -> None:
        self.rate = max(0.0, tokens_per_second)
        self.update()

    def _bands(self):
        used = self.system + self.memory + self.plan + self.history
        free = max(0, self.n_ctx - used - self.reserve)
        return [
            ("prompt", theme.SIGNAL, self.system),
            ("memory", theme.VIOLET, self.memory),
            ("plan", theme.PLAN, self.plan),
            ("transcript", theme.AMBER, self.history),
            ("free", theme.FREE, free),
            ("reserved", theme.LINE, self.reserve),
        ]

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        bar_h, top = 13, 3
        bands = self._bands()
        used = self.system + self.memory + self.plan + self.history

        p.setPen(Qt.NoPen)
        p.fillRect(QRectF(0, top, w, bar_h), QColor(theme.FREE))
        x = 0.0
        for _label, colour, size in bands:
            if size <= 0:
                continue
            width = w * size / self.n_ctx
            p.fillRect(QRectF(x, top, max(1.0, width), bar_h), QColor(colour))
            x += width

        pen = QPen(QColor(theme.TEXT_DIM))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(QRectF(0.5, top + 0.5, w - 1, bar_h - 1))
        p.setPen(QPen(QColor(theme.INK), 1))
        for q in (0.25, 0.5, 0.75):
            p.drawLine(int(w * q), top, int(w * q), top + bar_h)

        # legend: swatch, name, size — the key the bar needs to be readable
        p.setFont(mono_font(9))
        metrics = p.fontMetrics()
        y = top + bar_h + 6
        cursor = 0
        for label, colour, size in bands:
            text = f"{label} {size:,}"
            entry = 11 + metrics.horizontalAdvance(text) + 14
            if cursor + entry > w - 150:
                break
            p.setPen(Qt.NoPen)
            p.fillRect(QRectF(cursor, y + 3, 7, 7), QColor(colour))
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(int(cursor + 11), y + metrics.ascent() + 1, text)
            cursor += entry

        right = f"{used:,}/{self.n_ctx:,}  {100 * used / self.n_ctx:.0f}%"
        if self.rate > 0:
            right = f"{self.rate:.1f} tok/s   " + right
        if self.compactions:
            right += f"   {self.compactions}\u00d7 compacted"
        right += f"   {self.profile.upper()}"
        p.setPen(QColor(theme.AMBER))
        p.drawText(QRectF(0, y, w, metrics.height() + 2),
                   Qt.AlignRight | Qt.AlignVCenter, right)
        p.end()


# ------------------------------------------------------------- transcript --
CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def md_to_html(text: str) -> str:
    """Small markdown subset, enough for agent output in a QTextEdit."""
    blocks: list[str] = []

    def stash(m):
        body = html.escape(m.group(2).rstrip())
        blocks.append(
            f'<pre style="background:{theme.PANEL};border:1px solid {theme.LINE};'
            f'padding:8px;color:{theme.TEXT};">{body}</pre>'
        )
        return f"\x00{len(blocks) - 1}\x00"

    text = CODE_FENCE.sub(stash, text)
    out: list[str] = []
    in_list = False
    for line in text.split("\n"):
        raw = line.rstrip()
        stripped = raw.strip()
        if re.fullmatch(r"\x00\d+\x00", stripped):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(blocks[int(stripped.strip("\x00"))])
            continue
        esc = html.escape(raw)
        esc = re.sub(r"`([^`]+)`",
                     rf'<code style="background:{theme.PANEL};color:{theme.AMBER};">\1</code>', esc)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
        esc = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)\*", r"<i>\1</i>", esc)
        m = re.match(r"^(#{1,4})\s+(.*)", esc.strip())
        if m:
            if in_list:
                out.append("</ul>")
                in_list = False
            size = {1: 16, 2: 15, 3: 14, 4: 13}[len(m.group(1))]
            out.append(f'<div style="font-size:{size}px;font-weight:600;'
                       f'margin:8px 0 3px;">{m.group(2)}</div>')
            continue
        m = re.match(r"^\s*[-*+]\s+(.*)", esc)
        if m:
            if not in_list:
                out.append("<ul style='margin:2px 0 2px 14px;'>")
                in_list = True
            out.append(f"<li>{m.group(1)}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append("<br>" if not esc.strip() else f"<div>{esc}</div>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


class ChatView(QTextBrowser):
    followChanged = Signal(bool)
    actionRequested = Signal(str, int)      # action name, reply number

    """Streams raw tokens, then swaps them for rendered markdown when the turn
    closes. The assistant label only appears once something is actually written,
    so a run of back-to-back tool calls doesn't leave empty headers behind."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFrameStyle(0)
        self.document().setDocumentMargin(18)
        self._anchor: int | None = None
        self._pre_label = 0
        self._open_pending = False
        self._thought_anchor: int | None = None
        self._thought_pre = 0
        self._thought_buf = ""
        # Entries are recorded so the transcript can be rebuilt when the palette
        # changes; inline HTML styles do not follow a stylesheet swap.
        self._log: list[tuple] = []
        self._replaying = False
        self._reply_no = 0
        self._reply_text: dict[int, str] = {}
        self._thought_no = 0
        self._thought_full: dict[int, str] = {}
        self._thought_open: set[int] = set()
        # Links are the only clickable element a text document offers; buttons
        # cannot be embedded in the flow.
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._anchor_clicked)
        # Following means "keep the newest text in view". It is switched off the
        # moment the reader scrolls up, because yanking the view back to the
        # bottom mid-sentence makes output impossible to read while it streams.
        self.following = True
        self._saved_scroll: int | None = None
        self._writing = False
        self._reply_no = 0
        self._reply_text: dict[int, str] = {}
        self._thought_no = 0
        self._thought_full: dict[int, str] = {}
        self._thought_open: set[int] = set()
        # Links are the only clickable element QTextEdit offers; buttons cannot
        # be embedded in the document flow.
        self.setOpenLinks(False)          # QTextBrowser: handle them ourselves
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._anchor_clicked)
        self.verticalScrollBar().valueChanged.connect(self._scrolled)

    def _anchor_clicked(self, url) -> None:
        text = url.toString()
        if not text.startswith("kestrel:"):
            return
        _, _, rest = text.partition(":")
        action, _, index = rest.partition(":")
        number = int(index or 0)
        if action == "thought":
            # Toggling is a view change, so the transcript is simply redrawn
            # from its own record rather than patched in place.
            self._thought_open ^= {number}
            self.rerender()
            return
        self.actionRequested.emit(action, number)

    def _actions_html(self, index: int) -> str:
        """A row of actions under a reply."""
        links = [("retry", "retry"), ("speak", "read aloud"),
                 ("copy", "copy"), ("fork", "fork from here")]
        parts = [f'<a href="kestrel:{name}:{index}" '
                 f'style="color:{theme.TEXT_DIM};text-decoration:none;">{label}</a>'
                 for name, label in links]
        return (f'<div style="font-size:10px;color:{theme.TEXT_DIM};'
                f'margin-top:6px;">' + '  ·  '.join(parts) + '</div>')

    def _scrolled(self, value: int) -> None:
        # Writing text moves the cursor, which moves the scrollbar. Those moves
        # are ours, not the reader's, and must not be mistaken for intent.
        if self._writing:
            return
        bar = self.verticalScrollBar()
        at_bottom = value >= bar.maximum() - 4
        if at_bottom != self.following:
            self.following = at_bottom
            self.followChanged.emit(at_bottom)

    def set_following(self, follow: bool) -> None:
        self.following = follow
        if follow:
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())
        self.followChanged.emit(follow)

    def _keep_in_view(self) -> None:
        """Pin to the newest text, or hold the reader's position.

        QTextEdit.append and insertText both scroll to the cursor on their own,
        so holding position means actively restoring it after every write, not
        merely declining to scroll.
        """
        bar = self.verticalScrollBar()
        if self.following:
            bar.setValue(bar.maximum())
        elif self._saved_scroll is not None:
            bar.setValue(self._saved_scroll)
        self._saved_scroll = None
        self._writing = False

    def _mark_scroll(self) -> None:
        self._writing = True
        if not self.following:
            self._saved_scroll = self.verticalScrollBar().value()

    def _end(self) -> QTextCursor:
        """A cursor at the end of the document.

        When following, this is also the widget's cursor, so Qt keeps it in
        view. When not following, it is a detached document cursor: moving the
        widget's cursor makes Qt scroll to it — asynchronously, so restoring the
        scrollbar afterwards does not hold — and writing through a detached one
        avoids the problem rather than fighting it.
        """
        c = QTextCursor(self.document())
        c.movePosition(QTextCursor.End)
        if self.following:
            self.setTextCursor(c)
        return c

    def _append(self, html_text: str) -> None:
        # append() starts a fresh paragraph; insertHtml at the cursor would merge
        # the new entry into whatever block is already open.
        self._mark_scroll()
        if self.following:
            self.append(html_text)
        else:
            c = QTextCursor(self.document())
            c.movePosition(QTextCursor.End)
            if self.document().characterCount() > 1:
                c.insertBlock()
            c.insertHtml(html_text)
        self._end()
        self._keep_in_view()

    def label(self, text: str, color: str) -> str:
        """Speaker labels in brackets: the transcript interleaves three voices
        and a bare word is easy to read as content."""
        return (f'<div style="color:{color};font-size:11px;letter-spacing:0.6px;'
                f'margin:14px 0 3px;"><b>[{html.escape(text)}]</b></div>')

    @staticmethod
    def bubble(speaker: str, body_html: str, colour: str, background: str,
               align: str = "left", width: str = "78%", italic: bool = False) -> str:
        """One message as a bubble.

        QTextEdit understands a narrow slice of HTML: no flexbox, no margin
        auto. A full-width table with an aligned cell is the one construction
        that reliably puts a block on the right, and a nested table gives it a
        background that hugs the text.
        """
        style = "font-style:italic;" if italic else ""
        # `align` on the table itself is the attribute Qt honours; wrapping it in
        # an aligned cell does not move it.
        return (
            f'<table align="{align}" width="{width}" cellspacing="0" '
            f'cellpadding="9" bgcolor="{background}" '
            f'style="margin-top:10px;margin-bottom:10px;"><tr><td>'
            f'<div style="color:{colour};font-size:11px;letter-spacing:0.6px;">'
            f'<b>[{html.escape(speaker)}]</b></div>'
            f'<div style="color:{theme.TEXT};{style}">{body_html}</div>'
            f'</td></tr></table>')

    def add_user(self, text: str) -> None:
        self._record("add_user", text)
        self._append(self.bubble("You", md_to_html(text), theme.SIGNAL,
                                 theme.BUBBLE_YOU, align="right", width="72%"))

    def add_note(self, text: str, color: str = "") -> None:
        self._record("add_note", text, color)
        self._append(f'<div style="color:{color or theme.TEXT_DIM};font-size:11px;'
                     f'margin:6px 0;">{html.escape(text)}</div>')

    def add_error(self, text: str) -> None:
        self._record("add_error", text)
        self._append(f'<div style="color:{theme.ALERT};margin:8px 0;">'
                     f'{html.escape(text)}</div>')

    def add_tool_call(self, name: str, preview: str) -> None:
        self._record("add_tool_call", name, preview)
        if self._thought_anchor is not None:
            self._collapse_thought()
        self._append(
            f'<div style="margin:8px 0 2px;color:{theme.SIGNAL};font-size:11px;">'
            f'&#9654;&nbsp;<b>{html.escape(name)}</b> '
            f'<span style="color:{theme.TEXT_DIM};">{html.escape(preview)}</span></div>'
        )

    def add_tool_result(self, name: str, ok: bool, text: str, max_lines: int = 8) -> None:
        self._record("add_tool_result", name, ok, text, max_lines)
        lines = text.splitlines()
        shown = lines[:max_lines]
        tail = f"\n… {len(lines) - max_lines} more lines — see Activity" if len(lines) > max_lines else ""
        body = html.escape("\n".join(shown) + tail)
        colour = theme.TEXT_DIM if ok else theme.ALERT
        self._append(
            f'<pre style="margin:0 0 4px 14px;color:{colour};font-size:11px;'
            f'white-space:pre-wrap;">{body}</pre>'
        )

    def stream_thought(self, chunk: str) -> None:
        """Reasoning streams into a dim band that collapses to a summary line the
        moment the model starts producing its actual answer."""
        if self._thought_anchor is None:
            self._thought_pre = self._end().position()
            self._append(self.label("Thinking", theme.THINK))
            c = self._end()
            c.insertBlock()
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(theme.THINK))
            c.setBlockCharFormat(QTextCharFormat())
            c.setCharFormat(fmt)
            self.setTextCursor(c)
            self._thought_anchor = c.position()
            self._thought_buf = ""
        self._thought_buf += chunk
        self._mark_scroll()
        c = self._end()
        c.insertText(chunk)
        self._keep_in_view()

    def _collapse_thought(self, text: str = "", tokens: int = 0) -> None:
        """Replace the streamed trace with one summary line.

        Only ever called while the trace is the last thing in the document —
        either when the first answer token arrives, or when the turn ends with
        no answer at all. Deleting to the end at any other moment would take the
        assistant's reply with it.
        """
        if self._thought_anchor is None:
            return
        body = text or self._thought_buf
        if not tokens:
            tokens = max(1, int(len(body) / 3.7))
            approx = "~"
        else:
            approx = ""
        # The whole trace is kept so it can be expanded later; the head is what
        # is shown until then.
        full = body.strip()
        head = " ".join(body.split())
        if len(head) > 200:
            head = head[:199] + "\u2026"
        c = QTextCursor(self.document())
        c.setPosition(self._thought_pre)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.removeSelectedText()
        self.setTextCursor(c)
        self._thought_anchor = None
        self._thought_buf = ""
        self._record("_thought_line", head, tokens, approx, full)
        self._thought_line(head, tokens, approx, full)

    def _thought_line(self, head: str, tokens: int, approx: str = "",
                      full: str = "") -> None:
        """A thought, shown short with the whole of it one click away.

        The trace is the most interesting thing in the transcript when something
        has gone wrong and the least interesting when it has not, so it is
        collapsed by default rather than hidden or shown whole.
        """
        self._thought_no += 1
        index = self._thought_no
        whole = full or head
        self._thought_full[index] = whole
        expanded = index in self._thought_open
        shown = whole if expanded else _first_lines(whole, 2)
        more = len(whole) > len(shown)
        label = "less" if expanded else "more"
        toggle = (f'  <a href="kestrel:thought:{index}" '
                  f'style="color:{theme.THINK};text-decoration:none;">'
                  f'[{label}]</a>') if (more or expanded) else ""
        body = (f'<div style="color:{theme.TEXT_DIM};font-size:11px;'
                f'white-space:pre-wrap;">{html.escape(shown)}</div>'
                f'<div style="color:{theme.TEXT_DIM};font-size:10px;">'
                f'{approx}{tokens} tokens · not resent{toggle}</div>')
        self._append(self.bubble("Thinking", body, theme.THINK,
                                 theme.BUBBLE_THINK, width="70%", italic=True))

    def end_thought(self, text: str, tokens: int) -> None:
        """Called when the turn's generation finishes. If the trace already
        collapsed because an answer followed it, there is nothing left to do."""
        if self._thought_anchor is not None:
            self._collapse_thought(text, tokens)

    def thinking_open(self) -> bool:
        return self._thought_anchor is not None

    def begin_assistant(self) -> None:
        self._open_pending = True
        self._anchor = None

    def _open(self) -> None:
        if not self._open_pending:
            return
        self._pre_label = self._end().position()
        # Streamed text is plain until the turn closes, when it is replaced by
        # the bubble: a bubble cannot grow a word at a time in QTextEdit.
        self._append(self.label("Kestrel", theme.AMBER))
        c = self._end()
        c.insertBlock()
        # The label sets a small bold format; reset so streamed text is body copy.
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(theme.TEXT))
        c.setBlockCharFormat(QTextCharFormat())
        c.setCharFormat(fmt)
        if self.following:
            self.setTextCursor(c)
        self._anchor = c.position()
        self._open_pending = False

    def stream(self, chunk: str) -> None:
        if self._thought_anchor is not None:
            self._collapse_thought()
        self._open()
        self._mark_scroll()
        c = self._end()
        c.insertText(chunk)
        self._keep_in_view()

    def streaming(self) -> bool:
        return self._anchor is not None

    def streamed_text(self) -> str:
        if self._anchor is None:
            return ""
        c = QTextCursor(self.document())
        c.setPosition(self._anchor)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        return c.selectedText().replace("\u2029", "\n")

    def _record(self, kind: str, *args) -> None:
        if not self._replaying:
            self._log.append((kind, args))

    def reply_text(self, index: int) -> str:
        return self._reply_text.get(index, "")

    def _assistant_block(self, final: str) -> None:
        self._reply_no += 1
        self._reply_text[self._reply_no] = final
        self._append(self.bubble("Kestrel",
                                 md_to_html(final) + self._actions_html(self._reply_no),
                                 theme.AMBER, theme.BUBBLE_AI))

    def rerender(self) -> None:
        """Rebuild the transcript in the current palette."""
        entries = list(self._log)
        self._replaying = True
        try:
            self.clear()
            self._anchor = None
            self._open_pending = False
            self._thought_anchor = None
            self._thought_buf = ""
            self._reply_no = 0      # replay reproduces the same numbering
            self._thought_no = 0
            for kind, args in entries:
                getattr(self, kind)(*args)
        finally:
            self._replaying = False
            self._log = entries
        self._end()

    def clear_log(self) -> None:
        self._log.clear()

    def discard_open(self) -> None:
        """Roll back an opened turn that produced nothing but a tool call, so the
        transcript doesn't accumulate empty headers."""
        if self._anchor is None:
            return
        c = QTextCursor(self.document())
        c.setPosition(self._pre_label)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.removeSelectedText()
        if self.following:
            self.setTextCursor(c)
        self._anchor = None
        self._open_pending = True

    def end_assistant(self, final: str) -> None:
        """Swap the raw streamed text for rendered markdown."""
        self._record("_assistant_block", final)
        self._open()
        if self._anchor is None:
            # Nothing was streamed for this turn, so there is no region to
            # replace. Appending is correct; selecting from position 0 would
            # delete the entire transcript.
            self._assistant_block(final)
            return
        c = QTextCursor(self.document())
        # From before the label, not after it: the bubble carries its own, and
        # replacing only the streamed text leaves the old one stranded above.
        c.setPosition(self._pre_label)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.removeSelectedText()
        self._reply_no += 1
        self._reply_text[self._reply_no] = final
        c.insertHtml(self.bubble("Kestrel",
                                 md_to_html(final) + self._actions_html(self._reply_no),
                                 theme.AMBER, theme.BUBBLE_AI))
        self._end()
        self._keep_in_view()
        self._anchor = None
        self._open_pending = True


# --------------------------------------------------------------- activity --
class ActivityTree(QTreeWidget):
    """Every tool call, with the complete untruncated output behind an expander."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Step", "Detail"])
        self.setAlternatingRowColors(True)
        stretch_columns(self, first_stretch=1)
        self.setFont(mono_font(10))
        self._current: QTreeWidgetItem | None = None

    def add_note(self, label: str, detail: str) -> None:
        item = QTreeWidgetItem([label, detail])
        item.setForeground(0, QColor(theme.VIOLET))
        self.addTopLevelItem(item)
        self.scrollToBottom()

    def add_call(self, name: str, args: dict) -> None:
        preview = ", ".join(f"{k}={str(v)[:50]}" for k, v in args.items())
        item = QTreeWidgetItem([name, preview])
        item.setForeground(0, QColor(theme.SIGNAL))
        self.addTopLevelItem(item)
        self._current = item
        self.scrollToBottom()

    def add_result(self, name: str, ok: bool, text: str) -> None:
        parent = self._current
        if parent is None or parent.text(0) != name:
            parent = QTreeWidgetItem([name, ""])
            self.addTopLevelItem(parent)
        head = (text.splitlines() or [""])[0][:120]
        parent.setText(1, parent.text(1) + f"   →  {head}")
        parent.setForeground(0, QColor(theme.SIGNAL if ok else theme.ALERT))
        for line in text.splitlines()[:400] or ["(no output)"]:
            child = QTreeWidgetItem(["", line[:400]])
            child.setForeground(1, QColor(theme.TEXT_DIM))
            parent.addChild(child)
        self.scrollToBottom()


class Field(QWidget):
    """Label above control, mono eyebrow styling."""

    def __init__(self, label: str, widget: QWidget, hint: str = "", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lab = QLabel(label)
        lab.setObjectName("Eyebrow")
        lay.addWidget(lab)
        lay.addWidget(widget)
        if hint:
            h = QLabel(hint)
            h.setObjectName("Dim")
            h.setWordWrap(True)
            h.setStyleSheet("font-size:11px;")
            lay.addWidget(h)
        self.widget = widget


class Readout(QWidget):
    """A labelled value that does not drag the panel wider than it should be.

    A full filesystem path in a plain QLabel reports a minimum width equal to
    its own text, and a panel is at least as wide as the widest thing in it —
    so one long path was enough to force a horizontal scrollbar across the whole
    tab. The value is elided to fit and kept in full in the tooltip.
    """

    def __init__(self, label: str, value: str = "—", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        name = QLabel(label)
        name.setObjectName("Eyebrow")
        name.setMinimumWidth(96)
        name.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.value = QLabel(value)
        self.value.setObjectName("Readout")
        self.value.setMinimumWidth(0)
        self.value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._full = value
        lay.addWidget(name)
        lay.addWidget(self.value, 1)

    def set(self, text: str) -> None:
        self._full = str(text)
        self.value.setToolTip(self._full)
        self._elide()

    def _elide(self) -> None:
        metrics = self.value.fontMetrics()
        width = max(40, self.value.width())
        # The middle goes, not the end: the interesting parts of a path are the
        # start and the filename.
        self.value.setText(metrics.elidedText(self._full, Qt.ElideMiddle, width))

    def resizeEvent(self, event):  # noqa: N802
        self._elide()
        super().resizeEvent(event)


class CollapseHandle(QWidget):
    """A thin vertical strip with a chevron at its midpoint.

    Sits between a side panel and the centre column. Clicking anywhere on the
    strip collapses or restores the panel, so the target is the full height of
    the window rather than a small button someone has to aim at.
    """

    toggled = Signal(bool)          # True when the panel is now visible

    def __init__(self, side: str = "left", parent=None):
        super().__init__(parent)
        self.side = side
        self.expanded = True
        self._hover = False
        self.setFixedWidth(14)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip("Hide this panel")

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.setToolTip("Hide this panel" if expanded else "Show this panel")
        self.update()

    def enterEvent(self, event):  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event):  # noqa: N802
        self._hover = False
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.set_expanded(not self.expanded)
            self.toggled.emit(self.expanded)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(theme.PANEL if self._hover else theme.INK))

        line = QPen(QColor(theme.LINE))
        line.setWidth(1)
        p.setPen(line)
        x = w // 2
        p.drawLine(x, 8, x, h // 2 - 22)
        p.drawLine(x, h // 2 + 22, x, h - 8)

        # the chevron, pointing the way the panel will move
        pen = QPen(QColor(theme.AMBER if self._hover else theme.TEXT_DIM))
        pen.setWidth(2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        mid = h // 2
        pointing_left = (self.side == "left") == self.expanded
        dx = 3 if pointing_left else -3
        p.drawLine(x + dx, mid - 6, x - dx, mid)
        p.drawLine(x - dx, mid, x + dx, mid + 6)
        p.end()


def stretch_columns(tree, first_stretch: int = 0) -> None:
    """Let a tree fill, and shrink to, whatever width it is given.

    Fixed pixel columns look deliberate at the width they were chosen for and
    wrong at every other. `ResizeToContents` alone is no better: it grows to the
    longest cell and drags the panel wider, so the elastic column takes the
    slack and the rest are merely allowed to shrink.
    """
    header = tree.header()
    # The last section takes the slack. Without this the rightmost column is
    # whatever is left over, which on a narrow panel is nothing at all — and a
    # divider handle sits on a column's right edge, so there is no way to drag
    # the last one wider either.
    header.setStretchLastSection(True)
    header.setMinimumSectionSize(24)
    header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    last = tree.columnCount() - 1
    for i in range(tree.columnCount()):
        if i == last:
            mode = QHeaderView.Stretch          # absorbs the remaining width
        elif i == first_stretch:
            mode = QHeaderView.Stretch
        else:
            mode = QHeaderView.Interactive
        header.setSectionResizeMode(i, mode)
        if mode is QHeaderView.Interactive:
            tree.resizeColumnToContents(i)
    tree.setMinimumWidth(0)
    tree.setSizePolicy(QSizePolicy.Ignored, tree.sizePolicy().verticalPolicy())
    tree.setTextElideMode(Qt.ElideRight)


def glyph_icon(kind: str, size: int = 20, colour: str = "") -> QIcon:
    key = (kind, size, colour or theme.current)
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached
    icon = _draw_glyph(kind, size, colour)
    _ICON_CACHE[key] = icon
    return icon


def _draw_glyph(kind: str, size: int = 20, colour: str = "") -> QIcon:
    """Draw a small symbolic icon.

    Drawn rather than shipped: an icon font would be another dependency and
    another source of missing-glyph boxes, and these are simple enough that
    vector strokes are clearer at small sizes than a font would be.
    """
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(colour or theme.TEXT_DIM))
    pen.setWidthF(2.2)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    # The pixmap is 2x physical but QPainter works in logical coordinates, so
    # the canvas is `size`, not `size * 2`. Painting at physical scale here
    # drew each glyph four times too large and clipped away most of it.
    s2 = size
    m = s2 * 0.22          # margin
    a, b = m, s2 - m       # inner box
    mid = s2 / 2

    if kind == "status":                      # gauge dial
        p.drawArc(QRectF(a, a + 2, b - a, b - a), 30 * 16, 120 * 16)
        p.drawLine(int(mid), int(mid + 2), int(b - 3), int(a + 5))
    elif kind == "models":                     # stacked layers
        for i, y in enumerate((a + 2, mid, b - 2)):
            p.drawLine(int(a), int(y), int(b), int(y))
    elif kind == "params":                     # sliders
        p.drawLine(int(a), int(a + 3), int(b), int(a + 3))
        p.drawLine(int(a), int(mid + 1), int(b), int(mid + 1))
        p.drawLine(int(a), int(b - 1), int(b), int(b - 1))
        for x, y in ((mid + 4, a + 3), (a + 5, mid + 1), (mid + 1, b - 1)):
            p.drawEllipse(QRectF(x - 2.5, y - 2.5, 5, 5))
    elif kind == "cluster":                    # three linked nodes
        pts = [(mid, a + 2), (a + 2, b - 2), (b - 2, b - 2)]
        for x, y in pts:
            p.drawEllipse(QRectF(x - 3, y - 3, 6, 6))
        p.drawLine(int(pts[0][0]), int(pts[0][1] + 3), int(pts[1][0]), int(pts[1][1] - 3))
        p.drawLine(int(pts[0][0]), int(pts[0][1] + 3), int(pts[2][0]), int(pts[2][1] - 3))
    elif kind == "skills":                     # book
        p.drawLine(int(mid), int(a + 2), int(mid), int(b - 2))
        p.drawArc(QRectF(a, a, mid - a, b - a), -90 * 16, 180 * 16)
        p.drawArc(QRectF(mid, a, b - mid, b - a), 90 * 16, 180 * 16)
    elif kind == "memory":                     # database
        p.drawEllipse(QRectF(a, a + 1, b - a, (b - a) * 0.32))
        p.drawLine(int(a), int(a + 1 + (b - a) * 0.16), int(a), int(b - 4))
        p.drawLine(int(b), int(a + 1 + (b - a) * 0.16), int(b), int(b - 4))
        p.drawArc(QRectF(a, b - 4 - (b - a) * 0.16, b - a, (b - a) * 0.32), 180 * 16, 180 * 16)
    elif kind == "persona":                    # head and shoulders
        p.drawEllipse(QRectF(mid - 4.5, a + 1, 9, 9))
        p.drawArc(QRectF(a + 1, mid + 1, b - a - 2, (b - a)), 20 * 16, 140 * 16)
    elif kind == "speech":                     # waveform
        for i, h in enumerate((4, 9, 6, 11, 5)):
            x = a + i * (b - a) / 4.4 + 2
            p.drawLine(int(x), int(mid - h / 2), int(x), int(mid + h / 2))
    elif kind == "backend":                    # chip
        p.drawRect(QRectF(a + 2, a + 2, b - a - 4, b - a - 4))
        for i in range(3):
            y = a + 5 + i * (b - a - 10) / 2
            p.drawLine(int(a - 1), int(y), int(a + 2), int(y))
            p.drawLine(int(b - 2), int(y), int(b + 1), int(y))
    elif kind == "projects":                   # folder
        p.drawLine(int(a), int(a + 3), int(mid - 1), int(a + 3))
        p.drawLine(int(mid - 1), int(a + 3), int(mid + 1), int(a + 6))
        p.drawRect(QRectF(a, a + 6, b - a, b - a - 8))
    elif kind == "tools":                      # spanner
        p.drawLine(int(a + 3), int(b - 3), int(b - 5), int(a + 5))
        p.drawArc(QRectF(b - 9, a, 9, 9), 40 * 16, 260 * 16)
        p.drawEllipse(QRectF(a + 1, b - 6, 5, 5))
    elif kind == "monitor":                    # bar chart
        for i, h in enumerate((5, 10, 7, 12)):
            x = a + 1 + i * (b - a - 2) / 3.4
            p.drawLine(int(x), int(b - 1), int(x), int(b - 1 - h))
    elif kind == "plan":                       # checklist
        for i in range(3):
            y = a + 2 + i * (b - a - 4) / 2
            p.drawLine(int(a + 5), int(y), int(b), int(y))
            p.drawEllipse(QRectF(a - 1, y - 1.6, 3.2, 3.2))
    elif kind == "activity":                   # pulse
        p.drawLine(int(a), int(mid), int(a + 4), int(mid))
        p.drawLine(int(a + 4), int(mid), int(mid - 1), int(a + 3))
        p.drawLine(int(mid - 1), int(a + 3), int(mid + 2), int(b - 3))
        p.drawLine(int(mid + 2), int(b - 3), int(mid + 5), int(mid))
        p.drawLine(int(mid + 5), int(mid), int(b), int(mid))
    elif kind == "log":                        # lines of text
        for i, width in enumerate((1.0, 0.7, 0.9, 0.5)):
            y = a + 1 + i * (b - a - 2) / 3
            p.drawLine(int(a), int(y), int(a + (b - a) * width), int(y))
    else:
        p.drawEllipse(QRectF(a, a, b - a, b - a))
    p.end()
    return QIcon(pix)


class IconTabBar(QTabBar):
    """A vertical rail of upright icons.

    Qt rotates tab contents for West and East tab positions, which turns a
    symbolic icon on its side and clips it. Drawing the bar directly keeps the
    icons upright and gives room for a selection marker.
    """

    WIDTH = 46
    HEIGHT = 42

    def __init__(self, kinds: list[str], parent=None):
        super().__init__(parent)
        self.kinds = kinds
        self._hover = -1
        self.setDrawBase(False)
        self.setMouseTracking(True)
        self.setUsesScrollButtons(False)
        self.setExpanding(False)

    def tabSizeHint(self, index: int) -> QSize:  # noqa: N802
        return QSize(self.WIDTH, self.HEIGHT)

    def minimumTabSizeHint(self, index: int) -> QSize:  # noqa: N802
        return QSize(self.WIDTH, self.HEIGHT)

    def mouseMoveEvent(self, event):  # noqa: N802
        index = self.tabAt(event.position().toPoint())
        if index != self._hover:
            self._hover = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self._hover = -1
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.INK))
        for i in range(self.count()):
            rect = self.tabRect(i)
            selected = i == self.currentIndex()
            hovered = i == self._hover
            if selected:
                p.fillRect(rect, QColor(theme.PANEL))
                p.fillRect(QRectF(rect.left(), rect.top() + 6, 2.5,
                                  rect.height() - 12), QColor(theme.AMBER))
            elif hovered:
                p.fillRect(rect, QColor(theme.PANEL_HI))
            kind = self.kinds[i] if i < len(self.kinds) else ""
            colour = theme.AMBER if selected else (
                theme.TEXT if hovered else theme.TEXT_DIM)
            pix = glyph_icon(kind, 21, colour).pixmap(QSize(21, 21))
            p.drawPixmap(rect.center().x() - 10, rect.center().y() - 10, pix)
        p.end()


class TypingIndicator(QWidget):
    """Three pulsing dots while the model works.

    Generation begins with a silence that can run to tens of seconds on a large
    model — prompt ingestion, then reasoning before the first visible token. An
    idle window during that period is indistinguishable from a hung one, which
    is the actual complaint behind "it freezes".
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._phase = 0.0
        self._label = "thinking"
        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self, label: str = "thinking") -> None:
        self._label = label
        self._phase = 0.0
        self.show()
        self._timer.start()

    def set_label(self, label: str) -> None:
        if label != self._label:
            self._label = label
            self.update()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.14) % 1.0
        self.update()

    def paintEvent(self, event):  # noqa: N802
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(mono_font(9))
        metrics = p.fontMetrics()
        p.setPen(QColor(theme.TEXT_DIM))
        text = f"Kestrel is {self._label}"
        p.drawText(2, metrics.ascent() + 3, text)

        x = 6 + metrics.horizontalAdvance(text)
        y = self.height() / 2
        for i in range(3):
            # A travelling wave rather than three dots blinking together: the
            # direction of travel reads as progress.
            offset = math.sin((self._phase * 2 * math.pi) - i * 0.7)
            alpha = 90 + int(120 * max(0.0, offset))
            radius = 2.0 + 1.1 * max(0.0, offset)
            colour = QColor(theme.AMBER)
            colour.setAlpha(alpha)
            p.setBrush(colour)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(x + i * 9 - radius, y - radius - 1,
                                 radius * 2, radius * 2))
        p.end()


class BusyOverlay(QWidget):
    """A working indicator drawn over the transcript.

    Loading a model is the longest wait in the application and the one that most
    looks like a crash: llama.cpp maps tens of gigabytes, and the window has
    nothing to show meanwhile. This reports the stage and the server's own most
    recent output, so the wait is legibly progress rather than a hang.
    """

    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        self._title = "Working"
        self._detail = ""
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def attach(self, host: QWidget) -> None:
        self.setParent(host)
        host.installEventFilter(self)
        self.setGeometry(host.rect())

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() == QEvent.Resize and obj is self.parent():
            self.setGeometry(obj.rect())
        return False

    def begin(self, title: str, detail: str = "") -> None:
        self._title, self._detail = title, detail
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()
        self._timer.start()

    def update_detail(self, detail: str) -> None:
        if self.isVisible():
            self._detail = " ".join(str(detail).split())[:150]
            self.update()

    def end(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.02) % 1.0
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        event.accept()          # swallow clicks on whatever is underneath

    def paintEvent(self, event):  # noqa: N802
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        veil = QColor(theme.INK)
        veil.setAlpha(228)
        p.fillRect(self.rect(), veil)

        cx, cy = self.width() / 2, self.height() / 2 - 20
        track = QRectF(cx - 19, cy - 19, 38, 38)
        pen = QPen(QColor(theme.LINE))
        pen.setWidth(3)
        p.setPen(pen)
        p.drawEllipse(track)
        pen.setColor(QColor(theme.AMBER))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        span = 90 + 40 * math.sin(self._phase * 2 * math.pi)
        p.drawArc(track, int(-self._phase * 360 * 16), int(-span * 16))

        p.setFont(mono_font(10))
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(10, cy + 32, self.width() - 20, 20),
                   Qt.AlignHCenter, self._title)
        if self._detail:
            p.setFont(mono_font(8))
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(QRectF(14, cy + 54, self.width() - 28, 34),
                       Qt.AlignHCenter | Qt.TextWordWrap, self._detail)
        p.end()


class LazyTab(QWidget):
    """A tab whose contents are built the first time it is shown.

    Start-up time was the sum of every panel's constructor, and those constructors
    read directories, open databases and probe for binaries. Deferring them makes
    opening the window independent of how slow any one panel is — and most panels
    are never looked at in a given session.
    """

    built = Signal(object)

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self._factory = factory
        self._panel: QWidget | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

    @property
    def panel(self) -> QWidget | None:
        return self._panel

    def ensure(self) -> QWidget | None:
        if self._panel is None:
            try:
                self._panel = self._factory()
            except Exception as e:                      # a broken panel must not
                label = QLabel(f"This panel failed to load:\n{e}")  # take the window
                label.setWordWrap(True)
                self._panel = label
            self.layout().addWidget(self._panel)
            self.built.emit(self._panel)
        return self._panel

    def showEvent(self, event):  # noqa: N802
        self.ensure()
        super().showEvent(event)


class GpuSplit(QWidget):
    """Choose how much of a model sits in graphics memory and how much in RAM.

    llama.cpp offloads whole layers, so the split is a count rather than a
    percentage — but a count means nothing without knowing what it costs. The
    bar shows both sides in gigabytes as the handle moves, and marks the point
    beyond which the graphics device runs out, because exceeding it is what
    makes a model fail to load rather than run slowly.
    """

    changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_layer = 0
        self.per_layer_mb = 0.0
        self.vram_mb = 0
        self.system_mb = 0
        self.fits = 0
        self.integrated = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.bar = _SplitBar(self)
        lay.addWidget(self.bar)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumHeight(22)      # room for the handle to be grabbed
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._moved)
        lay.addWidget(self.slider)

        self.caption = QLabel("No model selected.")
        self.caption.setObjectName("Dim")
        self.caption.setWordWrap(True)
        lay.addWidget(self.caption)

    def set_model(self, info, ctx: int, vram_mb: int, system_mb: int,
                  integrated: bool, value: int) -> None:
        self.n_layer = info.n_layer if info else 0
        self.vram_mb, self.system_mb, self.integrated = vram_mb, system_mb, integrated
        if info and info.n_layer and info.file_size:
            weights = (info.file_size / (1024 ** 2)) / info.n_layer
            kv = 0.0
            if info.n_embd and ctx:
                kv = (2 * info.n_embd * ctx * 2) / (1024 ** 2) / 1
            self.per_layer_mb = weights + kv
        else:
            self.per_layer_mb = 0.0
        from ..gguf import layers_that_fit
        self.fits = (layers_that_fit(info, vram_mb, ctx, integrated=integrated,
                                     system_mb=system_mb) if info else 0)
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self.n_layer))
        self.slider.setValue(max(0, min(self.n_layer, value)))
        self.slider.blockSignals(False)
        self._refresh()

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, layers: int) -> None:
        self.slider.setValue(max(0, min(self.n_layer, layers)))

    def _moved(self, value: int) -> None:
        self._refresh()
        self.changed.emit(value)

    def _refresh(self) -> None:
        layers = self.slider.value()
        gpu_mb = layers * self.per_layer_mb
        ram_mb = max(0, (self.n_layer - layers)) * self.per_layer_mb
        self.bar.set_split(layers, self.n_layer, self.fits)
        if not self.n_layer:
            self.caption.setText("No model selected — pick one in the Models tab.")
            return
        over = ""
        if self.fits and layers > self.fits:
            over = ("  ·  beyond what the device reported; it may fail to "
                    "allocate" if not self.integrated else
                    "  ·  beyond the shared budget; it may fail to allocate")
        shared = (" (shared with system memory)" if self.integrated else "")
        self.caption.setText(
            f"GPU {gpu_mb / 1024:.1f} GB{shared}   ·   system RAM "
            f"{ram_mb / 1024:.1f} GB   ·   {layers} of {self.n_layer} layers"
            f"{over}")


class _SplitBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.layers = 0
        self.total = 0
        self.fits = 0

    def set_split(self, layers: int, total: int, fits: int) -> None:
        self.layers, self.total, self.fits = layers, total, fits
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.fillRect(QRectF(0, 3, w, h - 13), QColor(theme.PANEL))
        if not self.total:
            p.end()
            return
        cut = w * self.layers / self.total
        p.fillRect(QRectF(0, 3, cut, h - 13), QColor(theme.SIGNAL))
        p.fillRect(QRectF(cut, 3, w - cut, h - 13), QColor(theme.PLAN))

        if self.fits and self.fits < self.total:
            # Where the device stops being able to hold any more.
            x = w * self.fits / self.total
            pen = QPen(QColor(theme.ALERT))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(int(x), 1, int(x), h - 6)

        p.setFont(mono_font(8))
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(2, h - 10, w / 2, 10), Qt.AlignLeft, "GPU")
        p.drawText(QRectF(w / 2, h - 10, w / 2 - 2, 10), Qt.AlignRight, "system RAM")
        p.end()


class WheelGuard(QObject):
    """Stops the mouse wheel changing values while scrolling a panel.

    A spin box or drop-down under the pointer swallows the wheel and edits
    itself, so scrolling down a settings page silently changes the first control
    it passes over. Qt does this by default and it is a genuine hazard: the
    value that changed is the one you were not looking at.

    Unfocused controls therefore hand the wheel to the panel behind them. Click
    one first and it scrolls normally.
    """

    VALUE_WIDGETS = (QAbstractSpinBox, QComboBox, QSlider)
    TEXT_WIDGETS = (QTextEdit, QTextBrowser, QPlainTextEdit)

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() != QEvent.Wheel:
            return False
        widget = obj if isinstance(obj, QWidget) else None
        if widget is None:
            return False
        # A wheel event arrives at the viewport, not the widget that owns it,
        # so the ancestry is walked to find what is really being scrolled.
        # Comparing against `viewport()` directly is unreliable: two PySide
        # wrappers for the same C++ object are not necessarily the same Python
        # object, so an identity test silently fails.
        owner = widget
        while owner is not None and not isinstance(
                owner, self.VALUE_WIDGETS + self.TEXT_WIDGETS + (QAbstractScrollArea,)):
            owner = owner.parentWidget()
        if owner is None:
            return False

        if isinstance(owner, self.VALUE_WIDGETS):
            # Click-only, always. Focus makes no difference: a value should
            # never change because the pointer happened to be over it.
            self._pass_upwards(owner, event)
            return True
        if isinstance(owner, self.TEXT_WIDGETS):
            # Hand it to the page. Only where there is no page behind it does
            # the box scroll itself, which is what keeps the transcript and the
            # logs usable.
            return self._pass_upwards(owner, event)
        return False

    @staticmethod
    def _pass_upwards(widget, event) -> bool:
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                QApplication.sendEvent(parent.viewport(), event)
                return True
            parent = parent.parentWidget()
        return False


def install_wheel_guard(app) -> WheelGuard:
    """One filter for the whole application, rather than per widget.

    Installed on the application object so it covers controls built later —
    panels are created on demand, and a per-widget rule would have to be
    remembered every time one is added.
    """
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard


class WrappingDelegate(QStyledItemDelegate):
    """Lets a tree row grow to fit wrapped text.

    A QTreeWidget gives every row one line unless something tells it otherwise,
    so a step longer than the panel is simply cut off. This measures the text at
    the column's real width and asks for the height it needs.
    """

    def sizeHint(self, option, index):  # noqa: N802
        text = index.data() or ""
        width = max(60, option.rect.width() or
                    self.parent().columnWidth(index.column()) if self.parent() else 200)
        metrics = QFontMetrics(option.font)
        rect = metrics.boundingRect(0, 0, width - 8, 10_000,
                                    Qt.TextWordWrap | Qt.AlignLeft, str(text))
        base = super().sizeHint(option, index)
        return QSize(base.width(), max(base.height(), rect.height() + 8))

    def paint(self, painter, option, index):  # noqa: N802
        text = str(index.data() or "")
        if "\n" not in text and len(text) < 40:
            super().paint(painter, option, index)
            return
        painter.save()
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        colour = index.data(Qt.ForegroundRole)
        painter.setPen(colour.color() if colour else option.palette.text().color())
        painter.setFont(option.font)
        painter.drawText(option.rect.adjusted(4, 3, -4, -3),
                         Qt.TextWordWrap | Qt.AlignTop | Qt.AlignLeft, text)
        painter.restore()


def _first_lines(text: str, count: int = 2, limit: int = 220) -> str:
    """The opening of a trace: enough to recognise it, not enough to drown in."""
    lines = [l for l in str(text or "").splitlines() if l.strip()]
    head = " ".join(lines[:count])
    return head[:limit] + ("…" if len(head) > limit else "")
