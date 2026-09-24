"""The workspace: the chat beside panels that fold, tile and step aside.

Three rules, and every fold or unfold simply re-runs them:

1. The chat holds the left of the workspace and never folds. It is what
   Kestrel is; everything else is a view onto what it is doing.
2. Open panels stand side by side, each the full height, sharing the width
   equally. Stacked, three panels each got a third of the height and became
   squares too short to show a file or a plan; side by side, each keeps the
   full height and gives up width instead, which code and checklists spare
   more easily.
3. Folded panels become thin vertical strips along the right edge, each still
   showing its name and a one-line summary — so folding hides the detail, not
   the progress.

A panel can also be switched off altogether (Canvas and Plan from the top
bar), which removes it rather than folding it: a feature that is not in use
should not take a strip of the window to say so.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from . import theme


# Half the visible gap between two panels; each panel keeps this much clear
# on either side of its frame.
GAP = 6


class FoldPanel(QWidget):
    """A titled section that can be folded to a strip or switched off."""

    folded_changed = Signal(bool)
    MIN_WIDTH = 300

    def __init__(self, key: str, title: str, body: QWidget, parent=None):
        super().__init__(parent)
        self.key = key
        self.title = title
        self.body = body
        self.folded = False
        # Folded by the workspace for lack of room, not by the person. Only
        # these come back on their own when the window widens.
        self.auto_folded = False
        self.enabled_ = True
        self.summary = ""

        self.setObjectName("FoldPanel")
        self.setMinimumHeight(0)
        # Narrow enough that three fit beside the chat on a laptop screen,
        # wide enough that a line of code is still a line.
        self.setMinimumWidth(self.MIN_WIDTH)
        # So the stylesheet's background and border actually paint on a
        # plain QWidget, which otherwise ignores them.
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        # The gap between panels is drawn by each panel, not left to the
        # layout's spacing. When the window is narrower than everything's
        # minimum — Windows display scaling makes a 1920px screen 1536 wide
        # to Qt at 125% — a layout gives up its spacing first, and the panels
        # closed up against each other. A margin inside each panel cannot be
        # squeezed away. The stylesheet insets the frame by the same amount.
        lay.setContentsMargins(GAP + 1, 1, GAP + 1, 1)
        lay.setSpacing(0)

        self.banner = QWidget()
        self.banner.setObjectName("FoldBanner")
        bar = QHBoxLayout(self.banner)
        bar.setContentsMargins(6, 0, 10, 0)
        bar.setSpacing(8)
        self.fold_btn = QToolButton()
        self.fold_btn.setObjectName("FoldButton")
        self.fold_btn.setText("▾")
        self.fold_btn.setToolTip(f"Fold {title.lower()}")
        self.fold_btn.setAutoRaise(True)
        self.fold_btn.clicked.connect(lambda: self.set_folded(True))
        bar.addWidget(self.fold_btn)
        self.label = QLabel(title.upper())
        self.label.setObjectName("FoldTitle")
        bar.addWidget(self.label)
        bar.addStretch(1)
        self.detail = QLabel("")
        self.detail.setObjectName("FoldDetail")
        bar.addWidget(self.detail)
        self.banner.setFixedHeight(36)
        lay.addWidget(self.banner)
        # The body scrolls rather than squeezes. Three panels sharing a short
        # window leave each less than its controls need, and a layout forced
        # below its minimum overlaps its buttons into each other — scrolling
        # keeps every control whole and reachable.
        from PySide6.QtWidgets import QFrame, QScrollArea
        self.scroll = QScrollArea()
        self.scroll.setObjectName("FoldScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidget(body)
        lay.addWidget(self.scroll, 1)

    def add_banner_widget(self, widget) -> None:
        """Put a control in the banner, before the summary."""
        bar = self.banner.layout()
        bar.insertWidget(bar.count() - 1, widget)

    def set_title(self, title: str) -> None:
        self.title = title
        self.label.setText(title.upper())

    def set_summary(self, text: str) -> None:
        """One line of what is happening, for the banner and the strip."""
        self.summary = " ".join(str(text or "").split())
        self.detail.setText(self.summary[:60])

    def set_folded(self, folded: bool) -> None:
        if folded == self.folded:
            return
        self.folded = folded
        self.auto_folded = False         # a person's choice overrides the fit
        self.folded_changed.emit(folded)

    def set_enabled_feature(self, on: bool) -> None:
        self.enabled_ = bool(on)


class Strip(QToolButton):
    """A folded panel: its name and summary written sideways, one click to open."""

    WIDTH = 40

    def __init__(self, panel: FoldPanel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.setObjectName("FoldStrip")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"Unfold {panel.title.lower()}")
        self.setFixedWidth(self.WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.clicked.connect(lambda: panel.set_folded(False))

    def sizeHint(self) -> QSize:              # noqa: N802
        return QSize(self.WIDTH, 200)

    def paintEvent(self, event):              # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hover = self.underMouse()
        p.setPen(QColor(theme.LINE))
        p.setBrush(QColor(theme.PANEL_HI if hover else theme.PANEL))
        p.drawRoundedRect(rect, 9, 9)

        p.setPen(QColor(theme.TEXT_DIM))
        font = QFont(self.font())
        font.setPointSizeF(max(8.0, font.pointSizeF() - 0.5))
        p.setFont(font)
        p.drawText(QRectF(0, 8, self.width(), 18), Qt.AlignCenter, "◂")

        # Written downwards, top to bottom, as a spine reads.
        p.save()
        p.translate(self.width() / 2 + 5, 34)
        p.rotate(90)
        metrics = p.fontMetrics()
        room = max(40, self.height() - 50)
        name = self.panel.title.upper()
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(0, 0, name)
        summary = self.panel.summary
        if summary:
            offset = metrics.horizontalAdvance(name) + 14
            p.setPen(QColor(theme.AMBER))
            p.drawText(offset, 0, metrics.elidedText(
                summary, Qt.ElideRight, int(room - offset)))
        p.restore()
        p.end()


class Workspace(QWidget):
    """The chat and its panels, laid out by the three rules above."""

    layout_changed = Signal()

    def __init__(self, chat: QWidget, parent=None):
        super().__init__(parent)
        self.chat = chat
        self.panels: list[FoldPanel] = []
        self.strips: dict[str, Strip] = {}

        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(12, 12, 12, 12)
        self.row.setSpacing(12)
        # The chat never gets so narrow its composer buttons crush into
        # fragments: below this the side panels give way first.
        # Wide enough for the composer's whole button row — Following,
        # Continue, Speak, Dictate, Stop, Send — at their natural size. At
        # 460 they shrank and cut their own labels to "ollowin".
        chat.setMinimumWidth(560)
        self.row.addWidget(chat, 1)

        self.stack_host = QWidget()
        self.stack = QHBoxLayout(self.stack_host)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.stack.setSpacing(2)
        self.row.addWidget(self.stack_host, 1)

        self.strip_host = QWidget()
        self.strip_row = QHBoxLayout(self.strip_host)
        self.strip_row.setContentsMargins(0, 0, 0, 0)
        self.strip_row.setSpacing(8)
        self.row.addWidget(self.strip_host, 0)

    # -- membership -----------------------------------------------------------
    def add_panel(self, panel: FoldPanel) -> None:
        self.panels.append(panel)
        strip = Strip(panel)
        self.strips[panel.key] = strip
        panel.folded_changed.connect(lambda _f: self.relayout())
        self.relayout()

    def panel(self, key: str) -> FoldPanel | None:
        return next((p for p in self.panels if p.key == key), None)

    def set_feature(self, key: str, on: bool) -> None:
        """Switch a panel off entirely, or back on (Canvas and Plan)."""
        panel = self.panel(key)
        if panel is None:
            return
        panel.set_enabled_feature(on)
        if on:
            panel.folded = False       # switching on means wanting to see it
        self.relayout()

    # -- room ----------------------------------------------------------------
    def _needed(self, open_count: int, folded_count: int) -> int:
        """Width the workspace needs for this many open and folded panels."""
        margins = self.row.contentsMargins()
        need = margins.left() + margins.right() + self.chat.minimumWidth()
        if open_count:
            need += self.row.spacing() + open_count * FoldPanel.MIN_WIDTH \
                + (open_count - 1) * self.stack.spacing()
        if folded_count:
            need += self.row.spacing() + folded_count * Strip.WIDTH \
                + (folded_count - 1) * self.strip_row.spacing()
        return need

    def _fit(self) -> bool:
        """Fold the rightmost panel while there is not room, and unfold what
        was folded for lack of room once there is. Returns True on a change.

        Without this, a window narrower than everything's minimum — which
        Windows display scaling makes of an ordinary screen — squeezed the
        panels until they overlapped.
        """
        width = self.width()
        changed = False
        while True:
            opened = self.open_panels()
            folded = self.folded_panels()
            if len(opened) and self._needed(len(opened), len(folded)) > width:
                victim = opened[-1]
                victim.folded = True
                victim.auto_folded = True
                changed = True
                continue
            waiting = [p for p in folded if p.auto_folded]
            if waiting:
                candidate = waiting[0]
                if self._needed(len(opened) + 1, len(folded) - 1) <= width:
                    candidate.folded = False
                    candidate.auto_folded = False
                    changed = True
                    continue
            return changed

    def resizeEvent(self, event):          # noqa: N802
        super().resizeEvent(event)
        if self._fit():
            self._place()

    # -- the rules --------------------------------------------------------------
    def open_panels(self) -> list[FoldPanel]:
        return [p for p in self.panels if p.enabled_ and not p.folded]

    def folded_panels(self) -> list[FoldPanel]:
        return [p for p in self.panels if p.enabled_ and p.folded]

    def relayout(self) -> None:
        self._fit()
        self._place()

    def _place(self) -> None:
        # Take everything out, then put back what the rules say belongs.
        for p in self.panels:
            self.stack.removeWidget(p)
            p.setParent(self.stack_host)
            p.hide()
        for strip in self.strips.values():
            self.strip_row.removeWidget(strip)
            strip.setParent(self.strip_host)
            strip.hide()

        opened = self.open_panels()
        for p in opened:
            # Equal stretch: the open panels share the width evenly, each at
            # the full height of the workspace.
            self.stack.addWidget(p, 1)
            p.show()
        self.stack_host.setVisible(bool(opened))

        folded = self.folded_panels()
        for p in folded:
            strip = self.strips[p.key]
            self.strip_row.addWidget(strip)
            strip.show()
            strip.update()
        self.strip_host.setVisible(bool(folded))

        # With no open panel beside it the chat takes the whole row; with some
        # it shares, giving the panels a little more because code wants width.
        # Side by side, the panels need width more than the chat does: a
        # row of three buttons wants about 300px, and the chat reads fine at
        # its 460px floor. Two parts to the chat, three to the panels.
        self.row.setStretch(0, 2)
        self.row.setStretch(1, 3 if opened else 0)
        self.layout_changed.emit()
