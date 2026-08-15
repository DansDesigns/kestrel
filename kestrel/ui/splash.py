"""Start-up splash.

Opening Kestrel means scanning model directories, reading skill folders, probing
speech engines and looking for llama.cpp. Some of that shells out to other
programs. Doing it behind a blank unresponsive window looks like a hang, so the
work is narrated instead.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from . import theme
from .widgets import mono_font


class Splash(QWidget):
    WIDTH = 420
    HEIGHT = 190

    def __init__(self):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint
                         | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self._status = "starting"
        self._phase = 0.0
        self._steps: list[str] = []
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._centre()

    def _centre(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.move(area.center().x() - self.WIDTH // 2,
                      area.center().y() - self.HEIGHT // 2)

    def begin(self) -> None:
        self.show()
        self._timer.start()
        QApplication.processEvents()

    def message(self, text: str) -> None:
        """Report a stage. Events are pumped so the animation keeps moving even
        though the work is happening on this thread."""
        self._status = text
        self._steps.append(text)
        self.update()
        QApplication.processEvents()

    def finish(self, window=None) -> None:
        self._timer.stop()
        self.close()
        if window is not None:
            window.activateWindow()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.02) % 1.0
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        card = QRectF(0, 0, self.WIDTH, self.HEIGHT)
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.setBrush(QColor(theme.PANEL))
        p.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)

        f = QFont()
        f.setPointSize(17)
        f.setWeight(QFont.DemiBold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 2.5)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(0, 30, self.WIDTH, 30), Qt.AlignHCenter, "KESTREL")

        p.setFont(mono_font(9))
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(0, 58, self.WIDTH, 18), Qt.AlignHCenter,
                   "agentic harness for llama.cpp")

        # An arc sweeping a track: it reads as indeterminate progress, which is
        # honest here — the total work is not known in advance.
        track = QRectF(self.WIDTH / 2 - 17, 92, 34, 34)
        p.setBrush(Qt.NoBrush)
        pen = QPen(QColor(theme.LINE))
        pen.setWidth(3)
        p.setPen(pen)
        p.drawEllipse(track)
        pen.setColor(QColor(theme.AMBER))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        span = 90 + 40 * math.sin(self._phase * 2 * math.pi)
        p.drawArc(track, int(-self._phase * 360 * 16), int(-span * 16))

        p.setFont(mono_font(9))
        p.setPen(QColor(theme.AMBER))
        p.drawText(QRectF(12, 140, self.WIDTH - 24, 18), Qt.AlignHCenter,
                   self._status)
        # A count without a credible total is a false promise; the dots simply
        # show that stages are completing.
        done = len(self._steps)
        p.setPen(Qt.NoPen)
        width = min(done, 14) * 8
        for i in range(min(done, 14)):
            p.setBrush(QColor(theme.AMBER if i == done - 1 else theme.LINE))
            p.drawEllipse(QRectF(self.WIDTH / 2 - width / 2 + i * 8, 163, 4, 4))
        p.end()
