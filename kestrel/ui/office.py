"""The office: Kestrel as a person, at work in a small room.

Everything the character does is something the agent is actually doing. A tool
call walks it to the place that tool belongs — the desk for the canvas, the
whiteboard for the plan, the bookshelf for skills, the filing cabinet for
memory — so the room is a view of the real activity, readable at a glance
from across a room, not a separate simulation running alongside.

Drawn in software, as flat-shaded low-poly 3D, on purpose. The GPU is running
the model: a real 3D engine would take compute and shared memory straight out
of inference on integrated graphics, and contend with llama.cpp's queue on a
discrete card. A few dozen shaded polygons drawn by the CPU at a modest frame
rate cost next to nothing, and the view stops drawing entirely when hidden.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from . import theme


# ---------------------------------------------------------------- the look --
@dataclass
class Look:
    """How a persona appears. Read from a `look:` line in the persona file,
    or derived from its name so that every persona looks like someone."""
    skin: str = "#E0B48C"
    hair: str = "#3A2A20"
    shirt: str = "#3A7CA5"
    trousers: str = "#2E3440"
    shoes: str = "#1E1E1E"
    hat: bool = False

    @classmethod
    def for_persona(cls, name: str = "", text: str = "") -> "Look":
        look = cls.from_name(name or "Kestrel")
        match = re.search(r"^look:\s*(.+)$", text or "", re.M | re.I)
        if match:
            for part in match.group(1).split(","):
                words = part.strip().split()
                if not words:
                    continue
                key = words[0].lower()
                if key == "hat":
                    look.hat = True
                elif len(words) > 1 and hasattr(look, key):
                    setattr(look, key, words[1])
        return look

    @classmethod
    def from_name(cls, name: str) -> "Look":
        """A stable look from a name: the same persona always looks the same."""
        seed = hashlib.sha1(name.encode("utf-8")).digest()
        skins = ["#F1C9A5", "#E0B48C", "#C68E65", "#9C6B47", "#6E4A33"]
        hairs = ["#2B2016", "#5A3A22", "#8C5A2B", "#C9A66B", "#3B3B3B", "#A33A2A"]
        shirts = ["#3A7CA5", "#2E7D5B", "#A2543A", "#6B4FA0", "#C08A2E",
                  "#2F6F73", "#8E3B5C"]
        return cls(skin=skins[seed[0] % len(skins)],
                   hair=hairs[seed[1] % len(hairs)],
                   shirt=shirts[seed[2] % len(shirts)],
                   trousers=["#2E3440", "#3B4A5A", "#4A3B2E"][seed[3] % 3],
                   hat=seed[4] % 5 == 0)


# ----------------------------------------------------------------- geometry --
@dataclass
class Box:
    x: float
    y: float
    z: float
    w: float
    h: float
    d: float
    colour: str
    glow: bool = False            # drawn unshaded, for screens and lights


# Where each kind of work happens, and how to stand there: (x, z, facing, pose).
STATIONS = {
    "desk":       (0.5, -2.05, math.pi, "sit"),
    "whiteboard": (-3.9, 0.2, -math.pi / 2, "stand"),
    "bookshelf":  (-3.3, -2.7, math.pi, "reach"),
    "cabinet":    (2.8, -2.7, math.pi, "reach"),
    "window":     (-3.9, 2.6, -math.pi / 2, "stand"),
    "rack":       (3.3, -2.2, math.pi / 2, "stand"),
    "coffee":     (3.6, 1.6, math.pi / 2, "stand"),
    "armchair":   (2.0, 2.4, math.pi * 0.85, "sit"),
}

# What each tool means for where the character goes.
TOOL_PLACES = (
    (("canvas", "write_file", "edit_file", "patch", "apply"), "desk"),
    (("plan", "todo"), "whiteboard"),
    (("skill",), "bookshelf"),
    (("memory", "remember", "recall", "forget"), "cabinet"),
    (("web", "fetch", "search", "http", "browse"), "window"),
    (("read_file", "list", "grep", "glob", "tree", "find"), "desk"),
)


def place_for_tool(name: str) -> str:
    lowered = (name or "").lower()
    for words, place in TOOL_PLACES:
        if any(w in lowered for w in words):
            return place
    return "desk"


def _shade(colour: str, amount: float) -> QColor:
    c = QColor(colour)
    return QColor(max(0, min(255, int(c.red() * amount))),
                  max(0, min(255, int(c.green() * amount))),
                  max(0, min(255, int(c.blue() * amount))))


# ------------------------------------------------------------------- person --
@dataclass
class Person:
    x: float = 3.6
    z: float = 1.6
    facing: float = math.pi / 2
    pose: str = "stand"
    target: str = "coffee"
    walk_phase: float = 0.0
    walking: bool = False
    look: Look = field(default_factory=Look)

    def go(self, place: str) -> None:
        if place in STATIONS:
            self.target = place

    def step(self, dt: float) -> None:
        tx, tz, facing, pose = STATIONS[self.target]
        dx, dz = tx - self.x, tz - self.z
        distance = math.hypot(dx, dz)
        if distance > 0.05:
            speed = 2.4 * dt
            move = min(speed, distance)
            self.x += dx / distance * move
            self.z += dz / distance * move
            self.facing = math.atan2(dx, dz)
            self.walking = True
            self.pose = "stand"
            self.walk_phase += dt * 9.0
        else:
            self.walking = False
            self.facing += (facing - self.facing) * min(1.0, dt * 8)
            self.pose = pose

    def boxes(self, now: float, busy: bool) -> list[Box]:
        """The person as boxes, posed for what it is doing."""
        L = self.look
        swing = math.sin(self.walk_phase) * 0.35 if self.walking else 0.0
        sit = self.pose == "sit"
        base = 0.42 if sit else 0.0
        parts: list[tuple] = []           # local (x, y, z, w, h, d, colour)

        if sit:
            parts += [(-0.17, 0.42, -0.05, 0.14, 0.12, 0.42, L.trousers),
                      (0.03, 0.42, -0.05, 0.14, 0.12, 0.42, L.trousers),
                      (-0.17, 0.0, 0.3, 0.14, 0.44, 0.12, L.trousers),
                      (0.03, 0.0, 0.3, 0.14, 0.44, 0.12, L.trousers)]
        else:
            parts += [(-0.17, 0.0, -0.06 + swing * 0.3, 0.14, 0.78, 0.14, L.trousers),
                      (0.03, 0.0, -0.06 - swing * 0.3, 0.14, 0.78, 0.14, L.trousers)]
        torso_y = 0.54 if sit else 0.78
        parts.append((-0.22, torso_y, -0.1, 0.44, 0.62, 0.22, L.shirt))

        typing = sit and busy and self.target == "desk"
        reach = self.pose == "reach"
        for side, sx in ((-1, -0.34), (1, 0.22)):
            if typing:
                bob = math.sin(now * 14 + side) * 0.03
                parts.append((sx, torso_y + 0.34 + bob, 0.08, 0.12, 0.12, 0.34, L.shirt))
            elif reach and side == 1:
                parts.append((sx, torso_y + 0.55, -0.02, 0.12, 0.5, 0.12, L.shirt))
            else:
                a = -swing * side * 0.3
                parts.append((sx, torso_y + 0.02, -0.06 + a, 0.12, 0.56, 0.12, L.shirt))

        head_y = torso_y + 0.64
        parts.append((-0.16, head_y, -0.14, 0.32, 0.32, 0.3, L.skin))
        parts.append((-0.17, head_y + 0.24, -0.16, 0.34, 0.12, 0.33, L.hair))
        parts.append((-0.17, head_y + 0.08, -0.17, 0.34, 0.18, 0.08, L.hair))
        if L.hat:
            parts.append((-0.22, head_y + 0.34, -0.2, 0.44, 0.05, 0.4, L.shirt))
            parts.append((-0.15, head_y + 0.38, -0.14, 0.3, 0.14, 0.28, L.shirt))

        # Rotate each part about the person's own vertical axis.
        c, s = math.cos(self.facing), math.sin(self.facing)
        placed = []
        for (x, y, z, w, h, d, colour) in parts:
            cx, cz = x + w / 2, z + d / 2
            rx = cx * c + cz * s
            rz = -cx * s + cz * c
            # Boxes stay axis-aligned; turning is carried by their positions,
            # which at this scale reads as turning without per-box rotation.
            if abs(math.sin(self.facing)) > 0.7:
                w, d = d, w
            placed.append(Box(self.x + rx - w / 2, base * 0 + y, self.z + rz - d / 2,
                              w, h, d, colour))
        return placed


# ------------------------------------------------------------------- office --
WALL, FLOOR = "#D8CFC0", "#8F7A62"


def room_shell() -> list[Box]:
    """Floor and walls. Drawn first and never depth-sorted: they are behind
    everything in the room, and a face this large has an average depth in the
    middle of the room, which sorted it over whatever stood behind that line."""
    return [
        Box(-5, -0.1, -4, 10, 0.1, 8, FLOOR),
        Box(-5, 0, -4.2, 10, 2.6, 0.2, WALL),
        Box(-5.2, 0, -4, 0.2, 2.6, 8, WALL),
    ]


def office_boxes(now: float, server_ready: bool) -> list[Box]:
    wall = WALL
    wood, wood_dark = "#9B6B43", "#6E4A2E"
    boxes = [
        # desk, monitor, chair
        Box(-0.6, 0.72, -3.6, 2.2, 0.08, 1.0, wood),
        Box(-0.55, 0, -3.55, 0.08, 0.72, 0.9, wood_dark),
        Box(1.47, 0, -3.55, 0.08, 0.72, 0.9, wood_dark),
        Box(0.05, 0.8, -3.45, 0.9, 0.55, 0.06, "#20262E"),
        Box(0.1, 0.85, -3.4, 0.8, 0.45, 0.02, "#6FB7D9", glow=True),
        Box(0.4, 0.8, -3.5, 0.2, 0.08, 0.2, "#20262E"),
        Box(0.15, 0.42, -2.3, 0.7, 0.08, 0.6, "#2F3A46"),
        Box(0.15, 0.5, -1.72, 0.7, 0.7, 0.08, "#2F3A46"),
        # whiteboard
        Box(-5.0, 1.0, -1.2, 0.06, 1.3, 2.6, "#F4F4F0", glow=True),
        Box(-4.98, 1.0, -1.25, 0.08, 0.06, 2.7, "#8A8F96"),
        # bookshelf with books
        Box(-4.1, 0, -4.0, 1.6, 2.4, 0.5, wood_dark),
        # filing cabinet
        Box(2.4, 0, -4.0, 0.8, 1.3, 0.6, "#7B8591"),
        Box(2.45, 0.35, -3.39, 0.7, 0.04, 0.02, "#5A626C"),
        Box(2.45, 0.85, -3.39, 0.7, 0.04, 0.02, "#5A626C"),
        # server rack
        Box(4.0, 0, -4.0, 0.8, 2.2, 0.9, "#1F242B"),
        # window with the day outside
        Box(-5.05, 1.1, 1.8, 0.06, 1.4, 1.8, "#9DD3EC", glow=True),
        Box(-5.03, 1.78, 1.8, 0.08, 0.05, 1.8, wall),
        # counter and coffee machine
        Box(4.3, 0, 1.0, 0.7, 0.9, 1.4, wood),
        Box(4.45, 0.9, 1.35, 0.4, 0.5, 0.35, "#2B2B2B"),
        # armchair and plant
        Box(1.6, 0, 2.6, 0.9, 0.42, 0.8, "#7A3E3E"),
        Box(1.6, 0.42, 3.25, 0.9, 0.6, 0.15, "#7A3E3E"),
        Box(-3.0, 0, 3.0, 0.5, 0.45, 0.5, "#A0522D"),
        Box(-3.1, 0.45, 2.9, 0.7, 0.7, 0.7, "#3E7D4A"),
    ]
    for i, colour in enumerate(("#A33A2A", "#3A6EA5", "#C08A2E", "#2E7D5B",
                                "#6B4FA0", "#8C5A2B")):
        for shelf in (0.25, 1.05, 1.8):
            boxes.append(Box(-4.0 + i * 0.24, shelf, -3.62, 0.2, 0.5, 0.1, colour))
    # the rack's lights blink while the server is working
    for row in range(6):
        on = server_ready and int(now * 3 + row) % 4 != 0
        boxes.append(Box(4.1, 0.3 + row * 0.3, -3.09, 0.08, 0.06, 0.02,
                         "#57D98A" if on else "#3A3F46", glow=on))
    return boxes


# --------------------------------------------------------------------- view --
class OfficeView(QWidget):
    """The office, drawn and animated. Drag to turn the room."""

    placeClicked = Signal(str)

    FPS = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.person = Person()
        # From the front-right corner, so the two walls — and everything
        # hung on them — face the camera.
        self.yaw = 0.72
        self.pitch = 0.62
        self.busy = False
        self.thinking = False
        self.server_ready = False
        self.speech = ""
        self._speech_until = 0.0
        self._last = time.monotonic()
        self._idle_since = time.monotonic()
        self._drag = None
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / self.FPS))
        self.timer.timeout.connect(self._tick)

    # -- what the agent is doing ---------------------------------------------
    def set_look(self, look: Look) -> None:
        self.person.look = look
        self.update()

    def on_tool(self, name: str) -> None:
        self.busy = True
        self.thinking = False
        self.person.go(place_for_tool(name))
        self._idle_since = time.monotonic()

    def on_thinking(self) -> None:
        self.busy = True
        self.thinking = True
        if self.person.target not in ("desk", "whiteboard"):
            self.person.go("armchair")
        self._idle_since = time.monotonic()

    def on_replying(self) -> None:
        self.busy = True
        self.thinking = False
        self.person.go("desk")
        self._idle_since = time.monotonic()

    def on_done(self, text: str = "") -> None:
        self.busy = False
        self.thinking = False
        words = " ".join((text or "").split())
        if words:
            self.speech = words[:90] + ("…" if len(words) > 90 else "")
            self._speech_until = time.monotonic() + 6
        self._idle_since = time.monotonic()

    def set_server_ready(self, ready: bool) -> None:
        self.server_ready = ready
        if not ready and not self.busy:
            self.person.go("rack")

    # -- animation ------------------------------------------------------------
    def showEvent(self, event):                    # noqa: N802
        super().showEvent(event)
        self._last = time.monotonic()
        self.timer.start()

    def hideEvent(self, event):                    # noqa: N802
        # Not drawing what nobody can see is the whole of the cost control.
        self.timer.stop()
        super().hideEvent(event)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.2, now - self._last)
        self._last = now
        idle = now - self._idle_since
        if not self.busy and idle > 20 and self.person.target != "coffee":
            self.person.go("coffee")
        if not self.busy and idle > 45 and self.person.target == "coffee":
            self.person.go("window")
        self.person.step(dt)
        self.update()

    # -- drawing --------------------------------------------------------------
    def _project(self, x, y, z):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        rx = x * cy - z * sy
        rz = x * sy + z * cy
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        ry = y * cp - rz * sp
        depth = y * sp + rz * cp
        scale = min(self.width(), self.height()) / 9.4
        perspective = 1.0 / (1.0 + depth * 0.035)
        return (self.width() / 2 + rx * scale * perspective,
                self.height() * 0.6 - ry * scale * perspective,
                depth)

    def _faces(self, box: Box):
        x0, y0, z0 = box.x, box.y, box.z
        x1, y1, z1 = x0 + box.w, y0 + box.h, z0 + box.d
        return (
            # Wound the same way round as the sides when seen from outside,
            # or the back-face test throws away every floor and desktop.
            ((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), 1.0),   # top
            ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), 0.82),  # front
            ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), 0.6),   # back
            ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), 0.7),   # right
            ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), 0.66),  # left
        )

    def paintEvent(self, event):                   # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        now = time.monotonic()

        edge = QPen(QColor(0, 0, 0, 40))
        edge.setWidthF(0.6)
        p.setPen(edge)
        for _depth, pts, colour in self._polygons(room_shell(), sort=True):
            p.setBrush(colour)
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y, _ in pts]))

        things = office_boxes(now, self.server_ready) + \
            self.person.boxes(now, self.busy)
        for _depth, pts, colour in self._polygons(things, sort=True):
            p.setBrush(colour)
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y, _ in pts]))

        self._draw_bubbles(p, now)
        p.end()

    def _polygons(self, boxes, sort: bool):
        polys = []
        for box in boxes:
            for *corners, light in self._faces(box):
                pts = [self._project(*c) for c in corners]
                # Back faces are skipped by their winding on screen.
                area = sum(pts[i][0] * pts[(i + 1) % 4][1] -
                           pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))
                if area >= 0:
                    continue
                depth = sum(pt[2] for pt in pts) / 4
                colour = QColor(box.colour) if box.glow else _shade(box.colour, light)
                polys.append((depth, pts, colour))
        if sort:
            # Painter's algorithm: farthest first.
            polys.sort(key=lambda item: -item[0])
        return polys

    def _draw_bubbles(self, p: QPainter, now: float) -> None:
        hx, hy, _ = self._project(self.person.x, 2.05, self.person.z)
        if self.thinking:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 220))
            bob = math.sin(now * 3) * 2
            for i, r in enumerate((3, 5, 12)):
                p.drawEllipse(QPointF(hx + 6 + i * 8, hy - 10 - i * 12 + bob), r * 1.4, r)
            p.setPen(QColor("#555555"))
            p.setFont(QFont(self.font().family(), 9))
            p.drawText(QRectF(hx + 10, hy - 52 + bob, 30, 20), Qt.AlignCenter, "…")
        elif self.speech and now < self._speech_until:
            p.setFont(QFont(self.font().family(), 9))
            metrics = p.fontMetrics()
            width = min(260, metrics.horizontalAdvance(self.speech) + 20)
            rect = metrics.boundingRect(QRectF(0, 0, width - 20, 200).toRect(),
                                        Qt.TextWordWrap, self.speech)
            box = QRectF(hx - width / 2, hy - rect.height() - 34, width,
                         rect.height() + 16)
            p.setPen(QPen(QColor(0, 0, 0, 60)))
            p.setBrush(QColor(255, 255, 255, 235))
            p.drawRoundedRect(box, 9, 9)
            p.drawPolygon(QPolygonF([QPointF(hx - 6, box.bottom()),
                                     QPointF(hx + 6, box.bottom()),
                                     QPointF(hx, box.bottom() + 9)]))
            p.setPen(QColor("#222222"))
            p.drawText(box.adjusted(10, 8, -10, -8), Qt.TextWordWrap, self.speech)

    # -- turning the room -------------------------------------------------------
    def mousePressEvent(self, event):             # noqa: N802
        self._drag = (event.position().x(), self.yaw)

    def mouseMoveEvent(self, event):              # noqa: N802
        if self._drag:
            start, yaw = self._drag
            self.yaw = yaw + (event.position().x() - start) * 0.008
            self.update()

    def mouseReleaseEvent(self, event):           # noqa: N802
        self._drag = None
