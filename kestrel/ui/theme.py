"""Visual language.

An instrument panel rather than a chat window. The element being watched is
whether the model is about to run out of room, so each palette reserves a single
accent colour for telemetry and keeps everything else quiet.

Two palettes are provided. Dark is a cool slate the readouts sit on top of;
light is a paper-and-bronze daylight face rather than an inverted dark theme —
inverting a dark palette produces glare and washes the accent out. Both are
built from the same role names, so the rest of the interface never asks which
one is active.
"""
from __future__ import annotations

PALETTES = {
    "dark": {
        "INK": "#101720", "PANEL": "#18212C", "PANEL_HI": "#1F2A37",
        "LINE": "#243140", "TEXT": "#C9D4DF", "TEXT_DIM": "#93A3B4",
        "SELECT": "#2C4257", "ON_SELECT": "#EAF1F7",
        "AMBER": "#E8A33D", "SIGNAL": "#57B894", "ALERT": "#D96A5A",
        "VIOLET": "#8C7BC7", "PLAN": "#4FA3C7", "THINK": "#6FB3C9",
        "FREE": "#1B2530", "BUBBLE_YOU": "#1E3A46", "BUBBLE_AI": "#1C2531",
        "BUBBLE_THINK": "#161F2A",
        "ON_ACCENT": "#101720", "ACCENT_HOVER": "#F2B457",
    },
    "light": {
        "INK": "#F4F1EA", "PANEL": "#FFFFFF", "PANEL_HI": "#EAE5DA",
        "LINE": "#D6CFC1", "TEXT": "#1E252E", "TEXT_DIM": "#5A6470",
        "SELECT": "#CFE0EC", "ON_SELECT": "#12202C",
        "AMBER": "#A0670F", "SIGNAL": "#2E7D5B", "ALERT": "#B23B28",
        "VIOLET": "#584AA0", "PLAN": "#1F6F92", "THINK": "#12667C",
        "FREE": "#E4DED2", "BUBBLE_YOU": "#DCEBF2", "BUBBLE_AI": "#FFFFFF",
        "BUBBLE_THINK": "#EFEAE0",
        "ON_ACCENT": "#FFFFFF", "ACCENT_HOVER": "#8A580C",
    },
}

UI_FONTS = ["Inter", "SF Pro Text", "Segoe UI", "Ubuntu", "Cantarell", "Noto Sans", "sans-serif"]
MONO_FONTS = ["JetBrains Mono", "SF Mono", "Cascadia Mono", "Consolas",
              "DejaVu Sans Mono", "Menlo", "monospace"]

# Surfaces are tinted rather than replaced: one neutral palette per mode, mixed
# a little way towards a hue. Hand-writing a dozen full palettes would drift out
# of step the moment a role changed, and most of them would be wrong somewhere.
TINTS = {
    "slate":    ("Slate (neutral)", ""),
    "blue":     ("Blue",            "#3A6EA5"),
    "teal":     ("Teal",            "#2E8B84"),
    "green":    ("Green",           "#3F8B4F"),
    "orange":   ("Orange",          "#C2703A"),
    "red":      ("Red",             "#B04A45"),
    "violet":   ("Violet",          "#7A5AA8"),
    "graphite": ("Graphite",        "#7A7A7A"),
}

# What buttons, the gauge and every other accent are drawn in.
ACCENTS = {
    "amber":  ("Amber",  "#E8A33D", "#A0670F"),      # name, dark, light
    "blue":   ("Blue",   "#5A9BD8", "#1F5FA0"),
    "teal":   ("Teal",   "#4FB3A8", "#116F66"),
    "green":  ("Green",  "#6BBF73", "#2E7D3A"),
    "orange": ("Orange", "#EE8F52", "#B4551C"),
    "red":    ("Red",    "#E4726A", "#A63A33"),
    "violet": ("Violet", "#A98BE0", "#5B3E9E"),
    "pink":   ("Pink",   "#E087B4", "#A33C74"),
}

TINT_STRENGTH = {"dark": 0.16, "light": 0.10}

current = "dark"
tint = "slate"
accent = "amber"
# Overrides chosen by the user. Missing glyphs render as boxes, and which font
# has which glyph varies enough by platform that this has to be selectable
# rather than guessed.
ui_font_override = ""
mono_font_override = ""
base_size = 13

# Role names are module globals so existing call sites keep working; apply()
# rebinds them, and anything that reads them at paint time picks up the change.
INK = PANEL = PANEL_HI = LINE = TEXT = TEXT_DIM = ""
AMBER = SIGNAL = ALERT = VIOLET = PLAN = THINK = FREE = ""
SELECT = ON_SELECT = ""
BUBBLE_YOU = BUBBLE_AI = BUBBLE_THINK = ""
ON_ACCENT = ACCENT_HOVER = ""


def font_stack(fonts) -> str:
    return ", ".join(f'"{f}"' if " " in f else f for f in fonts)


def ui_fonts() -> list[str]:
    return ([ui_font_override] if ui_font_override else []) + UI_FONTS


def mono_fonts() -> list[str]:
    return ([mono_font_override] if mono_font_override else []) + MONO_FONTS


def set_fonts(ui: str = "", mono: str = "", size: int = 0) -> None:
    global ui_font_override, mono_font_override, base_size
    ui_font_override = ui or ""
    mono_font_override = mono or ""
    if size:
        base_size = max(9, min(20, int(size)))


def _rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, int(round(c)))):02X}" for c in rgb)


def mix(base: str, other: str, amount: float) -> str:
    """Move `base` a fraction of the way towards `other`."""
    a, b = _rgb(base), _rgb(other)
    return _hex(a[i] + (b[i] - a[i]) * amount for i in range(3))


def luminance(value: str) -> float:
    channels = []
    for c in _rgb(value):
        c = c / 255
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def readable_on(background: str) -> str:
    """Black or white, whichever can actually be read on this colour."""
    dark, light = "#12181F", "#FFFFFF"
    return dark if contrast(dark, background) >= contrast(light, background) else light


def legible_accent(colour: str, minimum: float = 4.5) -> tuple[str, str]:
    """Adjust an accent until text on it can be read. Returns (fill, text).

    A mid-tone fill is the awkward case: neither black nor white sits far
    enough from it, and a button label at 2.3:1 is decoration rather than
    writing. The hue is kept and the lightness moved until the better of the
    two passes.
    """
    fill = colour
    for _ in range(24):
        text = readable_on(fill)
        if contrast(text, fill) >= minimum:
            return fill, text
        # Push away from whichever end is closer, keeping the hue.
        fill = mix(fill, "#000000" if text == "#FFFFFF" else "#FFFFFF", 0.07)
    return fill, readable_on(fill)


# The roles that carry the surface hue. Text and the semantic colours are left
# alone: tinting those is how an interface becomes hard to read.
TINTED_ROLES = ("INK", "PANEL", "PANEL_HI", "LINE", "FREE", "BUBBLE_YOU",
                "BUBBLE_AI", "BUBBLE_THINK", "SELECT")


def build_palette(mode: str, tint_name: str, accent_name: str) -> dict:
    palette = dict(PALETTES[mode if mode in PALETTES else "dark"])
    hue = TINTS.get(tint_name, TINTS["slate"])[1]
    if hue:
        strength = TINT_STRENGTH.get(mode, 0.14)
        for role in TINTED_ROLES:
            if role in palette:
                # The selection keeps more of the hue: it is meant to stand out.
                amount = strength * (1.8 if role == "SELECT" else 1.0)
                palette[role] = mix(palette[role], hue, min(0.6, amount))
    chosen = ACCENTS.get(accent_name)
    if chosen:
        raw = chosen[1] if mode == "dark" else chosen[2]
        colour, on_colour = legible_accent(raw)
        palette["AMBER"] = colour
        palette["ACCENT_HOVER"] = mix(colour, "#FFFFFF" if mode == "dark"
                                      else "#000000", 0.18)
        palette["ON_ACCENT"] = on_colour
    return palette


def apply(name: str = "dark", ui: str | None = None, mono: str | None = None,
          size: int = 0, tint_name: str | None = None,
          accent_name: str | None = None) -> str:
    """Switch palette and return the stylesheet for it."""
    global current, tint, accent
    if ui is not None or mono is not None or size:
        set_fonts(ui or ui_font_override, mono or mono_font_override, size)
    current = name if name in PALETTES else "dark"
    if tint_name is not None:
        tint = tint_name if tint_name in TINTS else "slate"
    if accent_name is not None:
        accent = accent_name if accent_name in ACCENTS else "amber"
    for key, value in build_palette(current, tint, accent).items():
        globals()[key] = value
    return stylesheet()


def toggle() -> str:
    return apply("light" if current == "dark" else "dark",
                 tint_name=tint, accent_name=accent)


def stylesheet() -> str:
    p = build_palette(current, tint, accent)
    ink, panel, panel_hi = p["INK"], p["PANEL"], p["PANEL_HI"]
    line, text, dim = p["LINE"], p["TEXT"], p["TEXT_DIM"]
    select, on_select = p["SELECT"], p["ON_SELECT"]
    amber, on_accent, hover = p["AMBER"], p["ON_ACCENT"], p["ACCENT_HOVER"]
    alert = p["ALERT"]
    return f"""
QWidget {{
    background: {ink};
    color: {text};
    font-family: {font_stack(ui_fonts())};
    font-size: {base_size}px;
}}
QMainWindow::separator {{ background: {line}; width: 1px; height: 1px; }}

QFrame#Panel, QWidget#Panel {{ background: {panel}; border: 1px solid {line}; border-radius: 6px; }}
QWidget#TopBar {{ background: {panel}; border-bottom: 1px solid {line}; }}
QLabel#Wordmark {{ font-size: 14px; font-weight: 600; letter-spacing: 0.5px; }}

QScrollArea {{ border: none; }}
QLabel#Eyebrow {{
    margin-top: 4px;
    color: {dim};
    font-family: {font_stack(mono_fonts())};
    font-size: {max(9, base_size - 3)}px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}}
QLabel#Section {{
    color: {text};
    font-size: {max(12, base_size)}px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 2px;
}}
QLabel#Readout {{ font-family: {font_stack(mono_fonts())}; color: {amber};
                  font-size: {max(10, base_size - 1)}px; }}
QLabel#Dim {{ color: {dim}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {panel};
    border: 1px solid {line};
    border-radius: 5px;
    padding: 6px 9px;
    min-height: 20px;
    selection-background-color: {amber};
    selection-color: {on_accent};
}}
/* Spin boxes and drop-downs lose their contents to the control furniture
   unless a floor is set: the arrows take a fixed width whatever is left. */
/* Wide enough to read a value, narrow enough that three fit a side panel.
   These floors are what decide whether a row of controls overflows. */
QSpinBox, QDoubleSpinBox {{ min-width: 52px; }}
QComboBox {{ min-width: 72px; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 15px; }}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {amber}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {panel_hi}; border: 1px solid {line}; selection-background-color: {line};
}}

QPushButton {{
    background: {panel_hi};
    border: 1px solid {line};
    border-radius: 5px;
    padding: 7px 14px;
    min-height: 20px;
}}
QPushButton:hover {{ border-color: {dim}; }}
QPushButton:pressed {{ background: {line}; }}
QPushButton:disabled {{ color: {dim}; border-color: {panel}; }}
QPushButton#Primary {{ background: {amber}; color: {on_accent}; border: none; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {hover}; }}
QPushButton#Primary:disabled {{ background: {line}; color: {dim}; }}
QPushButton#Danger {{ border-color: {alert}; color: {alert}; }}
QPushButton#Chip {{ padding: 4px 10px; border-radius: 12px; font-size: 12px; }}
QPushButton#Chip:checked {{ background: {amber}; color: {on_accent}; border-color: {amber}; }}

QTabWidget::pane {{ border: 1px solid {line}; border-radius: 6px; top: -1px; }}
QTabWidget[collapsed="true"]::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent;
    color: {dim};
    padding: 7px 13px;
    border-bottom: 2px solid transparent;
    font-size: 12px;
}}
QTabBar::tab:selected {{ color: {text}; border-bottom-color: {amber}; }}

QTreeWidget, QListWidget, QTableWidget {{
    background: {panel}; border: 1px solid {line}; border-radius: 6px;
    alternate-background-color: {panel_hi};
    /* Both halves of a selection are set: inheriting the text colour from the
       unselected state is how a highlighted row ends up unreadable. */
    selection-background-color: {select};
    selection-color: {on_select};
}}
/* Inside an already-framed panel, a second frame wastes a margin on each side
   and reads as a box within a box. */
QTreeWidget#Flush, QTextEdit#Flush, QListWidget#Flush {{
    border: none; border-radius: 0; background: {ink};
}}
QTreeWidget::item, QListWidget::item {{ padding: 5px 4px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background: {select}; color: {on_select};
}}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {panel_hi}; }}
QHeaderView::section {{
    background: {panel_hi}; color: {dim}; border: none;
    border-bottom: 1px solid {line}; padding: 5px;
    font-size: 11px; letter-spacing: 0.6px;
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {line}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {dim}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {line}; border-radius: 5px; min-width: 30px; }}

QMenuBar {{ background: {ink}; }}
QMenuBar::item:selected {{ background: {panel_hi}; }}
QMenu {{ background: {panel_hi}; border: 1px solid {line}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {line}; }}

QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {line}; border-radius: 3px; background: {panel};
}}
QCheckBox::indicator:checked {{ background: {amber}; border-color: {amber}; }}
QGroupBox {{ border: 1px solid {line}; border-radius: 6px; margin-top: 14px; padding-top: 8px; }}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 5px;
    color: {dim}; font-size: 11px; letter-spacing: 1px;
}}
QSlider::groove:horizontal {{
    height: 4px; background: {panel_hi}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {amber}; border: none; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {hover}; }}
QSplitter::handle {{ background: {line}; }}
QSplitter::handle:hover {{ background: {dim}; }}
QStatusBar {{ background: {panel}; border-top: 1px solid {line}; }}
QProgressBar {{ background: {panel}; border: 1px solid {line}; border-radius: 4px; }}
QProgressBar::chunk {{ background: {amber}; border-radius: 3px; }}
QToolTip {{ background: {panel_hi}; color: {text}; border: 1px solid {line}; padding: 4px; }}
"""


apply("dark")
STYLESHEET = stylesheet()   # kept for older call sites
