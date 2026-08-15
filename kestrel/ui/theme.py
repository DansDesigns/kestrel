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
        "LINE": "#243140", "TEXT": "#C9D4DF", "TEXT_DIM": "#6E7E8E",
        "AMBER": "#E8A33D", "SIGNAL": "#57B894", "ALERT": "#D96A5A",
        "VIOLET": "#8C7BC7", "PLAN": "#4FA3C7", "THINK": "#6FB3C9",
        "FREE": "#1B2530", "BUBBLE_YOU": "#1E3A46", "BUBBLE_AI": "#1C2531",
        "BUBBLE_THINK": "#161F2A",
        "ON_ACCENT": "#101720", "ACCENT_HOVER": "#F2B457",
    },
    "light": {
        "INK": "#F4F1EA", "PANEL": "#FFFFFF", "PANEL_HI": "#EAE5DA",
        "LINE": "#D6CFC1", "TEXT": "#1E252E", "TEXT_DIM": "#6B7480",
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

current = "dark"
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


def apply(name: str = "dark", ui: str | None = None, mono: str | None = None,
          size: int = 0) -> str:
    """Switch palette and return the stylesheet for it."""
    global current
    if ui is not None or mono is not None or size:
        set_fonts(ui or ui_font_override, mono or mono_font_override, size)
    current = name if name in PALETTES else "dark"
    for key, value in PALETTES[current].items():
        globals()[key] = value
    return stylesheet()


def toggle() -> str:
    return apply("light" if current == "dark" else "dark")


def stylesheet() -> str:
    p = PALETTES[current]
    ink, panel, panel_hi = p["INK"], p["PANEL"], p["PANEL_HI"]
    line, text, dim = p["LINE"], p["TEXT"], p["TEXT_DIM"]
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
QSpinBox, QDoubleSpinBox {{ min-width: 74px; }}
QComboBox {{ min-width: 96px; }}
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
}}
/* Inside an already-framed panel, a second frame wastes a margin on each side
   and reads as a box within a box. */
QTreeWidget#Flush, QTextEdit#Flush, QListWidget#Flush {{
    border: none; border-radius: 0; background: {ink};
}}
QTreeWidget::item, QListWidget::item {{ padding: 5px 4px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: {line}; }}
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
QSplitter::handle {{ background: {line}; }}
QStatusBar {{ background: {panel}; border-top: 1px solid {line}; }}
QProgressBar {{ background: {panel}; border: 1px solid {line}; border-radius: 4px; }}
QProgressBar::chunk {{ background: {amber}; border-radius: 3px; }}
QToolTip {{ background: {panel_hi}; color: {text}; border: 1px solid {line}; padding: 4px; }}
"""


apply("dark")
STYLESHEET = stylesheet()   # kept for older call sites
