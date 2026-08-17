"""The canvas buffer.

One piece of text that both the person and the model can edit. The editor is a
view onto it; the tools below are the model's way in. Keeping it here rather
than inside the widget means the agent — which runs on its own thread and knows
nothing about Qt — can read and write it without reaching into the interface.

Code is the case this exists for. A model asked to write a function into a chat
reply has to reproduce the whole thing every time it changes a line, which is
slow, expensive in context, and error-prone. Writing to a buffer instead lets it
change the part it means to change.
"""
from __future__ import annotations

import threading
from typing import Callable


class CanvasBuffer:
    """Thread-safe text with change notification."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._text = ""
        self._language = "python"
        self._name = "scratch.py"
        self._listeners: list[Callable[[str], None]] = []

    # -- reading -------------------------------------------------------------
    @property
    def text(self) -> str:
        with self._lock:
            return self._text

    @property
    def language(self) -> str:
        with self._lock:
            return self._language

    @property
    def name(self) -> str:
        with self._lock:
            return self._name

    def lines(self) -> list[str]:
        return self.text.splitlines()

    def numbered(self, start: int = 1, end: int = 0) -> str:
        """The buffer with line numbers, which is how it is shown to the model.

        Without numbers it cannot refer to a line, and a request to change one
        turns into a request to rewrite everything.
        """
        rows = self.lines()
        if not rows:
            return "(the canvas is empty)"
        last = end or len(rows)
        first = max(1, start)
        width = len(str(min(last, len(rows))))
        out = []
        for number in range(first, min(last, len(rows)) + 1):
            out.append(f"{number:>{width}} | {rows[number - 1]}")
        return "\n".join(out)

    # -- writing -------------------------------------------------------------
    def set(self, text: str, language: str = "", name: str = "") -> None:
        with self._lock:
            self._text = str(text or "")
            if language:
                self._language = language
            if name:
                self._name = name
        self._notify()

    def append(self, text: str) -> None:
        with self._lock:
            joiner = "" if not self._text or self._text.endswith("\n") else "\n"
            self._text = self._text + joiner + str(text or "")
        self._notify()

    def replace_lines(self, start: int, end: int, text: str) -> tuple[bool, str]:
        """Swap a span of lines. Returns (changed, message).

        One-based and inclusive, matching how the buffer is shown and how people
        talk about line numbers.
        """
        with self._lock:
            rows = self._text.splitlines()
            if start < 1 or start > len(rows) + 1:
                return False, f"line {start} is outside the canvas (1–{len(rows)})"
            last = max(start, end)
            replacement = str(text or "").splitlines()
            rows[start - 1:last] = replacement
            self._text = "\n".join(rows) + ("\n" if rows else "")
            count = last - start + 1
        self._notify()
        return True, f"replaced {count} line(s) with {len(replacement)}"

    def clear(self) -> None:
        self.set("")

    # -- notification --------------------------------------------------------
    def listen(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            try:
                fn(self._text)
            except Exception:
                continue


# One buffer for the application. The canvas is a single shared surface by
# design — two of them would raise the question of which one the model meant.
BUFFER = CanvasBuffer()
