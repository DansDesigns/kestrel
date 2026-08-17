"""Canvas tools.

The model's access to the shared editor. Reading is line-numbered and writing is
line-addressed, so changing one function does not mean reproducing the file.
"""
from __future__ import annotations

from pathlib import Path

from ..canvas import BUFFER
from . import DANGER_SAFE, DANGER_WRITE, Param, Registry, Tool, ToolResult


def register(reg: Registry) -> None:
    def canvas_read(start: int = 1, end: int = 0) -> ToolResult:
        return ToolResult(BUFFER.numbered(int(start or 1), int(end or 0)))

    def canvas_write(text: str, language: str = "") -> ToolResult:
        BUFFER.set(text, language=language)
        return ToolResult(f"Canvas replaced — {len(BUFFER.lines())} line(s). It "
                          "is on screen; do not repeat it in your reply.")

    def canvas_append(text: str) -> ToolResult:
        BUFFER.append(text)
        return ToolResult(f"Appended — {len(BUFFER.lines())} line(s) total.")

    def canvas_edit(start: int, end: int, text: str) -> ToolResult:
        changed, message = BUFFER.replace_lines(int(start), int(end), text)
        if not changed:
            return ToolResult(message, ok=False)
        return ToolResult(f"{message} — {len(BUFFER.lines())} line(s) total.")

    def canvas_save(path: str = "") -> ToolResult:
        root = Path(reg.workspace).expanduser().resolve()
        target = (root / (path or BUFFER.name)).resolve()
        try:
            target.relative_to(root)         # the workspace is the boundary
        except ValueError:
            return ToolResult("that path is outside the workspace", ok=False)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(BUFFER.text, "utf-8")
        except OSError as e:
            return ToolResult(f"could not save: {e}", ok=False)
        return ToolResult(f"Saved the canvas to {target.name} "
                          f"({len(BUFFER.lines())} lines).")

    reg.add(Tool("canvas_save", "Write the canvas to a file in the workspace.",
                 [Param("path", "string", "Where to save it, e.g. src/main.py.")],
                 canvas_save, DANGER_WRITE,
                 detail="The normal way to finish a piece of code: write it in "
                        "the canvas, then save it here."))
    reg.add(Tool("canvas_read", "Read the shared canvas, with line numbers.",
                 [Param("start", "integer", "First line, default 1."),
                  Param("end", "integer", "Last line, default the end.")],
                 canvas_read, DANGER_SAFE,
                 detail="Read before editing so your line numbers match what is "
                        "actually there."))
    reg.add(Tool("canvas_write", "Replace the whole canvas with this text.",
                 [Param("text", "string", "The complete new contents.",
                        required=True),
                  Param("language", "string", "python, javascript, shell…")],
                 canvas_write, DANGER_WRITE,
                 detail="Use this for code you are writing. It appears in the "
                        "canvas for the user to read, edit and save, so it does "
                        "not need repeating in your reply."))
    reg.add(Tool("canvas_append", "Add text to the end of the canvas.",
                 [Param("text", "string", "What to add.", required=True)],
                 canvas_append, DANGER_WRITE))
    reg.add(Tool("canvas_edit", "Replace lines start..end of the canvas.",
                 [Param("start", "integer", "First line to replace.",
                        required=True),
                  Param("end", "integer", "Last line, inclusive.", required=True),
                  Param("text", "string", "What to put there.", required=True)],
                 canvas_edit, DANGER_WRITE,
                 detail="Line numbers are one-based and inclusive, as shown by "
                        "canvas_read."))
