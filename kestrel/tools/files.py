"""Filesystem tools. Paths resolve inside the workspace unless absolute and allowed."""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from . import DANGER_SAFE, DANGER_WRITE, Param, Tool, ToolResult

MAX_READ_LINES = 400
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".kestrel", ".mypy_cache"}


class Sandbox:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise PermissionError(
                f"{p} is outside the workspace ({self.root}). "
                "Work inside the workspace, or ask the user to move the file in."
            )
        return p

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)


def register(reg, workspace: Path) -> None:
    sb = Sandbox(workspace)

    def list_dir(path: str = ".") -> ToolResult:
        d = sb.resolve(path)
        if not d.exists():
            return ToolResult(f"{sb.rel(d)} does not exist.", ok=False)
        if d.is_file():
            return ToolResult(f"{sb.rel(d)} is a file ({d.stat().st_size} bytes).")
        rows = []
        for item in sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if item.name in SKIP_DIRS:
                continue
            if item.is_dir():
                rows.append(f"{item.name}/")
            else:
                rows.append(f"{item.name}  {item.stat().st_size}b")
        body = "\n".join(rows) or "(empty)"
        return ToolResult(f"{sb.rel(d)}/\n{body}")

    def read_file(path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> ToolResult:
        f = sb.resolve(path)
        if not f.is_file():
            return ToolResult(f"No file at {sb.rel(f)}.", ok=False)
        try:
            text = f.read_text("utf-8", errors="replace")
        except OSError as e:
            return ToolResult(f"Cannot read {sb.rel(f)}: {e}", ok=False)
        lines = text.splitlines()
        start = max(1, int(offset)) - 1
        end = min(len(lines), start + max(1, int(limit)))
        chunk = lines[start:end]
        numbered = "\n".join(f"{start + i + 1:>5}  {ln}" for i, ln in enumerate(chunk))
        header = f"{sb.rel(f)} lines {start + 1}-{end} of {len(lines)}"
        more = ""
        if end < len(lines):
            more = f"\n[{len(lines) - end} more lines — read again with offset={end + 1}]"
        return ToolResult(f"{header}\n{numbered}{more}", full=text)

    def write_file(path: str, content: str) -> ToolResult:
        f = sb.resolve(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        existed = f.exists()
        f.write_text(content, "utf-8")
        verb = "Overwrote" if existed else "Wrote"
        return ToolResult(f"{verb} {sb.rel(f)} ({len(content)} chars, "
                          f"{content.count(chr(10)) + 1} lines).")

    def edit_file(path: str, find: str, replace: str) -> ToolResult:
        f = sb.resolve(path)
        if not f.is_file():
            return ToolResult(f"No file at {sb.rel(f)}.", ok=False)
        text = f.read_text("utf-8", errors="replace")
        hits = text.count(find)
        if hits == 0:
            return ToolResult(f"That exact text is not in {sb.rel(f)}. "
                              "Read the file and copy the target text verbatim, whitespace included.",
                              ok=False)
        if hits > 1:
            return ToolResult(f"That text appears {hits} times in {sb.rel(f)}. "
                              "Include surrounding lines so the match is unique.", ok=False)
        f.write_text(text.replace(find, replace, 1), "utf-8")
        return ToolResult(f"Edited {sb.rel(f)}.")

    def find_files(pattern: str, path: str = ".") -> ToolResult:
        d = sb.resolve(path)
        hits = []
        for p in d.rglob("*"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.is_file() and fnmatch.fnmatch(p.name, pattern):
                hits.append(sb.rel(p))
            if len(hits) >= 300:
                break
        body = "\n".join(sorted(hits)) or "(no matches)"
        return ToolResult(f"{len(hits)} match(es) for {pattern}:\n{body}", full=body)

    def grep(pattern: str, path: str = ".", glob: str = "*") -> ToolResult:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(f"Bad regex: {e}", ok=False)
        d = sb.resolve(path)
        targets = [d] if d.is_file() else [
            p for p in d.rglob(glob)
            if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
        ]
        out = []
        for p in targets[:2000]:
            try:
                for i, ln in enumerate(p.read_text("utf-8", errors="replace").splitlines(), 1):
                    if rx.search(ln):
                        out.append(f"{sb.rel(p)}:{i}: {ln.strip()[:200]}")
                        if len(out) >= 200:
                            raise StopIteration
            except StopIteration:
                break
            except OSError:
                continue
        body = "\n".join(out) or "(no matches)"
        return ToolResult(f"{len(out)} hit(s):\n{body}", full=body)

    reg.add(Tool("list_dir", "List a directory.",
                 [Param("path", "string", "Directory, relative to the workspace.", default=".")],
                 list_dir, DANGER_SAFE))
    reg.add(Tool("read_file", "Read a text file, one page at a time.",
                 [Param("path", "string", "File to read.", required=True),
                  Param("offset", "integer", "First line number.", default=1),
                  Param("limit", "integer", "How many lines.", default=MAX_READ_LINES)],
                 read_file, DANGER_SAFE,
                 detail="Output is line-numbered. Page through long files with offset."))
    reg.add(Tool("write_file", "Create or overwrite a file.",
                 [Param("path", "string", "File to write.", required=True),
                  Param("content", "string", "Full new contents.", required=True)],
                 write_file, DANGER_WRITE))
    reg.add(Tool("edit_file", "Replace one exact snippet in a file.",
                 [Param("path", "string", "File to edit.", required=True),
                  Param("find", "string", "Exact text to replace; must be unique.", required=True),
                  Param("replace", "string", "Replacement text.", required=True)],
                 edit_file, DANGER_WRITE))
    reg.add(Tool("find_files", "Find files by name pattern.",
                 [Param("pattern", "string", "Glob such as *.py.", required=True),
                  Param("path", "string", "Where to search.", default=".")],
                 find_files, DANGER_SAFE))
    reg.add(Tool("grep", "Search file contents with a regex.",
                 [Param("pattern", "string", "Regular expression.", required=True),
                  Param("path", "string", "Where to search.", default="."),
                  Param("glob", "string", "Restrict to matching filenames.", default="*")],
                 grep, DANGER_SAFE))
