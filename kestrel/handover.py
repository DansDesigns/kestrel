"""The handover.

A context window fills. When it does, the transcript is summarised and the
detail is gone — and starting a fresh conversation throws away even the summary.
Either way the work is mid-flight and the next turn begins knowing less than the
last one did.

The handover is the small, deliberate part of that state which is worth keeping:
what the task is, what has been done, what is next, which files were touched, and
what is in the way. A few hundred tokens, written to the project folder as
markdown, reloaded when a conversation resumes or a new one starts in the same
project.

It is not a transcript and does not try to be. A transcript is what the model
said; a handover is what a competent colleague would write on a card before
going home.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

FILENAME = "HANDOVER.md"

PROMPT = """Write a handover note for whoever picks this task up next.

Task: {task}

{plan}

Recent work:
{recent}

Write it under these exact headings, a few lines each, no preamble:

## Where this is
## What has been done
## What is next
## Files touched
## In the way

Be concrete: name files, decisions and errors. Leave a heading out entirely if
there is nothing true to say under it. Do not invent progress."""


@dataclass
class Handover:
    task: str = ""
    body: str = ""
    when: float = 0.0
    turns: int = 0
    files: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.body.strip()

    def markdown(self) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.when or time.time()))
        head = [f"# Handover", "", f"_{stamp}"]
        if self.turns:
            head[-1] += f" · after {self.turns} turn(s)"
        head[-1] += "_"
        if self.task:
            head += ["", f"**Task:** {self.task}"]
        return "\n".join(head + ["", self.body.strip(), ""])

    def block(self) -> str:
        """The compact form handed to the model when work resumes."""
        if self.empty:
            return ""
        lines = ["## Picking up from last time"]
        if self.task:
            lines.append(f"Task: {self.task}")
        lines.append(self.body.strip())
        lines.append("This is a summary, not a transcript — the detail is gone. "
                     "Check anything you are unsure of rather than assuming it.")
        return "\n".join(lines)


def path_for(workspace: str | Path) -> Path:
    return Path(workspace).expanduser() / FILENAME


def load(workspace: str | Path) -> Handover:
    path = path_for(workspace)
    try:
        text = path.read_text("utf-8")
    except OSError:
        return Handover()
    task = ""
    match = re.search(r"^\*\*Task:\*\*\s*(.+)$", text, re.M)
    if match:
        task = match.group(1).strip()
    body = text.split("\n## ", 1)
    body = ("## " + body[1]) if len(body) > 1 else ""
    try:
        when = path.stat().st_mtime
    except OSError:
        when = 0.0
    return Handover(task=task, body=body, when=when)


def save(workspace: str | Path, note: Handover) -> Path | None:
    if note.empty:
        return None
    path = path_for(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(note.markdown(), "utf-8")
        return path
    except OSError:
        return None


def clear(workspace: str | Path) -> None:
    try:
        path_for(workspace).unlink(missing_ok=True)
    except OSError:
        pass


def stale(note: Handover, hours: float = 72.0) -> bool:
    """Old enough that resuming it silently would be presumptuous."""
    return bool(note.when) and (time.time() - note.when) > hours * 3600
