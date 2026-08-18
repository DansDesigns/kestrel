"""The thinking log.

A reasoning trace is discarded after the turn that produced it — sending it back
costs a fortune in context and confuses most models. The cost of throwing it all
away is that the model cannot tell it has been here before, which is how a small
model ends up thinking the same thought at step 2, step 5 and step 9.

What is kept instead is a one-line summary per step, written to a file beside the
project. A dozen of those cost a few dozen tokens, and they answer the question
the model actually needs answered: what have I already considered?

The file is markdown for the same reason PLAN.md is — the reasoning behind a
piece of work is worth reading later, by a person, without Kestrel running.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

FILENAME = "THOUGHTS.md"
MAX_ENTRIES = 400
RECALL = 8              # how many recent thoughts to show the model


def summarise(text: str, limit: int = 160) -> str:
    """One line from a trace that may run to thousands of words.

    The opening sentences carry the intent; the rest is working. Deliberately
    crude — this is an index, not a précis.
    """
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    out = ""
    for part in parts:
        if len(out) + len(part) > limit:
            break
        out = f"{out} {part}".strip()
    return (out or cleaned)[:limit].rstrip(" ,;:")


def fingerprint(text: str) -> str:
    """A loose key for 'the model already thought this'.

    Word order and punctuation vary between repetitions of the same idea, so the
    key is the sorted set of substantial words.
    """
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    meaty = sorted({w for w in words if len(w) > 3})[:12]
    return " ".join(meaty)


@dataclass
class Thought:
    step: int
    text: str
    when: float = 0.0
    repeats: int = 1
    task: str = ""
    session: str = ""

    @property
    def key(self) -> str:
        return fingerprint(self.text)


@dataclass
class ThoughtLog:
    root: Path | None = None
    task: str = ""
    session: str = ""
    entries: list[Thought] = field(default_factory=list)

    @classmethod
    def load(cls, workspace: str | Path) -> "ThoughtLog":
        log = cls(root=Path(workspace).expanduser())
        path = log.path
        if path is None or not path.exists():
            return log
        current_task = ""
        current_session = ""
        try:
            for line in path.read_text("utf-8").splitlines():
                session_head = re.match(r"^# (?!Thinking log)(.*)$", line.strip())
                if session_head:
                    current_session = session_head.group(1)
                    continue
                heading = re.match(r"^## (.*)$", line.strip())
                if heading:
                    current_task = heading.group(1)
                    continue
                match = re.match(r"^- \*\*(\d+)\*\*\s+(.*)$", line.strip())
                if match:
                    log.entries.append(Thought(step=int(match.group(1)),
                                               text=match.group(2),
                                               task=current_task,
                                               session=current_session))
        except OSError:
            pass
        return log

    @property
    def path(self) -> Path | None:
        return (self.root / FILENAME) if self.root else None

    # -- writing -------------------------------------------------------------
    def add(self, step: int, text: str) -> tuple[bool, int]:
        """Record a thought. Returns (was_new, times_seen).

        A repetition is counted rather than appended: the useful signal is that
        it happened again, not another copy of it.
        """
        summary = summarise(text)
        if len(summary) < 12:
            return False, 0
        key = fingerprint(summary)
        for entry in reversed([e for e in self.entries[-40:]
                               if (not self.task or e.task == self.task)
                               and (not self.session or e.session == self.session)]):
            if entry.key == key:
                entry.repeats += 1
                self.save()
                return False, entry.repeats
        self.entries.append(Thought(step=step, text=summary, when=time.time(),
                                    task=self.task, session=self.session))
        self.entries = self.entries[-MAX_ENTRIES:]
        self.save()
        return True, 1

    def start_session(self, session: str) -> None:
        """Name the conversation these thoughts belong to."""
        self.session = " ".join(str(session or "").split())[:60]

    def start_task(self, task: str) -> None:
        """Name what is being worked on now.

        Recall is per task, not per project. Reasoning about last week's
        installer is not context for today's question — carrying it over is how
        a fresh request arrives with a page of irrelevant history attached.
        """
        self.task = " ".join(str(task or "").split())[:120]

    def clear(self) -> None:
        self.entries = []
        path = self.path
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def save(self) -> None:
        path = self.path
        if path is None or not self.entries:
            return
        lines = ["# Thinking log", ""]
        if self.task:
            lines += [f"Current task: {self.task}", ""]
        lines.append("One line per step, so the reasoning behind the work "
                     "survives the conversation it happened in.")
        lines.append("")
        seen = (None, None)
        for entry in self.entries[-160:]:
            if (entry.session, entry.task) != seen:
                if entry.session != seen[0]:
                    lines += ["", f"# {entry.session or 'unnamed conversation'}"]
                seen = (entry.session, entry.task)
                lines += ["", f"## {entry.task or 'general'}", ""]
            again = f"  _(considered {entry.repeats} times)_" if entry.repeats > 1 else ""
            lines.append(f"- **{entry.step}** {entry.text}{again}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n", "utf-8")
        except OSError:
            pass

    # -- reading -------------------------------------------------------------
    def block(self, limit: int = RECALL) -> str:
        """The compact form shown to the model each step."""
        if not self.entries:
            return ""
        # Recall is narrowed twice: to this conversation, and to this task
        # within it. Reasoning from another conversation is someone else's
        # working, however recent.
        mine = [e for e in self.entries
                if e.task == self.task
                and (not self.session or e.session == self.session)] if self.task else []
        if not mine:
            return ""
        recent = mine[-limit:]
        lines = ["## Already considered"]
        for entry in recent:
            again = f" (x{entry.repeats})" if entry.repeats > 1 else ""
            lines.append(f"- step {entry.step}: {entry.text}{again}")
        lines.append("Do not repeat this reasoning; act on it or go further.")
        return "\n".join(lines)

    def looping(self, threshold: int = 3) -> str:
        """The thought being circled, if there is one."""
        for entry in reversed(self.entries[-6:]):
            if entry.repeats >= threshold:
                return entry.text
        return ""
