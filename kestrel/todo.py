"""The task checklist.

The agent keeps a plan, and that plan is re-rendered into every single prompt.
That is the point: the transcript gets compacted away, but the checklist does
not, so the model always knows what it set out to do, what it has finished, and
what it is on. It is the cheapest possible working memory — a six-step plan
costs about fifty tokens.

It is also the honest progress indicator. The interface shows exactly what the
model believes the state to be, rather than a spinner.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

TODO, DOING, DONE, BLOCKED = "todo", "doing", "done", "blocked"
STATUSES = [TODO, DOING, DONE, BLOCKED]
# What the model sees in the prompt stays plain ASCII; the interface draws its
# own marks, because a tick reads as success in a way that "x" does not.
MARKS = {TODO: " ", DOING: ">", DONE: "x", BLOCKED: "!"}
DISPLAY_MARKS = {TODO: "○", DOING: "▸", DONE: "✓", BLOCKED: "!"}
_ALIASES = {
    "pending": TODO, "open": TODO, "not started": TODO, "todo": TODO,
    "active": DOING, "in progress": DOING, "in_progress": DOING,
    "started": DOING, "working": DOING, "doing": DOING,
    "complete": DONE, "completed": DONE, "finished": DONE, "done": DONE,
    "blocked": BLOCKED, "stuck": BLOCKED, "failed": BLOCKED,
}


def normalise_status(value: str) -> str:
    v = str(value or "").strip().lower()
    return _ALIASES.get(v, TODO if v not in STATUSES else v)


@dataclass
class TodoItem:
    id: int
    text: str
    status: str = TODO
    note: str = ""

    def line(self, current: bool = False) -> str:
        s = f"{self.id}. [{MARKS.get(self.status, ' ')}] {self.text}"
        if self.note:
            s += f" — {self.note}"
        if current:
            s += "   <- you are here"
        return s


@dataclass
class TodoList:
    """Persisted per workspace so a plan survives restarts, not just compaction."""

    path: Path | None = None
    title: str = ""
    items: list[TodoItem] = field(default_factory=list)
    updated: float = 0.0
    stale_steps: int = 0        # steps taken since the plan last changed
    root: Path | None = None    # the project folder PLAN.md is written into

    # -- lifecycle -----------------------------------------------------------
    @classmethod
    def load(cls, workspace: str | Path) -> "TodoList":
        p = Path(workspace) / ".kestrel" / "todo.json"
        tl = cls(path=p, root=Path(workspace))
        if p.exists():
            try:
                raw = json.loads(p.read_text("utf-8"))
                tl.title = raw.get("title", "")
                tl.updated = float(raw.get("updated") or 0)
                tl.items = [TodoItem(**{k: v for k, v in it.items()
                                        if k in TodoItem.__annotations__})
                            for it in raw.get("items", [])]
            except (OSError, ValueError, TypeError):
                pass
        return tl

    MARKDOWN_MARKS = {TODO: " ", DOING: ">", DONE: "x", BLOCKED: "!"}

    def markdown(self) -> str:
        """The plan as a document, not a data structure.

        Written beside the project's files so the state of the work is legible
        without Kestrel running — in an editor, on a phone, in a diff.
        """
        done, total = self.progress
        title = self.title.strip() or "Plan"
        lines = [f"# {title}", ""]
        stamp = time.strftime("%Y-%m-%d %H:%M")
        lines.append(f"Updated {stamp} · {done} of {total} done")
        lines.append("")
        for item in self.items:
            mark = self.MARKDOWN_MARKS.get(item.status, " ")
            note = f" — {item.note}" if item.note else ""
            lines.append(f"- [{mark}] {item.text}{note}")
        if any(i.status == BLOCKED for i in self.items):
            lines += ["", "`!` marks a step that could not be completed."]
        if any(i.status == DOING for i in self.items):
            lines += ["", "`>` marks the step in progress."]
        return "\n".join(lines) + "\n"

    def write_markdown(self) -> Path | None:
        """Keep PLAN.md beside the work, updated on every change."""
        if not self.root:
            return None
        target = Path(self.root).expanduser() / "PLAN.md"
        try:
            if not self.items:
                target.unlink(missing_ok=True)
                return None
            target.write_text(self.markdown(), "utf-8")
            return target
        except OSError:
            return None

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "title": self.title, "updated": self.updated,
                "items": [asdict(i) for i in self.items],
            }, indent=1), "utf-8")
            # The JSON is the state; the markdown is the same thing for a
            # person, kept beside the project's own files.
            self.write_markdown()
        except OSError:
            pass

    # -- mutation ------------------------------------------------------------
    def set_plan(self, steps, title: str = "") -> "TodoList":
        """Replace the plan. Accepts a list, or a newline/numbered string, since
        small models produce all three shapes."""
        self.items = []
        for text in _as_steps(steps):
            self.items.append(TodoItem(id=len(self.items) + 1, text=text))
        self.title = title or self.title
        self._touch()
        return self

    def add(self, text: str) -> TodoItem:
        item = TodoItem(id=(max((i.id for i in self.items), default=0) + 1),
                        text=" ".join(str(text).split())[:300])
        self.items.append(item)
        self._touch()
        return item

    def update(self, item_id: int, status: str = "", note: str = "") -> TodoItem | None:
        item = self.get(item_id)
        if item is None:
            return None
        if status:
            item.status = normalise_status(status)
            # Only one step is ever in flight.
            if item.status == DOING:
                for other in self.items:
                    if other is not item and other.status == DOING:
                        other.status = TODO
        if note:
            item.note = " ".join(str(note).split())[:160]
        self._touch()
        return item

    def remove(self, item_id: int) -> bool:
        """Drop a step. Ids are deliberately not renumbered: the model refers to
        them by number, and shifting them under it causes it to close the wrong
        step."""
        item = self.get(item_id)
        if item is None:
            return False
        self.items.remove(item)
        self._touch()
        return True

    def clear_done(self) -> int:
        before = len(self.items)
        self.items = [i for i in self.items if i.status != DONE]
        self._touch()
        return before - len(self.items)

    def clear(self) -> None:
        self.items = []
        self.title = ""
        self._touch()

    def _touch(self) -> None:
        self.updated = time.time()
        self.stale_steps = 0
        self.save()

    def to_state(self) -> dict:
        """The plan as plain data, for storing alongside a conversation."""
        return {"title": self.title, "updated": self.updated,
                "items": [asdict(i) for i in self.items]}

    def load_state(self, state: dict | None) -> None:
        """Adopt a stored plan, replacing whatever is current.

        A checklist belongs to the conversation that produced it: returning to
        an old conversation should show the plan as it was left, and starting a
        new one should start from nothing.
        """
        state = state or {}
        self.title = str(state.get("title") or "")
        self.items = []
        for raw in state.get("items") or []:
            if not isinstance(raw, dict) or not raw.get("text"):
                continue
            self.items.append(TodoItem(
                id=int(raw.get("id") or len(self.items) + 1),
                text=str(raw.get("text")),
                status=normalise_status(raw.get("status", TODO)),
                note=str(raw.get("note") or "")))
        self._touch()

    # -- reading -------------------------------------------------------------
    def get(self, item_id: int) -> TodoItem | None:
        try:
            item_id = int(item_id)
        except (TypeError, ValueError):
            return None
        return next((i for i in self.items if i.id == item_id), None)

    @property
    def current(self) -> TodoItem | None:
        doing = next((i for i in self.items if i.status == DOING), None)
        if doing:
            return doing
        return next((i for i in self.items if i.status == TODO), None)

    @property
    def progress(self) -> tuple[int, int]:
        return sum(1 for i in self.items if i.status == DONE), len(self.items)

    @property
    def complete(self) -> bool:
        return bool(self.items) and all(
            i.status in (DONE, BLOCKED) for i in self.items)

    def render(self, counter=None, token_budget: int = 0) -> str:
        """The block injected into every prompt."""
        if not self.items:
            return ""
        done, total = self.progress
        head = f"Plan ({done}/{total} done)"
        if self.title:
            head += f" — {self.title}"
        current = self.current
        lines = [head + ":"]
        for item in self.items:
            lines.append(item.line(current is item))
        block = "\n".join(lines)

        if counter is not None and token_budget and counter.count(block) > token_budget:
            # Too long to send whole: keep the head, the current step and its
            # neighbours, and summarise the rest by count.
            idx = self.items.index(current) if current in self.items else 0
            lo, hi = max(0, idx - 1), min(len(self.items), idx + 3)
            kept = self.items[lo:hi]
            lines = [head + ":"]
            if lo:
                lines.append(f"… {lo} earlier step(s)")
            lines += [i.line(current is i) for i in kept]
            if hi < len(self.items):
                lines.append(f"… {len(self.items) - hi} later step(s)")
            block = "\n".join(lines)
        return block

    def nudge(self) -> str:
        """Reminder appended to a tool result when the plan has gone stale.

        Models drift: they work through the task and quietly stop maintaining
        the checklist. A short prod at the point of action costs a dozen tokens
        and keeps the plan honest.
        """
        if not self.items or self.complete:
            return ""
        if self.stale_steps < 3:
            return ""
        cur = self.current
        if cur is None:
            return ""
        return (f"\n[plan reminder: step {cur.id} is still open — mark it done "
                f"with todo, or revise the plan if it changed]")


def _as_steps(steps) -> list[str]:
    if isinstance(steps, str):
        raw = re.split(r"[\n;]+", steps)
    elif isinstance(steps, (list, tuple)):
        raw = []
        for s in steps:
            raw.extend(re.split(r"[\n;]+", str(s)) if isinstance(s, str) else [str(s)])
    else:
        raw = [str(steps)]
    out = []
    for line in raw:
        text = re.sub(r"^\s*(?:\d+[.)]|[-*+]|\[.\])\s*", "", line).strip()
        text = " ".join(text.split())
        if text:
            out.append(text[:300])
    return out[:24]
