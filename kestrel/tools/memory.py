"""Tools for the long-term memory store."""
from __future__ import annotations

from . import DANGER_SAFE, DANGER_WRITE, Param, Tool, ToolResult
from ..memory import KINDS


def register(reg, store_provider, scope_provider) -> None:
    """`store_provider` returns a MemoryStore or None when memory is disabled."""

    def _store():
        s = store_provider()
        if s is None:
            raise RuntimeError("Long-term memory is switched off in Settings.")
        return s

    def remember(text: str, kind: str = "fact", importance: int = 3) -> ToolResult:
        store = _store()
        try:
            mid, created = store.remember(text, kind, importance, source="agent",
                                          scope=scope_provider())
        except ValueError as e:
            # Told, rather than silently dropped, so the model stops trying.
            return ToolResult(f"Not stored: {e}. Memory is for things still true "
                              "in a month — preferences, setup, decisions. Not "
                              "times, dates, or tool output.", ok=False)
        verb = "Remembered" if created else "Already knew that; kept it"
        return ToolResult(f"{verb} (#{mid}).")

    def recall(query: str, limit: int = 5) -> ToolResult:
        store = _store()
        hits = store.search(query, limit=max(1, min(int(limit), 20)),
                            scope=scope_provider())
        if not hits:
            return ToolResult(f"Nothing remembered about '{query}'.")
        lines = [f"#{m.id} [{m.kind}] {m.text}" for m in hits]
        return ToolResult("Recalled:\n" + "\n".join(lines))

    def forget(memory_id: int) -> ToolResult:
        store = _store()
        ok = store.forget(int(memory_id))
        return ToolResult(f"Forgot #{memory_id}." if ok else f"No memory #{memory_id}.",
                          ok=ok)

    reg.add(Tool(
        "remember", "Save something worth knowing in future sessions.",
        [Param("text", "string", "One durable fact, still true in a month.", required=True),
         Param("kind", "string", f"One of: {', '.join(KINDS)}.", default="fact"),
         Param("importance", "integer", "1 trivial to 5 essential.", default=3)],
        remember, DANGER_WRITE,
        detail="For durable things only — preferences, setup, decisions, "
               "procedures that worked. Not file contents or this turn's output."))
    reg.add(Tool(
        "recall", "Search what you remember from earlier sessions.",
        [Param("query", "string", "What you want to know.", required=True),
         Param("limit", "integer", "How many to return.", default=5)],
        recall, DANGER_SAFE))
    reg.add(Tool(
        "forget", "Delete a memory that is wrong or out of date.",
        [Param("memory_id", "integer", "The #id shown next to the memory.", required=True)],
        forget, DANGER_WRITE))
