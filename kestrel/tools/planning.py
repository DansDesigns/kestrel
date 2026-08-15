"""Checklist tools."""
from __future__ import annotations

from . import DANGER_SAFE, Param, Tool, ToolResult
from ..todo import STATUSES


def register(reg, provider) -> None:
    """`provider` returns the live TodoList."""

    def plan(steps: str, title: str = "") -> ToolResult:
        tl = provider()
        tl.set_plan(steps, title)
        if not tl.items:
            return ToolResult("That plan had no steps in it. Give one step per line.",
                              ok=False)
        return ToolResult(f"Plan set ({len(tl.items)} steps). Start with step 1.\n"
                          + tl.render())

    def todo(id: int, status: str = "done", note: str = "") -> ToolResult:
        tl = provider()
        item = tl.update(id, status, note)
        if item is None:
            return ToolResult(f"No step {id}. Current plan:\n" + (tl.render() or "(empty)"),
                              ok=False)
        done, total = tl.progress
        tail = "\nEverything is closed — call finish." if tl.complete else ""
        return ToolResult(f"Step {item.id} -> {item.status} ({done}/{total} done).{tail}")

    def plan_add(text: str) -> ToolResult:
        tl = provider()
        item = tl.add(text)
        return ToolResult(f"Added step {item.id}: {item.text}")

    reg.add(Tool("plan", "Write the checklist for this task, one step per line.",
                 [Param("steps", "string", "Steps, one per line.", required=True),
                  Param("title", "string", "Short name for the task.")],
                 plan, DANGER_SAFE,
                 detail="Do this first for anything needing more than one step. "
                        "Three to seven steps is usually right."))
    reg.add(Tool("todo", "Update a checklist step as you go.",
                 [Param("id", "integer", "Step number.", required=True),
                  Param("status", "string", f"One of: {', '.join(STATUSES)}.", default="done"),
                  Param("note", "string", "Optional short note, e.g. why it is blocked.")],
                 todo, DANGER_SAFE,
                 detail="Mark a step doing when you start it and done when it works."))
    reg.add(Tool("plan_add", "Append a step you did not anticipate.",
                 [Param("text", "string", "The new step.", required=True)],
                 plan_add, DANGER_SAFE))
