"""Tools for reaching skills the system prompt only named."""
from __future__ import annotations

from .. import skills as skillmod
from . import DANGER_SAFE, Param, Tool, ToolResult


def register(reg, provider) -> None:
    """`provider` is a zero-arg callable returning the current list of Skills."""

    def skill_find(query: str) -> ToolResult:
        found = skillmod.search(provider(), query, limit=12)
        if not found:
            return ToolResult(f"No skill matches '{query}'. Carry on without one.")
        lines = [f"- {s.name}: {s.short(160)}" for s in found]
        return ToolResult("Matching skills (open one to read it):\n" + "\n".join(lines))

    def skill_open(name: str) -> ToolResult:
        by_name = {s.name: s for s in provider()}
        sk = by_name.get(name)
        if sk is None:
            near = skillmod.search(provider(), name, limit=5)
            hint = (" Closest: " + ", ".join(s.name for s in near)) if near else ""
            return ToolResult(f"No skill called '{name}'.{hint}", ok=False)
        body = sk.body()
        res = sk.resources()
        extra = ""
        if res:
            listed = "\n".join(f"  {r}" for r in res[:25])
            extra = (f"\n\nBundled files (read or run them with the file and shell tools, "
                     f"rooted at {sk.root}):\n{listed}")
        return ToolResult(f"# Skill: {sk.name}\n{body}{extra}", full=body + extra)

    reg.add(Tool("skill_find", "Search installed skills by keyword.",
                 [Param("query", "string", "What you are trying to do.", required=True)],
                 skill_find, DANGER_SAFE))
    reg.add(Tool("skill_open", "Read a skill's full instructions before using it.",
                 [Param("name", "string", "Exact skill name.", required=True)],
                 skill_open, DANGER_SAFE,
                 detail="Open a skill when its description matches the task, then follow it."))
