"""Tools for reaching skills the system prompt only named."""
from __future__ import annotations

from .. import skills as skillmod
import subprocess
import sys

from . import DANGER_EXEC, DANGER_SAFE, Param, Tool, ToolResult


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
        warning = ""
        absent = sk.missing()
        if absent:
            # Before the instructions, not after. A model that reads a skill
            # through and only then finds it cannot run has spent those tokens
            # for nothing.
            warning = (f"NOT READY: this skill needs {', '.join(absent)}, "
                       f"{'which is' if len(absent) == 1 else 'which are'} not "
                       f"installed. Fix with: {sk.install_hint()}\n"
                       "skill_install does the python packages; programs you "
                       "must install yourself. Otherwise work without it and "
                       "say what you could not do.\n\n")
        return ToolResult(f"{warning}# Skill: {sk.name}\n{body}{extra}",
                          full=body + extra)

    def skill_index() -> ToolResult:
        found = provider()
        if not found:
            return ToolResult("No skills are installed.")
        lines = [f"{s.name} — {s.description or 'no description'}"
                 + (f"  [needs {', '.join(s.requires)}]" if s.requires else "")
                 for s in sorted(found, key=lambda s: s.name.lower())]
        return ToolResult(f"{len(found)} skills. Open one with skill_open.\n"
                          + "\n".join(lines))

    reg.add(Tool("skill_index", "List every installed skill with its description.",
                 [], skill_index, DANGER_SAFE,
                 detail="The prompt names only the ones that look relevant to "
                        "what you are doing. Read this when none of them fit "
                        "and you want to see what else is here."))
    reg.add(Tool("skill_find", "Search installed skills by keyword.",
                 [Param("query", "string", "What you are trying to do.", required=True)],
                 skill_find, DANGER_SAFE))
    def skill_install(name: str) -> ToolResult:
        sk = {s.name: s for s in provider()}.get(name)
        if sk is None:
            return ToolResult(f"No skill called '{name}'.", ok=False)
        absent = [a for a in sk.missing() if a.startswith("python:")]
        if not absent:
            others = sk.missing()
            if others:
                return ToolResult(
                    f"{', '.join(others)} cannot be installed from here — "
                    "they are programs rather than packages. "
                    + sk.install_hint(), ok=False)
            return ToolResult("Everything it needs is already installed.")
        packages = [a.split(":", 1)[1] for a in absent]
        command = [sys.executable, "-m", "pip", "install", *packages]
        try:
            out = subprocess.run(command, capture_output=True, text=True,
                                 timeout=600)
        except Exception as e:
            return ToolResult(f"Could not run pip: {e}", ok=False)
        if out.returncode != 0:
            tail = (out.stderr or out.stdout or "").strip().splitlines()[-4:]
            return ToolResult("pip failed:\n" + "\n".join(tail), ok=False)
        still = sk.missing()
        return ToolResult(
            f"Installed {', '.join(packages)}."
            + (f" Still missing: {', '.join(still)}." if still else ""))

    reg.add(Tool("skill_install",
                 "Install the python packages a skill needs.",
                 [Param("name", "string", "The skill's name.", required=True)],
                 skill_install, DANGER_EXEC,
                 detail="Only packages, and only the ones that skill declares. "
                        "Programs like ffmpeg have to be installed by the user."))
    reg.add(Tool("skill_open", "Read a skill's full instructions before using it.",
                 [Param("name", "string", "Exact skill name.", required=True)],
                 skill_open, DANGER_SAFE,
                 detail="Open a skill when its description matches the task, then follow it."))
