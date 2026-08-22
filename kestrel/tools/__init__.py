"""Tool registry.

Tools carry two renderings of themselves: a full JSON schema for models that
support native tool calling, and a one-line signature for models that don't (or
for windows too small to afford schemas — a dozen JSON schemas is 1,500+ tokens,
which is a third of a 4k window before anyone has said anything).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DANGER_SAFE = "safe"      # reads only
DANGER_WRITE = "write"    # touches the filesystem
DANGER_EXEC = "exec"      # runs arbitrary code


@dataclass
class Param:
    name: str
    type: str = "string"
    desc: str = ""
    required: bool = False
    default: Any = None


@dataclass
class ToolResult:
    text: str                       # what the model sees (may be trimmed later)
    full: str = ""                  # complete output, spooled to disk if large
    ok: bool = True
    display: str = ""               # what the UI shows

    def __post_init__(self):
        if not self.full:
            self.full = self.text
        if not self.display:
            self.display = self.text


@dataclass
class Tool:
    name: str
    summary: str
    params: list[Param]
    handler: Callable[..., ToolResult]
    danger: str = DANGER_SAFE
    detail: str = ""

    def signature(self) -> str:
        bits = []
        for p in self.params:
            if p.required:
                bits.append(p.name)
            else:
                d = "" if p.default is None else f"={json.dumps(p.default)}"
                bits.append(f"{p.name}{d or '?'}")
        return f"{self.name}({', '.join(bits)})"

    def line(self, verbosity: int) -> str:
        if verbosity <= 0:
            return f"{self.signature()} — {self.summary}"
        out = [f"{self.signature()} — {self.summary}"]
        if verbosity >= 2:
            for p in self.params:
                if p.desc:
                    out.append(f"    {p.name}: {p.desc}")
            if verbosity >= 3 and self.detail:
                out.append(f"    {self.detail}")
        return "\n".join(out)

    def schema(self) -> dict:
        props: dict[str, dict] = {}
        required: list[str] = []
        for p in self.params:
            entry: dict[str, Any] = {"type": p.type}
            if p.desc:
                entry["description"] = p.desc
            props[p.name] = entry
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.summary,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }


class Registry:
    def __init__(self, workspace: Path, approver: Callable[[Tool, dict], bool] | None = None,
                 approval_mode: str = "safe"):
        self.workspace = Path(workspace)
        self.tools: dict[str, Tool] = {}
        self.approver = approver
        self.approval_mode = approval_mode
        self.artifacts = self.workspace / ".kestrel" / "artifacts"
        self._counter = 0

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self.tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def listing(self, verbosity: int) -> str:
        """The tools, at whatever length there is room for.

        At the tightest budget only the names appear, with tool_help to fetch a
        signature. A model cannot call a tool it has not been told exists — a
        folder path would not do, because it would need a tool to look in the
        folder — but it does not need every parameter of every tool in front of
        it to know what is available.
        """
        if verbosity <= 0:
            # Only what is needed to find the rest. A model cannot call a tool
            # it has never heard of, so these four have to be named — but the
            # other twenty do not, and naming them costs their tokens on every
            # turn of every conversation.
            return ("tool_list() names every tool. tool_help(name) gives one's "
                    "arguments. skill_index() lists the skills. finish(answer) "
                    "ends the task.\n"
                    f"There are {len(self.tools)} tools. Look before you assume "
                    "something is missing.")
        return "\n".join(t.line(verbosity) for t in self.tools.values())

    def help_for(self, name: str) -> str:
        tool = self.tools.get(str(name or "").strip())
        if tool is None:
            close = [n for n in self.tools if str(name)[:4] in n]
            return (f"There is no tool called {name}."
                    + (f" Did you mean: {', '.join(close)}?" if close else ""))
        text = tool.line(3)
        if getattr(tool, "detail", ""):
            text += "\n" + tool.detail
        return text

    def needs_approval(self, tool: Tool) -> bool:
        if self.approval_mode == "never":
            return False
        if self.approval_mode == "always":
            return True
        return tool.danger in (DANGER_WRITE, DANGER_EXEC)

    def spool(self, text: str, label: str = "out") -> Path:
        self._counter += 1
        self.artifacts.mkdir(parents=True, exist_ok=True)
        p = self.artifacts / f"{self._counter:03d}-{label}.txt"
        try:
            p.write_text(text, "utf-8")
        except OSError:
            pass
        return p

    def call(self, name: str, args: dict) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            close = [n for n in self.tools if n.startswith(name[:4])]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return ToolResult(f"No tool named '{name}'. Available: {', '.join(self.tools)}.{hint}",
                              ok=False)
        if self.needs_approval(tool) and self.approver is not None:
            if not self.approver(tool, args):
                return ToolResult(f"The user declined to run {tool.name}. "
                                  "Try a different approach or ask them what they'd prefer.",
                                  ok=False)
        clean: dict[str, Any] = {}
        for p in tool.params:
            if p.name in args:
                clean[p.name] = args[p.name]
            elif p.required:
                # The signature, not just the complaint: a model told only what
                # is missing has to guess the rest, and guesses the same way
                # again on the next attempt.
                given = ", ".join(sorted(args)) or "nothing"
                return ToolResult(
                    f"{tool.name} needs a '{p.name}' argument. "
                    f"Call it as {tool.signature()}. You passed: {given}.",
                    ok=False)
        try:
            return tool.handler(**clean)
        except TypeError as e:
            return ToolResult(f"{tool.name} rejected those arguments: {e}", ok=False)
        except Exception as e:  # tools must never take the loop down
            return ToolResult(f"{tool.name} failed: {type(e).__name__}: {e}", ok=False)


def write_index(reg, path) -> "Path | None":
    """The tool catalogue as a file, for the model and for a person.

    The prompt carries four names; this carries all of them with their
    arguments. Written on startup so it cannot drift from what is registered.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    lines = [f"# Tools ({len(reg.tools)} available)", "",
             "Call tool_help(name) for any of these.", ""]
    for name, tool in reg.tools.items():
        lines.append(f"## {name}")
        lines.append(tool.summary)
        lines.append(f"`{tool.signature()}`")
        if getattr(tool, "detail", ""):
            lines.append(tool.detail)
        lines.append("")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), "utf-8")
        return path
    except OSError:
        return None


def build_registry(cfg, skills_provider, approver=None, memory_provider=None,
                   todo_provider=None, persona_provider=None,
                   roster_provider=None, delegate_fn=None) -> Registry:
    """Assemble the default toolset."""
    from . import files, shell, skillset

    ws = cfg.workspace_path()
    reg = Registry(ws, approver=approver, approval_mode=cfg.approval)
    def tool_help(name: str) -> ToolResult:
        return ToolResult(reg.help_for(name))

    def tool_list(contains: str = "") -> ToolResult:
        names = [n for n in reg.tools
                 if not contains or contains.lower() in n.lower()]
        if not names:
            return ToolResult(f"No tool with '{contains}' in its name. "
                              f"All of them: {', '.join(reg.tools)}")
        return ToolResult(", ".join(names)
                          + "\n\ntool_help(name) for what one takes.")

    reg.add(Tool("tool_list", "Name every tool, or those matching a word.",
                 [Param("contains", "string", "Optional filter, e.g. canvas.")],
                 tool_list, DANGER_SAFE,
                 detail="The prompt names only the few needed to find the rest. "
                        "Call this when you want to know what is available."))
    reg.add(Tool("tool_help", "What arguments a tool takes.",
                 [Param("name", "string", "The tool's name.", required=True)],
                 tool_help, DANGER_SAFE,
                 detail="Only the names are listed when the context is small. "
                        "Read a tool's arguments here before calling it."))
    files.register(reg, ws)
    shell.register(reg, ws)
    skillset.register(reg, skills_provider)
    if getattr(cfg, "canvas_enabled", True):
        from . import canvas as canvastools
        canvastools.register(reg)
    if roster_provider is not None:
        from . import team as teamtools
        teamtools.register(reg, roster_provider, delegate_fn)
    if todo_provider is not None and cfg.todo_enabled:
        from . import planning
        planning.register(reg, todo_provider)
    if memory_provider is not None and cfg.memory.enabled:
        from . import memory as memtools
        memtools.register(reg, memory_provider, cfg.memory_scope)
    if persona_provider is not None:
        from ..persona import register_tool
        register_tool(reg, persona_provider)

    reg.add(Tool(
        name="finish",
        summary="End the task and give the user your answer.",
        params=[Param("answer", "string", "The complete reply for the user.", required=True)],
        handler=lambda answer: ToolResult(answer),
        danger=DANGER_SAFE,
    ))
    return reg
