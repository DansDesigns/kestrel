"""Team tools.

How one agent reaches another, and the shared surface they hand work over on.
Messages are asynchronous because the agents share a model and run one at a
time: pretending otherwise would be a lie about what is happening.
"""
from __future__ import annotations

from pathlib import Path

from . import DANGER_SAFE, DANGER_WRITE, Param, Tool, ToolResult


def register(reg, roster_provider, delegate_fn=None) -> None:
    """`roster_provider` returns the live Roster; `delegate_fn(to, task)` runs
    a specialist and returns (ok, what they said)."""

    def delegate(to: str, task: str) -> ToolResult:
        if delegate_fn is None:
            return ToolResult("Delegation is not available here.", ok=False)
        ok, said = delegate_fn(to, task)
        return ToolResult(said, ok=ok)

    def agent_list() -> ToolResult:
        roster = roster_provider()
        if roster is None or not roster.agents:
            return ToolResult("No other agents on this project.")
        lines = []
        for agent in roster.agents:
            here = " (you)" if agent.name == roster.active else ""
            busy = f" — {agent.activity}" if agent.activity else ""
            lines.append(f"{agent.name}{here}: {agent.speciality} "
                         f"[{agent.status}]{busy}")
        return ToolResult("\n".join(lines))

    def agent_send(to: str, message: str) -> ToolResult:
        roster = roster_provider()
        if roster is None:
            return ToolResult("There is no team on this project.", ok=False)
        ok, said = roster.send(roster.active, to, message)
        return ToolResult(said, ok=ok)

    def whiteboard_write(name: str, content: str) -> ToolResult:
        roster = roster_provider()
        if roster is None:
            return ToolResult("There is no whiteboard here.", ok=False)
        folder = roster.ensure_whiteboard()
        target = (folder / Path(name).name)
        try:
            target.write_text(content, "utf-8")
        except OSError as e:
            return ToolResult(f"Could not write it: {e}", ok=False)
        return ToolResult(f"Put {target.name} on the whiteboard "
                          f"({len(content.splitlines())} lines). Any agent can "
                          "read it.")

    def whiteboard_read(name: str = "") -> ToolResult:
        roster = roster_provider()
        if roster is None:
            return ToolResult("There is no whiteboard here.", ok=False)
        folder = roster.ensure_whiteboard()
        if not name:
            items = sorted(f.name for f in folder.iterdir() if f.is_file())
            return ToolResult("On the whiteboard: " + (", ".join(items) or "nothing yet"))
        target = folder / Path(name).name
        try:
            return ToolResult(target.read_text("utf-8", errors="replace")[:8000])
        except OSError:
            return ToolResult(f"There is no {name} on the whiteboard.", ok=False)

    if delegate_fn is not None:
        reg.add(Tool("delegate", "Hand a piece of work to another agent and "
                                 "wait for their answer.",
                     [Param("to", "string", "Their name.", required=True),
                      Param("task", "string",
                            "What you want done, in full — they cannot see "
                            "your conversation.", required=True)],
                     delegate, DANGER_SAFE,
                     detail="They work while you wait, then you read what came "
                            "back and decide what happens next. Give them "
                            "everything they need: they have their own "
                            "conversation and cannot see yours. Writing or "
                            "changing code goes to Builder, checking it to "
                            "Reviewer, writing it up to Scribe."))
    reg.add(Tool("agent_list", "Who else is working on this project.",
                 [], agent_list, DANGER_SAFE,
                 detail="Names, specialities and what each is doing."))
    reg.add(Tool("agent_send", "Send a message to another agent.",
                 [Param("to", "string", "Their name.", required=True),
                  Param("message", "string", "What to tell them.", required=True)],
                 agent_send, DANGER_SAFE,
                 detail="They read it when they next run, not immediately — you "
                        "share one model and take turns. Say what you need "
                        "rather than starting a conversation."))
    reg.add(Tool("whiteboard_write", "Put a file on the shared whiteboard.",
                 [Param("name", "string", "File name.", required=True),
                  Param("content", "string", "What to write.", required=True)],
                 whiteboard_write, DANGER_WRITE,
                 detail="How work is handed over: a file outlives the "
                        "conversation that produced it and a person can read it."))
    reg.add(Tool("whiteboard_read", "Read the whiteboard, or list what is on it.",
                 [Param("name", "string", "File to read. Omit to list them.")],
                 whiteboard_read, DANGER_SAFE))
