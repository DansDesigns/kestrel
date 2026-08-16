"""System prompt construction, sized to the window.

Four verbosity tiers. The nano tier is roughly 200 tokens including the tool
listing, which is what makes a 4k window workable.
"""
from __future__ import annotations

from .skills import Skill, index_lines

TOOL_BLOCK_OPEN = "<tool>"
TOOL_BLOCK_CLOSE = "</tool>"

_TEXT_PROTOCOL_TERSE = """To act, reply with exactly one block and nothing else:
<tool>
{"name": "tool_name", "arguments": {"key": "value"}}
</tool>
You get the result, then continue. Call finish when done."""

_TEXT_PROTOCOL_FULL = """## Calling a tool

To act, emit exactly one block, with no prose before or after it:

<tool>
{"name": "read_file", "arguments": {"path": "notes.md"}}
</tool>

The block must contain a single JSON object with "name" and "arguments".
Stop after the closing tag; the result comes back as the next message.
When the task is complete, call finish with your answer for the user.
"""

_GUIDANCE = {
    1: [
        "Take one step at a time and check the result before the next.",
        "Prefer reading a file over guessing its contents.",
    ],
    2: [
        "Work in small verified steps: act, read the result, then decide the next move.",
        "Never invent file contents, paths, or command output — look them up.",
        "If a skill's description matches the task, open it before starting.",
        "If a tool fails twice the same way, change approach rather than retrying.",
        "Keep answers to the user short and concrete.",
    ],
    3: [
        "Work in small verified steps: act, read the result, then decide the next move.",
        "Never invent file contents, paths, or command output — look them up.",
        "If a skill's description matches the task, open it with skill_open before starting, "
        "and follow its instructions over your own defaults.",
        "If a tool fails twice the same way, change approach rather than retrying.",
        "Tool output longer than the preview is saved to a file; page through it with "
        "read_file rather than asking for it again.",
        "State assumptions you had to make. If a request is ambiguous in a way that "
        "changes the work, ask before doing it.",
        "Keep answers to the user short and concrete.",
    ],
}


def build_system(
    *,
    workspace: str,
    tool_listing: str,
    skills: list[Skill],
    budget,
    dialect: str,
    persona: str = "",
    has_plan_tools: bool = False,
) -> str:
    v = budget.verbosity
    parts: list[str] = []

    # The persona layer replaces the stock identity line rather than stacking on
    # top of it, so voice costs nothing extra.
    if persona.strip():
        parts.append(persona.strip() + f"\nWorking directory: {workspace}")
    elif v == 0:
        parts.append(f"You are Kestrel, an agent with tools. Working directory: {workspace}")
    else:
        parts.append(
            "You are Kestrel, a capable agent that gets things done by using tools.\n"
            f"Working directory: {workspace}"
        )

    if dialect == "text":
        parts.append(_TEXT_PROTOCOL_TERSE if v <= 1 else _TEXT_PROTOCOL_FULL)

    header = "Tools:" if v == 0 else "## Tools\n"
    parts.append(f"{header}\n{tool_listing}")

    if skills:
        chars = 70 if v == 0 else (110 if v == 1 else 200)
        lines = index_lines(skills, budget.max_skills, chars)
        head = "Skills:" if v == 0 else (
            "## Skills\n\nThese are procedures you can load on demand. Only the name and "
            "description are here; open one to get its instructions.\n"
        )
        parts.append(f"{head}\n" + "\n".join(lines))

    if has_plan_tools:
        parts.append(_PLAN_RULES[0] if v <= 1 else _PLAN_RULES[1])

    rules = _GUIDANCE.get(max(1, v)) if v else None
    if rules:
        parts.append("## How to work\n\n" + "\n".join(f"- {r}" for r in rules))

    return "\n\n".join(parts).strip()


_PLAN_RULES = [
    "If a checklist already exists below, work through it in order — do not "
    "replace it. Otherwise call plan with one step per line. Mark each step "
    "doing when you start it and done only when you have checked it works.",
    """## The checklist

**If a checklist is already shown below, it is your instructions.** It may have
been written by hand rather than by you. Work through it in the order given,
starting from the first step that is not done. Do not call plan — that would
discard it. If a step turns out to be wrong or impossible, mark it blocked with
the reason, or call plan_add for a step it turns out to need; otherwise follow
what is written.

If there is no checklist and the task takes more than one step, call plan with
one step per line — three to seven is usually right. Either way, then work it:

- mark a step doing with todo before you start it
- mark it done only once you have evidence it worked — you ran the command, you
  read the file back, the test passed — not when you intend to do it
- mark it blocked with a short note if you cannot proceed
- call plan_add if the task turns out to need a step you did not foresee

The checklist appears in every prompt you receive. It is the only part of this
conversation guaranteed to survive; the transcript above it may be summarised
away. Keep it accurate and you can always pick up where you left off.""",
]

PLAN_PROMPT = """Break this task into a short checklist of concrete steps.

Three to seven steps. One per line, no numbering, no commentary. Each step should
be something you can tell is finished. If the task is a single action, reply with
that one line.

Task: %s"""

SUMMARY_PROMPT = (
    "Compress the notes below into at most {cap} tokens. Keep decisions made, facts "
    "discovered, file paths, and anything still outstanding. Drop pleasantries and "
    "repeated attempts. Write terse bullet points, no preamble.\n\n{blob}"
)
