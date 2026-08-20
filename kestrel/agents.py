"""Several agents, one model.

The usual way to build a team of agents is to give each one its own model, which
on a laptop means loading none of them. Kestrel does the opposite: one set of
weights stays loaded and the *role* is swapped around it — persona, system
prompt, conversation, and the tools each is allowed. Switching costs a prompt
rebuild rather than a model load, so a four-agent team runs on the same hardware
as one agent.

What makes that a team rather than a costume change is the two things they
share:

  **A whiteboard** — a folder in the project every agent can read and write.
  Work is handed over as files, which survive the conversation that produced
  them and can be read by a person.

  **Mailboxes** — a message from one agent to another, delivered at the start of
  the recipient's next turn. Asynchronous on purpose: agents run one at a time
  on one model, so anything else would be a lie about what is happening.

Each agent keeps its own conversation. That is the point of the separation — a
reviewer that has read the implementer's every keystroke is not a reviewer, it
is the same context with a different hat on.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

WHITEBOARD = "whiteboard"
STATE_FILE = "agents.json"

IDLE, WORKING, WAITING, BLOCKED = "idle", "working", "waiting", "blocked"


@dataclass
class Message:
    sender: str
    text: str
    when: float = field(default_factory=time.time)
    read: bool = False

    def line(self) -> str:
        return f"from {self.sender}: {self.text}"


@dataclass
class AgentProfile:
    """A role the shared model can take."""

    name: str
    speciality: str = ""
    # What the role does. Character — tone, manner, background — belongs to a
    # persona, which is a richer thing with its own tiering and its own file.
    brief: str = ""
    persona_file: str = ""          # a persona for this role, or "" to inherit
    tools: list[str] = field(default_factory=list)   # empty means all of them
    status: str = IDLE
    activity: str = ""              # what it is doing, in a few words
    history: list[dict] = field(default_factory=list)
    inbox: list[Message] = field(default_factory=list)
    turns: int = 0

    @property
    def unread(self) -> int:
        return sum(1 for m in self.inbox if not m.read)

    def summary(self) -> str:
        bits = [self.speciality or "no speciality set"]
        if self.turns:
            bits.append(f"{self.turns} turn(s)")
        if self.unread:
            bits.append(f"{self.unread} unread")
        return " · ".join(bits)

    def briefing(self, voice: str = "", speaking_as: str = "") -> str:
        """Who the model is being, said once.

        A persona and a role both want to open with "You are …", and a prompt
        that says it four times over — as Quartermaster, as Lead, as the one
        the user talks to, as one of several agents — is asking the model to
        decide which of them it believes. The name, the job and the manner are
        therefore assembled into a single statement.
        """
        opening = f"You are {self.name}"
        if speaking_as and speaking_as.lower() != self.name.lower():
            opening += f", speaking as {speaking_as}"
        if self.speciality:
            opening += f". Your speciality is {self.speciality}"
        lines = [opening + "."]
        if voice:
            lines.append(voice.strip())
        if self.brief:
            # Any stray "You are …" in a hand-written brief is dropped for the
            # same reason: the opening line has already said who this is.
            from .agent import strip_identity
            lines.append(strip_identity(self.brief) or self.brief)
        lines.append(
            "Others share this project with you: hand work over as files on the "
            "whiteboard, and agent_send tells one of them something — they read "
            "it when they next run.")
        return "\n".join(lines)

    def deliver(self, message: Message) -> None:
        self.inbox.append(message)

    def take_unread(self) -> list[Message]:
        fresh = [m for m in self.inbox if not m.read]
        for m in fresh:
            m.read = True
        # A mailbox is not an archive; the last fifty are plenty to look back on.
        self.inbox = self.inbox[-50:]
        return fresh


class Roster:
    """The agents in a project, and the state they share."""

    def __init__(self, workspace: str | Path):
        self.root = Path(workspace).expanduser()
        self.agents: list[AgentProfile] = []
        self.active: str = ""
        self.load()

    # -- the shared surfaces -------------------------------------------------
    @property
    def whiteboard(self) -> Path:
        return self.root / WHITEBOARD

    def ensure_whiteboard(self) -> Path:
        try:
            self.whiteboard.mkdir(parents=True, exist_ok=True)
            readme = self.whiteboard / "README.md"
            if not readme.exists():
                readme.write_text(
                    "# Whiteboard\n\nShared between the agents on this project. "
                    "Anything here can be read and changed by any of them, and "
                    "by you.\n", "utf-8")
        except OSError:
            pass
        return self.whiteboard

    # -- membership ----------------------------------------------------------
    def get(self, name: str) -> AgentProfile | None:
        low = str(name or "").strip().lower()
        return next((a for a in self.agents if a.name.lower() == low), None)

    def current(self) -> AgentProfile | None:
        return self.get(self.active) or (self.agents[0] if self.agents else None)

    def add(self, name: str, speciality: str = "", brief: str = "",
            persona_file: str = "") -> AgentProfile:
        existing = self.get(name)
        if existing:
            return existing
        agent = AgentProfile(name=name.strip()[:40],
                             speciality=speciality.strip()[:120],
                             brief=brief.strip(), persona_file=persona_file)
        self.agents.append(agent)
        if not self.active:
            self.active = agent.name
        self.save()
        return agent

    def remove(self, name: str) -> bool:
        agent = self.get(name)
        if agent is None:
            return False
        self.agents.remove(agent)
        if self.active.lower() == name.lower():
            self.active = self.agents[0].name if self.agents else ""
        self.save()
        return True

    def switch(self, name: str) -> AgentProfile | None:
        agent = self.get(name)
        if agent is None:
            return None
        self.active = agent.name
        self.save()
        return agent

    def send(self, sender: str, to: str, text: str) -> tuple[bool, str]:
        target = self.get(to)
        if target is None:
            known = ", ".join(a.name for a in self.agents) or "none"
            return False, f"There is no agent called {to}. There is: {known}."
        if target.name.lower() == str(sender).lower():
            return False, "That is you."
        target.deliver(Message(sender=sender, text=" ".join(str(text).split())[:800]))
        if target.status == IDLE:
            target.status = WAITING
            target.activity = f"message from {sender}"
        self.save()
        return True, f"Sent to {target.name}. They will see it when they next run."

    def note(self, name: str, status: str = "", activity: str = "") -> None:
        agent = self.get(name)
        if agent is None:
            return
        if status:
            agent.status = status
        if activity:
            agent.activity = " ".join(str(activity).split())[:80]
        self.save()

    # -- persistence ---------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.root / ".kestrel" / STATE_FILE

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            # Fresh objects, not a copy of a shared list: the same profiles
            # would otherwise be handed to every project, so a persona chosen
            # in one would appear in the next.
            self.agents = _default_team()
            self.active = self.agents[0].name if self.agents else ""
            return
        self.agents = []
        for item in raw.get("agents", []):
            try:
                inbox = [Message(**m) for m in item.pop("inbox", [])]
                agent = AgentProfile(**item)
                agent.inbox = inbox
                self.agents.append(agent)
            except (TypeError, ValueError):
                continue
        self.active = raw.get("active", "")
        if not self.agents:
            self.agents = _default_team()
            self.active = self.agents[0].name

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"active": self.active, "agents": []}
            for agent in self.agents:
                item = asdict(agent)
                item["inbox"] = [asdict(m) for m in agent.inbox]
                payload["agents"].append(item)
            self.path.write_text(json.dumps(payload, indent=1), "utf-8")
        except OSError:
            pass


# What each role is for, in the words a task is likely to use. Used to suggest
# a specialist when the lead does not name one.
ROUTING = {
    "Builder": ("write", "implement", "add", "create", "build", "code", "fix",
                "refactor", "script", "function", "class", "module", "install",
                "compile", "bug", "error", "patch", "feature"),
    "Reviewer": ("review", "check", "audit", "verify", "test", "inspect",
                 "critique", "correct", "safe", "secure", "quality"),
    "Scribe": ("document", "write up", "notes", "readme", "explain", "summar",
               "changelog", "comment", "guide", "instructions"),
}


def route(text: str, names: list[str]) -> str:
    """Which specialist a task most likely belongs to, or "" if unclear.

    Deliberately a heuristic and deliberately allowed to abstain. The lead
    decides; this only offers an opinion when nobody has, and a confident wrong
    answer would be worse than none — the work would go to someone who is not
    equipped for it and the lead would never know.
    """
    low = " ".join(str(text or "").lower().split())
    if not low:
        return ""
    scores: dict[str, int] = {}
    for name, words in ROUTING.items():
        if name not in names:
            continue
        hits = sum(1 for w in words if w in low)
        if hits:
            scores[name] = hits
    if not scores:
        return ""
    best = max(scores, key=scores.get)
    # A tie is not a decision.
    if list(scores.values()).count(scores[best]) > 1:
        return ""
    return best


def _default_team() -> list[AgentProfile]:
    """A small team that covers the shape of most work.

    Deliberately few. Every agent is another conversation to keep straight, and
    a team of nine on one model spends its time waiting rather than working.
    """
    return [
        AgentProfile(
            name="Lead",
            speciality="breaking work down and deciding what happens next",
            brief="The user talks to you. You keep the plan honest "
                  "and you delegate: writing or changing code goes to Builder, "
                  "checking it goes to Reviewer, writing it up goes to Scribe. "
                  "Use the delegate tool rather than doing their work yourself, "
                  "then read what came back and decide what happens next. Small "
                  "questions you can simply answer."),
        AgentProfile(
            name="Builder",
            speciality="writing and changing code",
            brief="You do the writing. You work in the canvas, save real files, "
                  "and you say what you changed rather than pasting it back."),
        AgentProfile(
            name="Reviewer",
            speciality="reading code for faults before they are shipped",
            brief="Look for what will break. You are specific about lines "
                  "and reasons, and you say when something is fine rather than "
                  "inventing problems."),
        AgentProfile(
            name="Scribe",
            speciality="notes, documentation and keeping the whiteboard tidy",
            brief="Write things down so they survive the conversation. "
                  "Short, plain, and where someone will find it."),
    ]


DEFAULT_TEAM = _default_team()
