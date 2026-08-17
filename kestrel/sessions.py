"""Conversations and projects.

Two related pieces of state that outlive a single exchange:

  Sessions   a saved transcript, so a conversation can be left and returned to.
             Stored per project, because a conversation belongs to the work it
             was about.
  Projects   a folder under the workspace root with its own files, memory scope,
             checklist and sessions. Switching project switches all four
             together, which is the point: an agent that remembers one project's
             decisions while working on another is worse than one that remembers
             nothing.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

SAFE_NAME = re.compile(r"[^A-Za-z0-9 ._-]")


def slug(name: str) -> str:
    cleaned = SAFE_NAME.sub("", str(name or "")).strip().strip(".")
    cleaned = re.sub(r"\s+", "-", cleaned)
    return (cleaned or "untitled")[:60]


# ------------------------------------------------------------- projects ----
# "2026-08-16 telemetry-rewrite" — the date first so a plain alphabetical
# listing in any file manager is also chronological.
PROJECT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ _-]+(.*)$")
PLAN_FILE = "PLAN.md"


@dataclass
class Project:
    name: str
    path: Path

    @property
    def created(self) -> str:
        """The date in the folder name, or the folder's own date."""
        match = PROJECT_RE.match(self.name)
        if match:
            return match.group(1)
        try:
            return time.strftime("%Y-%m-%d", time.localtime(self.path.stat().st_ctime))
        except OSError:
            return ""

    @property
    def display(self) -> str:
        match = PROJECT_RE.match(self.name)
        return (match.group(2) or self.name) if match else self.name

    @property
    def plan_path(self) -> Path:
        return self.path / PLAN_FILE

    def plan_summary(self) -> str:
        """Read the progress line out of the plan file, without parsing it all.

        The file is written for people; this only needs the count, and reading
        the first few lines is enough for that.
        """
        try:
            with open(self.plan_path, "r", encoding="utf-8") as handle:
                for _ in range(6):
                    line = handle.readline()
                    if not line:
                        break
                    if " done" in line and "·" in line:
                        return line.split("·")[-1].strip()
        except OSError:
            pass
        return ""

    def session_dir(self) -> Path:
        return self.path / ".kestrel" / "sessions"

    def stats(self, cap: int = 250, deadline: float = 0.35) -> str:
        """A rough size, bounded in both count and time.

        Counting recursively without a limit is fine for a project folder and
        catastrophic for a home directory or a drive root — it walks the entire
        filesystem before the window can open. The count stops at a cap and a
        deadline, and says so rather than pretending to be exact.
        """
        files = 0
        truncated = False
        started = time.time()
        try:
            for entry in self.path.rglob("*"):
                if any(part.startswith(".") for part in entry.parts[-2:]):
                    continue
                if entry.is_file():
                    files += 1
                if files >= cap or time.time() - started > deadline:
                    truncated = True
                    break
        except (OSError, PermissionError):
            pass
        count = f"{files}+" if truncated else str(files)
        return f"{count} file(s) · {len(list_sessions(self))} conversation(s)"


def projects_root(workspace_root: str | Path) -> Path:
    p = Path(workspace_root).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


# Directories that are somebody's whole computer rather than a project.
SYSTEM_DIRS = {"windows", "program files", "program files (x86)", "programdata",
               "system32", "appdata", "$recycle.bin", "node_modules",
               "library", "applications", "onedrive"}


def risky_root(path: str | Path) -> str:
    """Why this is a poor choice of workspace root, or an empty string.

    The workspace is where the agent's file tools are confined and where project
    folders are created. A home directory or a drive root is both slow to scan
    and far too broad a sandbox.
    """
    p = Path(path).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        return ""
    if resolved == Path.home():
        return ("this is your home directory — every folder in it would be "
                "treated as a project, and the agent's file tools would be "
                "scoped to all of it")
    if resolved.parent == resolved:
        return "this is a drive root"
    if resolved.name.lower() in SYSTEM_DIRS:
        return "this is a system directory"
    try:
        children = 0
        for child in resolved.iterdir():
            children += 1
            if children > 60:
                return (f"this contains more than 60 folders — projects are "
                        "easier to find in a directory of their own")
    except (OSError, PermissionError):
        return "this directory cannot be read"
    return ""


def list_projects(workspace_root: str | Path) -> list[Project]:
    root = projects_root(workspace_root)
    out = []
    try:
        for child in root.iterdir():
            if (child.is_dir() and not child.name.startswith(".")
                    and child.name.lower() not in SYSTEM_DIRS):
                out.append(Project(name=child.name, path=child))
    except (OSError, PermissionError):
        pass
    # Newest first: the project being worked on is nearly always a recent one.
    out.sort(key=lambda p: (p.created, p.name), reverse=True)
    return out


def create_project(workspace_root: str | Path, name: str,
                   when: str = "") -> Project:
    """Make a dated project folder.

    The date leads the name so that projects sort chronologically wherever they
    are listed — in Kestrel, in Explorer, in a terminal — without anything
    needing to read metadata.
    """
    root = projects_root(workspace_root)
    stamp = when or time.strftime("%Y-%m-%d")
    folder = root / f"{stamp} {slug(name)}"
    suffix = 2
    while folder.exists() and any(folder.iterdir()):
        folder = root / f"{stamp} {slug(name)}-{suffix}"
        suffix += 1
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".kestrel").mkdir(exist_ok=True)
    readme = folder / "README.md"
    if not readme.exists():
        readme.write_text(f"# {name}\n\nCreated {stamp}.\n", "utf-8")
    return Project(name=folder.name, path=folder)


# ------------------------------------------------------------- sessions ----
@dataclass
class Session:
    id: str
    title: str = ""
    created: float = 0.0
    updated: float = 0.0
    messages: list[dict] = field(default_factory=list)
    digest: str = ""
    model: str = ""
    plan: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def turns(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "user")

    def when(self) -> str:
        stamp = self.updated or self.created
        if not stamp:
            return ""
        delta = time.time() - stamp
        if delta < 3600:
            return f"{int(delta // 60)}m ago"
        if delta < 86400:
            return f"{int(delta // 3600)}h ago"
        if delta < 7 * 86400:
            return f"{int(delta // 86400)}d ago"
        return time.strftime("%d %b", time.localtime(stamp))

    def summary(self) -> str:
        bits = [f"{self.turns} turn(s)", self.when()]
        steps = self.plan.get("items") if isinstance(self.plan, dict) else None
        if steps:
            done = sum(1 for i in steps if i.get("status") == "done")
            bits.append(f"plan {done}/{len(steps)}")
        if self.model:
            bits.append(self.model)
        return " · ".join(b for b in bits if b)


def new_session() -> Session:
    # A time-derived suffix is not enough: two sessions created in the same
    # millisecond share a filename and one silently overwrites the other.
    now = time.time()
    return Session(id=f"{int(now)}-{uuid.uuid4().hex[:8]}", created=now, updated=now)


def title_from(messages: list[dict]) -> str:
    """First thing the user actually said, trimmed. Better than a timestamp and
    cheaper than asking a model for a title."""
    for m in messages:
        if m.get("role") == "user":
            text = " ".join(str(m.get("content") or "").split())
            if text:
                return text[:70] + ("…" if len(text) > 70 else "")
    return "Empty conversation"


def save_session(session: Session, project: Project) -> Path:
    directory = project.session_dir()
    directory.mkdir(parents=True, exist_ok=True)
    session.updated = time.time()
    if not session.title:
        session.title = title_from(session.messages)
    path = directory / f"{session.id}.json"
    payload = {
        "id": session.id, "title": session.title, "created": session.created,
        "updated": session.updated, "messages": session.messages,
        "digest": session.digest, "model": session.model,
        "plan": session.plan,
    }
    path.write_text(json.dumps(payload, indent=1), "utf-8")
    session.path = path
    return path


def load_session(path: str | Path) -> Session | None:
    p = Path(path)
    try:
        raw = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    session = Session(
        id=str(raw.get("id") or p.stem), title=str(raw.get("title") or ""),
        created=float(raw.get("created") or 0), updated=float(raw.get("updated") or 0),
        messages=list(raw.get("messages") or []), digest=str(raw.get("digest") or ""),
        model=str(raw.get("model") or ""),
        plan=raw.get("plan") if isinstance(raw.get("plan"), dict) else {}, path=p)
    return session


def list_sessions(project: Project, limit: int = 200) -> list[Session]:
    directory = project.session_dir()
    if not directory.is_dir():
        return []
    out = []
    for f in directory.glob("*.json"):
        session = load_session(f)
        if session is not None and session.messages:
            out.append(session)
    out.sort(key=lambda s: -(s.updated or s.created))
    return out[:limit]


def delete_session(session: Session) -> bool:
    if session.path and session.path.exists():
        try:
            session.path.unlink()
            return True
        except OSError:
            return False
    return False
