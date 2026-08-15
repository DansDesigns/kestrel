"""The personality layer.

A SOUL.md-style file re-sent on every turn is a tax you pay forever: two
thousand tokens of backstory, charged again at every step, on a window that may
only be four thousand tokens wide.

Kestrel separates the two things such a file conflates. *Voice* — how the agent
sounds — is small and belongs in every prompt. *Lore* — history, detail,
worldbuilding — is large, rarely load-bearing, and belongs behind a tool call.

So a persona is compiled once into tiers. The window decides which tier ships:
one line at nano, a compact block at standard. The full text is never resent;
the model pulls it with the `persona` tool if it actually needs it, exactly like
a skill. Swapping personality is then a cheap operation rather than a rewrite of
the prompt budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .skills import _parse_frontmatter, FRONTMATTER


@dataclass
class Persona:
    name: str = ""
    voice: str = ""                       # how it sounds, one line
    traits: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    full_text: str = ""                   # everything, never auto-injected
    path: Path | None = None

    # -- tiered compilation ---------------------------------------------------
    def compile(self, level: int, budget_chars: int = 0) -> str:
        """Render at a verbosity tier. 0 is a single line."""
        if not self.any_content():
            return ""
        name = self.name or "Kestrel"

        if level <= 0:
            bits = [self.voice] if self.voice else []
            if not bits and self.traits:
                bits = [", ".join(self.traits[:3])]
            line = f"You are {name}. " + (bits[0] if bits else "")
            return _cap(line.strip(), budget_chars)

        parts = [f"You are {name}."]
        if self.voice:
            parts.append(self.voice)
        if level >= 2 and self.traits:
            parts.append("You are " + ", ".join(self.traits[:6]) + ".")
        keep = 2 if level == 1 else (5 if level == 2 else len(self.rules))
        for rule in self.rules[:keep]:
            parts.append("- " + rule)
        if level >= 2 and self.has_lore():
            parts.append("(Call the persona tool if you need your full background.)")
        return _cap("\n".join(parts), budget_chars)

    def any_content(self) -> bool:
        return bool(self.name or self.voice or self.traits or self.rules or self.full_text)

    def has_lore(self) -> bool:
        """Is there materially more in the file than the distilled fields?

        Measured against the raw fields rather than a compiled tier, because
        compile() consults this and the two would chase each other.
        """
        injected = len(self.voice) + sum(len(t) for t in self.traits) \
            + sum(len(r) for r in self.rules) + len(self.name)
        return len(self.full_text) > max(400, injected + 200)

    def summary(self) -> str:
        done = [f"name    {self.name or '(unnamed)'}"]
        if self.voice:
            done.append(f"voice   {self.voice}")
        if self.traits:
            done.append("traits  " + ", ".join(self.traits))
        if self.rules:
            done.append(f"rules   {len(self.rules)}")
        done.append(f"full    {len(self.full_text)} chars"
                    + (" (available via the persona tool)" if self.has_lore() else ""))
        return "\n".join(done)


def _cap(text: str, budget_chars: int) -> str:
    if budget_chars and len(text) > budget_chars:
        return text[: max(0, budget_chars - 1)].rstrip() + "…"
    return text


def parse(text: str, path: Path | None = None) -> Persona:
    """Read a persona file.

    Structured frontmatter is used when present. Otherwise the file is distilled
    heuristically, which is what lets an existing SOUL.md be dropped in and used
    without rewriting it.
    """
    fm = _parse_frontmatter(text)
    body = FRONTMATTER.sub("", text, count=1).strip()
    p = Persona(full_text=body, path=path)

    p.name = str(fm.get("name") or "").strip()
    p.voice = " ".join(str(fm.get("voice") or "").split())
    p.traits = _as_list(fm.get("traits"))
    p.rules = _as_list(fm.get("rules"))

    if not p.name:
        m = re.search(r"^#\s*(.+)$", body, re.MULTILINE)
        p.name = m.group(1).strip()[:60] if m else (path.stem if path else "")
    if not p.voice:
        p.voice = _first_sentences(body, 2)
    if not p.rules:
        p.rules = _bullets(body)
    return p


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [" ".join(str(v).split()) for v in value if str(v).strip()][:10]
    if isinstance(value, str):
        parts = re.split(r"[,\n]", value)
        return [" ".join(x.split()) for x in parts if x.strip()][:10]
    return []


def _first_sentences(body: str, n: int) -> str:
    for para in body.split("\n\n"):
        para = para.strip()
        if not para or para.startswith(("#", "-", "*", ">")):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(para.split()))
        return " ".join(sentences[:n])[:300]
    return ""


def _bullets(body: str) -> list[str]:
    out = []
    for line in body.splitlines():
        m = re.match(r"^\s*[-*+]\s+(.{4,200})$", line)
        if m:
            out.append(" ".join(m.group(1).split()))
        if len(out) >= 8:
            break
    return out


def discover(dirs) -> list[Persona]:
    found: dict[str, Persona] = {}
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for f in sorted(root.glob("*.md")):
            try:
                p = parse(f.read_text("utf-8", errors="replace"), f)
            except OSError:
                continue
            key = (p.name or f.stem).lower()
            if key not in found:
                found[key] = p
    return sorted(found.values(), key=lambda p: (p.name or "").lower())


def load_file(path: str | Path) -> Persona | None:
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        return parse(p.read_text("utf-8", errors="replace"), p)
    except OSError:
        return None


def default_persona_dirs(config_dir: Path, workspace: str) -> list[str]:
    # The bundled folder is found relative to the package, not the working
    # directory: launched from a menu shortcut, the two are not the same.
    shipped = Path(__file__).resolve().parent.parent / "personas"
    return [
        str(config_dir / "personas"),
        str(Path(workspace).expanduser() / ".kestrel" / "personas"),
        str(shipped),
        str(Path.cwd() / "personas"),
    ]


def register_tool(reg, provider) -> None:
    """Level-2 access: the full persona text, on request only."""
    from .tools import DANGER_SAFE, Tool, ToolResult

    def persona() -> ToolResult:
        p = provider()
        if p is None or not p.full_text:
            return ToolResult("No persona is loaded beyond what is already in your "
                              "instructions.")
        return ToolResult(p.full_text, full=p.full_text)

    reg.add(Tool("persona", "Read your full character background.", [],
                 persona, DANGER_SAFE,
                 detail="Only when the detail actually matters; your voice is "
                        "already described above."))
