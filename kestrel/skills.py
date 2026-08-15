"""Agent Skills loader.

Implements the open agentskills.io layout, which is what Hermes, Claude Code,
Codex CLI and the rest all read:

    skill-name/
      SKILL.md      required: YAML frontmatter (name, description) + markdown body
      scripts/      optional: executables the body can invoke
      references/   optional: docs pulled in only when needed
      assets/       optional: templates and data

Progressive disclosure is the whole point. Level 1 is name + description in the
system prompt (~25 tokens each, and trimmed harder on small windows). Level 2 is
the body, fetched by the skill_open tool. Level 3 is bundled files, reached with
the ordinary file and shell tools.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is a soft dependency
    yaml = None

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
NAME_OK = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclass
class Skill:
    name: str
    description: str
    path: Path                       # the SKILL.md itself
    root: Path                       # the skill folder
    source: str = ""                 # which search dir it came from
    license: str = ""
    metadata: dict = field(default_factory=dict)
    valid: bool = True
    problem: str = ""

    def body(self) -> str:
        try:
            text = self.path.read_text("utf-8", errors="replace")
        except OSError as e:
            return f"(could not read {self.path}: {e})"
        return FRONTMATTER.sub("", text, count=1).strip()

    def resources(self) -> list[str]:
        out: list[str] = []
        for sub in ("scripts", "references", "assets"):
            d = self.root / sub
            if d.is_dir():
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        out.append(str(f.relative_to(self.root)))
        return out

    def short(self, chars: int = 110) -> str:
        d = " ".join(self.description.split())
        return d if len(d) <= chars else d[: chars - 1].rstrip() + "…"


def _parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER.match(text)
    if not m:
        return {}
    block = m.group(1)
    if yaml is not None:
        try:
            data = yaml.safe_load(block)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return _mini_yaml(block)


def _mini_yaml(block: str) -> dict:
    """Enough YAML for frontmatter when PyYAML isn't installed.

    Handles `key: value`, folded/literal scalars (`>-`, `|`) and skips nested
    mappings, which is all the spec requires for name and description.
    """
    out: dict[str, str] = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#") or line.startswith((" ", "\t")):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val in (">", ">-", "|", "|-", ">+", "|+"):
            buf: list[str] = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                buf.append(lines[i].strip())
                i += 1
            joiner = "\n" if val.startswith("|") else " "
            out[key] = joiner.join(b for b in buf if b).strip()
        elif val:
            out[key] = val.strip("'\"")
    return out


def load_skill(skill_md: Path, source: str = "") -> Skill | None:
    try:
        text = skill_md.read_text("utf-8", errors="replace")
    except OSError:
        return None
    fm = _parse_frontmatter(text)
    root = skill_md.parent
    name = str(fm.get("name") or root.name).strip()
    desc = " ".join(str(fm.get("description") or "").split())
    sk = Skill(
        name=name, description=desc, path=skill_md, root=root, source=source,
        license=str(fm.get("license") or ""),
        metadata=fm.get("metadata") if isinstance(fm.get("metadata"), dict) else {},
    )
    has_frontmatter = bool(FRONTMATTER.match(text))
    body = FRONTMATTER.sub("", text, count=1).strip()
    if not desc:
        # Third-party skills with sloppy frontmatter should still work, so the
        # first paragraph stands in — but only when the file otherwise looks
        # like a skill. Without that check any stray .md dropped in the folder
        # would be advertised to the model as a capability.
        first = next((q.strip() for q in body.split("\n\n")
                      if q.strip() and not q.startswith("#")), "")
        sk.description = " ".join(first.split())[:400]
        sk.problem = "no description in frontmatter"
        if not re.search(r"^#{1,3}\s+\S", body, re.MULTILINE):
            sk.valid = False
            sk.problem = ("no YAML frontmatter and no heading — add a "
                          "name and description block at the top")
    if not NAME_OK.match(name):
        sk.problem = (sk.problem + "; " if sk.problem else "") + "name is not spec-compliant"
    if not sk.description or len(sk.description) < 15:
        sk.valid = False
        sk.problem = "description missing or too short to be useful"
    if has_frontmatter and not fm.get("description"):
        sk.problem = "frontmatter has no description field"
    return sk


def discover_detailed(dirs, max_depth: int = 4):
    """Every SKILL.md found, split into usable and rejected.

    Rejections are surfaced rather than swallowed: a file dropped into the
    folder that does not appear is otherwise indistinguishable from one that was
    never noticed.
    """
    found: dict[str, Skill] = {}
    rejected: list[tuple[str, str]] = []
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for skill_md in _walk(root, max_depth):
            sk = load_skill(skill_md, source=str(root))
            if sk is None:
                rejected.append((str(skill_md), "unreadable"))
            elif not sk.valid:
                rejected.append((str(skill_md), sk.problem or "malformed"))
            elif sk.name not in found:
                found[sk.name] = sk
    return sorted(found.values(), key=lambda s: s.name), rejected


def discover(dirs: list[str] | list[Path], max_depth: int = 4) -> list[Skill]:
    """Find every SKILL.md under the given roots. Later duplicates lose."""
    found: dict[str, Skill] = {}
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for skill_md in _walk(root, max_depth):
            sk = load_skill(skill_md, source=str(root))
            if sk and sk.valid and sk.name not in found:
                found[sk.name] = sk
    return sorted(found.values(), key=lambda s: s.name)


def _walk(root: Path, max_depth: int):
    base = len(root.parts)
    try:
        entries = list(root.rglob("SKILL.md"))
    except OSError:
        return
    for p in entries:
        if len(p.parts) - base <= max_depth + 1 and p.is_file():
            yield p


def index_lines(skills: list[Skill], limit: int, chars: int) -> list[str]:
    """Level-1 listing, sized to the budget."""
    out = []
    for sk in skills[:limit]:
        out.append(f"- {sk.name}: {sk.short(chars)}")
    if len(skills) > limit:
        out.append(f"- (+{len(skills) - limit} more; use skill_find to search by keyword)")
    return out


def search(skills: list[Skill], query: str, limit: int = 10) -> list[Skill]:
    q = [w for w in re.split(r"\W+", query.lower()) if len(w) > 2]
    if not q:
        return skills[:limit]
    scored = []
    for sk in skills:
        hay = (sk.name + " " + sk.description).lower()
        score = sum(hay.count(w) for w in q) + 3 * sum(w in sk.name.lower() for w in q)
        if score:
            scored.append((score, sk))
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored[:limit]]


TEMPLATE = """---
name: {name}
description: {description}
---

# {title}

State what this skill does and when it applies. The description above is what
the model sees in every prompt; this body is loaded only when the skill is
opened, so it can be as long as it needs to be.

## When to use this

- the situation that should trigger it
- another situation
- when NOT to use it, if that is easy to get wrong

## Steps

1. First thing to do.
2. Second thing, with the exact command or path where that matters.
3. How to tell it worked.

## Notes

Anything the model would otherwise get wrong: a flag that is easy to forget, an
error message and what it really means, a value that must match somewhere else.
"""


def create_skill(directory, name: str, description: str = "") -> Path:
    """Write a new skill folder from the template.

    The frontmatter is filled in because that is the part that must be right —
    a skill without a usable description is invisible to the model, and it is
    the most common thing to leave out when starting from a blank file.
    """
    folder = Path(directory).expanduser() / (slug_name(name) or "new-skill")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "SKILL.md"
    if target.exists():
        return target
    title = name.replace("-", " ").strip().title()
    body = TEMPLATE.format(
        name=slug_name(name) or "new-skill",
        description=(description.strip() or
                     "Describe what this does and when to use it. Mention the "
                     "words someone would use when they need it."),
        title=title or "New skill")
    target.write_text(body, "utf-8")
    (folder / "scripts").mkdir(exist_ok=True)
    (folder / "references").mkdir(exist_ok=True)
    return target


def slug_name(name: str) -> str:
    """Skill names are lowercase and hyphenated by the specification."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    return cleaned[:60]
