"""Fetching skills from a GitHub repository.

Three shapes turn up in the wild and all three have to work:

* one skill, with SKILL.md at the top of the repository
* many skills, each in its own folder somewhere below
* neither — a curated list of links to other repositories, which looks like a
  skill collection until you download it

The third is the one worth handling carefully. Saying "no skills found" is
true and useless; saying "this is an index of other repositories" tells
somebody what to do next.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

BRANCHES = ("main", "master")
AGENT = {"User-Agent": "Kestrel"}
# Folders that are packaging rather than content. A skill inside one of these
# is a fixture for that repository's own tests, not something to install.
SKIP = {".github", "node_modules", "tests", "test", "fixtures", "examples",
        "__pycache__", "docs", "site", "evals"}


@dataclass
class Found:
    """One skill inside a downloaded archive."""
    name: str
    description: str
    member: str                 # the SKILL.md path inside the zip
    root: str                   # its folder inside the zip
    files: int = 0
    requires: list = field(default_factory=list)


def parse_repo(text: str) -> tuple[str, str]:
    """owner and repo, from a URL or an owner/repo pair."""
    cleaned = (text or "").strip().rstrip("/")
    if not cleaned:
        return "", ""
    # With or without a scheme: people paste "github.com/owner/repo" as often
    # as the full address, and dropping only the scheme leaves the host as the
    # owner.
    cleaned = re.sub(r"^(https?://)?(www\.)?github\.com/", "", cleaned,
                     flags=re.I)
    cleaned = re.sub(r"\.git$", "", cleaned)
    if "://" in cleaned:
        # Some other host. Better to say so than to take the first two path
        # pieces and fail confusingly against GitHub.
        return "", ""
    parts = [p for p in cleaned.split("/") if p]
    if len(parts) >= 2 and all(re.fullmatch(r"[\w.-]+", p) for p in parts[:2]):
        return parts[0], parts[1]
    return "", ""


def fetch(owner: str, repo: str, timeout: float = 120.0) -> bytes:
    """The repository as a zip, trying the usual branch names.

    Tried rather than looked up: asking the API which branch is default costs a
    request against a rate limit that is shared by everyone behind one address,
    and two attempts cover almost every repository.
    """
    last = ""
    for branch in BRANCHES:
        url = (f"https://codeload.github.com/{owner}/{repo}"
               f"/zip/refs/heads/{branch}")
        try:
            request = urllib.request.Request(url, headers=AGENT)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            last = f"{e.code} {e.reason}"
            continue
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach GitHub: {e.reason}") from None
    raise RuntimeError(
        f"github.com/{owner}/{repo} could not be downloaded ({last}). "
        "Check the address, and that the repository is public.")


def _frontmatter(text: str) -> dict:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not match:
        return {}
    fields: dict = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.startswith((" ", "\t", "#")):
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip("\"'")
    return fields


def survey(archive: bytes) -> list[Found]:
    """Every skill in the archive, wherever it is."""
    found: list[Found] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        for member in names:
            if not member.endswith("/SKILL.md"):
                continue
            parts = member.split("/")
            if any(p in SKIP for p in parts):
                continue
            root = member[: -len("SKILL.md")]
            try:
                text = bundle.read(member).decode("utf-8", "replace")
            except KeyError:
                continue
            fields = _frontmatter(text)
            # The folder name when the frontmatter has none, and for a skill at
            # the top of a repository that folder is the repository itself.
            default = parts[-2] if len(parts) >= 2 else "skill"
            if len(parts) == 2:
                default = re.sub(r"-(main|master)$", "", parts[0])
            found.append(Found(
                name=str(fields.get("name") or default).strip(),
                description=" ".join(
                    str(fields.get("description") or "").split())[:300],
                member=member,
                root=root,
                files=sum(1 for n in names if n.startswith(root)),
                requires=[p.strip() for p in
                          str(fields.get("requires") or "").replace(";", ",")
                          .split(",") if p.strip()],
            ))
    found.sort(key=lambda f: f.name.lower())
    return found


def install(archive: bytes, chosen: list[Found], into: Path,
            on_step=None) -> list[str]:
    """Write the chosen skills into the skills folder, one folder each."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for skill in chosen:
            folder = into / _safe(skill.name)
            if folder.exists():
                # Replaced rather than merged: a file left from an older
                # version of a skill is worse than either version of it.
                shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
            for member in bundle.namelist():
                if not member.startswith(skill.root) or member.endswith("/"):
                    continue
                relative = member[len(skill.root):]
                if not relative:
                    continue
                destination = (folder / relative).resolve()
                # Never outside the folder, whatever the archive claims.
                if not str(destination).startswith(str(folder.resolve())):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, \
                        open(destination, "wb") as out:
                    shutil.copyfileobj(source, out)
            written.append(skill.name)
            if on_step:
                on_step(skill.name)
    return written


def index_hint(archive: bytes) -> str:
    """When a repository holds no skills, say what it does hold.

    A curated list is the common case: it reads as a skill collection and
    contains only links. Pointing that out is more use than reporting nothing
    found.
    """
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        readme = next((n for n in bundle.namelist()
                       if n.lower().endswith("readme.md")), "")
        if not readme:
            return ""
        text = bundle.read(readme).decode("utf-8", "replace")
    links = set(re.findall(r"https?://github\.com/([\w.-]+/[\w.-]+)", text))
    links = {l for l in links if not l.lower().endswith((".png", ".svg"))}
    if len(links) < 5:
        return ""
    return (f"This repository holds no skills of its own — it is an index "
            f"linking to about {len(links)} other repositories. Open one of "
            "those and paste its address instead.")


def _safe(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "skill")).strip("-.")
    return cleaned[:60] or "skill"
