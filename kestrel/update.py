"""Update checking.

A single `version.txt` in the repository, fetched and compared with the one
shipped here. Deliberately not an auto-updater: this is a local-first tool that
people modify, and replacing their files from the network without asking would
be a poor trade for saving them a `git pull`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = "dansdesigns/kestrel"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/version.txt"
BRANCHES = ("main", "master")
RELEASES = f"https://github.com/{REPO}"

VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def local_version() -> str:
    """The installed version, from the package and nowhere else.

    Kestrel does not ship a version.txt of its own: the file this compares
    against is yours, in your repository, and writing one here would overwrite
    the number you set every time the package is updated.
    """
    from . import __version__
    return __version__


def parse(version: str) -> tuple[int, ...]:
    match = VERSION_RE.search(str(version or ""))
    return tuple(int(p) for p in match.groups()) if match else (0, 0, 0)


def compare(local: str, remote: str) -> int:
    """-1 behind, 0 level, 1 ahead."""
    a, b = parse(local), parse(remote)
    return (a > b) - (a < b)


@dataclass
class Result:
    local: str = ""
    remote: str = ""
    available: bool = False
    error: str = ""
    url: str = RELEASES

    def summary(self) -> str:
        if self.error:
            return f"Could not check: {self.error}"
        if not self.remote:
            return f"Installed {self.local}. No version published yet."
        if self.available:
            return f"Update available: {self.remote} (installed {self.local})"
        if compare(self.local, self.remote) > 0:
            return f"Installed {self.local}, ahead of published {self.remote}"
        return f"Up to date ({self.local})"


def check(timeout: float = 12.0) -> Result:
    """Fetch the published version and compare.

    Both default branch names are tried, because a repository created today and
    one created a decade ago disagree about which is which, and a 404 from the
    wrong guess is not an error worth reporting.
    """
    import requests

    result = Result(local=local_version())
    last_error = ""
    for branch in BRANCHES:
        try:
            response = requests.get(RAW.format(repo=REPO, branch=branch),
                                    timeout=timeout)
        except Exception as e:
            last_error = str(e)
            continue
        if response.status_code == 404:
            last_error = "version.txt not found"
            continue
        if not response.ok:
            last_error = f"HTTP {response.status_code}"
            continue
        text = (response.text or "").strip().splitlines()
        if not text:
            last_error = "version.txt is empty"
            continue
        result.remote = text[0].strip()
        result.available = compare(result.local, result.remote) < 0
        return result
    result.error = last_error or "no response"
    return result
