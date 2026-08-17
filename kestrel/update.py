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


ZIP_URL = "https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
KEEP = {".git", ".venv", "venv", "__pycache__"}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def is_git_checkout() -> bool:
    return (project_root() / ".git").is_dir()


def apply(progress=None) -> tuple[bool, str]:
    """Update this installation in place. Returns (updated, message).

    A git checkout is pulled, because that is what the person expects of one
    and it keeps their history. Anything else is replaced from the published
    archive, file by file, with a backup of what was there first.

    Settings, memory, conversations and projects live outside the program
    folder, so none of this touches them.
    """
    say = progress or (lambda line: None)
    if is_git_checkout():
        return _git_pull(say)
    return _replace_from_archive(say)


def _git_pull(say) -> tuple[bool, str]:
    import subprocess
    say("$ git pull --ff-only")
    try:
        result = subprocess.run(["git", "pull", "--ff-only"], cwd=str(project_root()),
                                capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"could not run git: {e}"
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[-12:]:
        say(line)
    if result.returncode != 0:
        return False, ("git could not update this checkout — most often local "
                       "changes are in the way. " + output.splitlines()[-1][:120]
                       if output else "git failed")
    if "Already up to date" in output:
        return True, "Already up to date."
    return True, "Updated. Restart Kestrel to run the new version."


def _replace_from_archive(say) -> tuple[bool, str]:
    import io
    import shutil
    import zipfile

    import requests

    root = project_root()
    for branch in BRANCHES:
        say(f"downloading {branch}…")
        try:
            response = requests.get(ZIP_URL.format(repo=REPO, branch=branch),
                                    timeout=180)
        except Exception as e:
            return False, f"download failed: {e}"
        if response.status_code == 404:
            continue
        if not response.ok:
            return False, f"download failed: HTTP {response.status_code}"

        try:
            archive = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            return False, "the download was not a valid archive"

        names = archive.namelist()
        if not names:
            return False, "the archive was empty"
        prefix = names[0].split("/")[0] + "/"

        backup = root.parent / f"{root.name}.backup"
        say(f"backing up to {backup}")
        try:
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            shutil.copytree(root, backup,
                            ignore=shutil.ignore_patterns(*KEEP))
        except OSError as e:
            return False, f"could not back up the current copy: {e}"

        say("unpacking")
        written = 0
        try:
            for name in names:
                if name.endswith("/") or not name.startswith(prefix):
                    continue
                relative = name[len(prefix):]
                if not relative or relative.split("/")[0] in KEEP:
                    continue
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                written += 1
        except OSError as e:
            return False, (f"unpacking failed after {written} files: {e}. "
                           f"The previous copy is at {backup}.")
        say(f"{written} files written")
        return True, (f"Updated {written} files. Restart Kestrel to run the new "
                      f"version. The previous copy is kept at {backup}.")
    return False, "no published branch was found"
