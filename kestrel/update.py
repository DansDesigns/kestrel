"""Update checking.

A single `version.txt` in the repository, fetched and compared with the one
shipped here. Deliberately not an auto-updater: this is a local-first tool that
people modify, and replacing their files from the network without asking would
be a poor trade for saving them a `git pull`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO = "dansdesigns/kestrel"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/version.txt"
BRANCHES = ("main", "master")
RELEASES = f"https://github.com/{REPO}"

VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def local_version() -> str:
    """The installed version.

    Read from version.txt when there is one — that file belongs to whoever
    maintains the checkout, and it is the number they set. Kestrel never
    creates or overwrites it; it only reports what is there, falling back to the
    package constant when the file is absent.
    """
    candidate = project_root() / "version.txt"
    try:
        first = candidate.read_text("utf-8").strip().splitlines()[0].strip()
        if first:
            return first
    except (OSError, IndexError):
        pass
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
    """Download to a temporary folder, then replace this installation.

    Nothing is written into the program folder until a complete, verified copy
    exists somewhere else — an update that fails halfway through should leave
    the working installation alone rather than a half-replaced one. The backup
    goes to the system temporary directory rather than beside the program, so
    updating does not leave a litter of folders next to it.
    """
    import io
    import shutil
    import tempfile
    import zipfile

    import requests

    root = project_root()
    staging = Path(tempfile.mkdtemp(prefix="kestrel-update-"))
    try:
        payload = None
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
            payload = response.content
            break
        if payload is None:
            return False, "no published branch was found"

        say(f"unpacking {len(payload) // 1024:,} KB")
        try:
            zipfile.ZipFile(io.BytesIO(payload)).extractall(staging)
        except zipfile.BadZipFile:
            return False, "the download was not a valid archive"

        # Find the copy of Kestrel inside, wherever the archive chose to put it.
        source = None
        for candidate in [staging, *staging.iterdir()]:
            if (candidate / "kestrel" / "__init__.py").is_file():
                source = candidate
                break
        if source is None:
            return False, "the archive does not look like Kestrel"

        backup = Path(tempfile.mkdtemp(prefix="kestrel-backup-"))
        say(f"backing up to {backup}")
        try:
            shutil.copytree(root, backup / "kestrel-previous",
                            ignore=shutil.ignore_patterns(*KEEP),
                            dirs_exist_ok=True)
        except OSError as e:
            return False, f"could not back up the current copy: {e}"

        say("replacing files")
        written = 0
        for item in source.rglob("*"):
            relative = item.relative_to(source)
            if relative.parts and relative.parts[0] in KEEP:
                continue
            target = root / relative
            try:
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
                    written += 1
            except OSError as e:
                return False, (f"stopped after {written} files: {e}. Nothing "
                               f"else was changed; the previous copy is at "
                               f"{backup}.")
        # A module dropped upstream has to go, or it stays importable and the
        # old code keeps running. Only inside the package directory, which is
        # entirely ours — anything the user put elsewhere is left alone.
        removed = 0
        package = root / "kestrel"
        if package.is_dir() and (source / "kestrel").is_dir():
            for existing in package.rglob("*.py"):
                relative = existing.relative_to(root)
                if not (source / relative).exists():
                    try:
                        existing.unlink()
                        removed += 1
                    except OSError:
                        pass
        say(f"{written} files written"
            + (f", {removed} removed" if removed else ""))
        return True, (f"Updated {written} files to {local_version()}. "
                      f"The previous copy is at {backup} and can be deleted "
                      "once you are happy.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def restart() -> bool:
    """Relaunch Kestrel and leave. Returns False if it could not be done."""
    import subprocess
    import sys

    root = project_root()
    launcher = root / ("run.bat" if os.name == "nt" else "run.sh")
    try:
        if launcher.exists():
            command = ([str(launcher)] if os.name == "nt"
                       else ["bash", str(launcher)])
        else:
            command = [sys.executable, "-m", "kestrel"]
        subprocess.Popen(command, cwd=str(root), close_fds=True)
        return True
    except Exception:
        return False
