"""Menu shortcuts.

    python -m kestrel.shortcut            create it
    python -m kestrel.shortcut --remove   take it away

An application launched from a terminal by path is one people forget they have.
This puts it where the rest of their software is: the Start Menu on Windows, the
applications menu on Linux, and /Applications on macOS.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def launcher() -> Path:
    """The script a menu entry should run, preferring the one that activates the
    virtual environment over the bare interpreter."""
    root = project_root()
    name = "run.bat" if os.name == "nt" else "run.sh"
    candidate = root / name
    return candidate if candidate.exists() else root


def icon_path(extension: str) -> Path | None:
    candidate = project_root() / "assets" / f"kestrel.{extension}"
    return candidate if candidate.exists() else None


# ------------------------------------------------------------------ linux --
def linux_target() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "applications" / "kestrel.desktop"


def install_linux() -> Path:
    target = linux_target()
    target.parent.mkdir(parents=True, exist_ok=True)
    run = launcher()
    icon = icon_path("png")
    entry = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=Kestrel",
        "GenericName=AI agent",
        "Comment=Agentic harness for llama.cpp",
        # bash, not the script directly: a checkout copied or unzipped from an
        # archive frequently loses its executable bit.
        f"Exec=bash {shell_quote(run)}",
        f"Path={shell_quote(project_root())}",
        "Terminal=false",
        "Categories=Development;Utility;Science;",
        "Keywords=llama;llm;ai;agent;",
        "StartupWMClass=Kestrel",
    ]
    if icon:
        entry.insert(6, f"Icon={icon}")
    target.write_text("\n".join(entry) + "\n", "utf-8")
    target.chmod(0o755)
    _refresh_linux_menu(target.parent)
    return target


def _refresh_linux_menu(directory: Path) -> None:
    """Some desktops only notice a new entry after the database is rebuilt."""
    if not shutil_which("update-desktop-database"):
        return
    try:
        subprocess.run(["update-desktop-database", str(directory)],
                       capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        pass


def shutil_which(name: str) -> str:
    import shutil
    return shutil.which(name) or ""


def shell_quote(path) -> str:
    text = str(path)
    return f'"{text}"' if " " in text else text


# ---------------------------------------------------------------- windows --
def windows_target() -> Path:
    base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    return (Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
            / "Kestrel.lnk")


def install_windows() -> Path:
    target = windows_target()
    target.parent.mkdir(parents=True, exist_ok=True)
    run = launcher()
    icon = icon_path("ico")
    # WScript.Shell is present on every supported Windows and needs no
    # dependency, unlike pywin32.
    script = [
        "$s = (New-Object -COM WScript.Shell).CreateShortcut('%s')" % target,
        "$s.TargetPath = '%s'" % run,
        "$s.WorkingDirectory = '%s'" % project_root(),
        "$s.Description = 'Agentic harness for llama.cpp'",
        "$s.WindowStyle = 7",          # start minimised: run.bat opens a console
    ]
    if icon:
        script.append("$s.IconLocation = '%s'" % icon)
    script.append("$s.Save()")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", "; ".join(script)],
        capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip()[:300])
    return target


# ------------------------------------------------------------------ macos --
def macos_target() -> Path:
    return Path.home() / "Applications" / "Kestrel.command"


def install_macos() -> Path:
    target = macos_target()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f'#!/bin/bash\ncd {shell_quote(project_root())}\n'
                      f'exec bash {shell_quote(launcher())}\n', "utf-8")
    target.chmod(0o755)
    return target


# ------------------------------------------------------------------- entry --
def install() -> Path:
    system = platform.system()
    if os.name == "nt":
        return install_windows()
    if system == "Darwin":
        return install_macos()
    return install_linux()


def remove() -> list[str]:
    removed = []
    for target in (linux_target(), windows_target(), macos_target()):
        try:
            if target.exists():
                target.unlink()
                removed.append(str(target))
        except OSError:
            continue
    return removed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kestrel.shortcut", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remove", action="store_true", help="delete the shortcut")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.remove:
        gone = remove()
        if not args.quiet:
            print("\n".join(f"removed {g}" for g in gone) or "no shortcut found")
        return 0
    try:
        target = install()
    except Exception as e:
        # Never fatal: a missing menu entry is a cosmetic problem, and the
        # installer should not fail over one.
        if not args.quiet:
            print(f"could not create the shortcut: {e}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"shortcut created: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
