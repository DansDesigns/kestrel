"""Kestrel's installer.

One file. It downloads the latest Kestrel from GitHub, finds a Python on the
machine, installs the libraries into a virtual environment of Kestrel's own,
and makes the shortcuts.

Nothing is bundled: the installer stays small and always fetches the current
version, so it does not go stale between releases.

All of it runs on the standard library, because an installer that needs
something installed before it can install anything is not an installer.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP = "Kestrel"
REPO = "dansdesigns/kestrel"
BRANCH = "main"
SOURCE = f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}"
RELEASES = f"https://api.github.com/repos/{REPO}/releases/latest"
WINDOWS = sys.platform == "win32"
QUIET = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MIN_PYTHON = (3, 10)


# ---------------------------------------------------------------- python ---
def python_candidates() -> list[Path]:
    """Every Python on this machine that might do, best first."""
    found: list[Path] = []

    def remember(path: str | None) -> None:
        if not path:
            return
        resolved = Path(path)
        if resolved.is_file() and resolved not in found:
            found.append(resolved)

    if WINDOWS:
        # The py launcher knows about every installation, including ones that
        # were never added to PATH — which is most of them, since the installer
        # tickbox is off by default.
        try:
            listed = subprocess.run(["py", "-0p"], capture_output=True,
                                    text=True, timeout=20, creationflags=QUIET)
            for line in listed.stdout.splitlines():
                parts = line.split()
                if parts and parts[-1].lower().endswith("python.exe"):
                    remember(parts[-1])
        except Exception:
            pass
    for name in ("python3", "python"):
        remember(shutil.which(name))
    return found


def python_version(exe: Path) -> tuple[int, ...]:
    try:
        out = subprocess.run(
            [str(exe), "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=20, creationflags=QUIET)
        return tuple(int(p) for p in out.stdout.strip().split("."))
    except Exception:
        return (0,)


def from_store(exe: Path) -> bool:
    """Is this the Microsoft Store build?

    It lives under WindowsApps and carries an MSIX package identity that
    overrides what a process claims about itself — which is why a program run
    through it shows Python's name and icon on the taskbar no matter what it
    asks for. Usable, but the python.org build is preferred when both are here.
    """
    return "windowsapps" in str(exe).lower()


def usable_python() -> tuple[Path | None, str]:
    """The best Python here, or an account of why there is not one."""
    seen = []
    store: Path | None = None
    for exe in python_candidates():
        version = python_version(exe)
        if version >= MIN_PYTHON:
            if from_store(exe):
                store = store or exe
                continue                 # keep looking for a plain one
            return exe, f"Python {'.'.join(map(str, version))}"
        if version > (0,):
            seen.append(".".join(map(str, version)))
    if store is not None:
        return store, (f"Python {'.'.join(map(str, python_version(store)))} "
                       "(Microsoft Store)")
    if seen:
        return None, (f"Python {', '.join(seen)} found, but Kestrel needs "
                      f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.")
    return None, "No Python found on this machine."


# -------------------------------------------------------------------- ui ---
def _pip_subject(line: str) -> str:
    """The library pip is working on, from a line of its output."""
    for prefix in ("Collecting ", "Downloading ", "Building wheel for ",
                   "Installing collected packages: "):
        if line.startswith(prefix):
            rest = line[len(prefix):]
            # "PySide6-6.11.2-cp39-abi3-win_amd64.whl (72.4 MB)" and
            # "PySide6>=6.6" both want to read as PySide6.
            name = rest.split(",")[0].split()[0]
            for separator in ("==", ">=", "<=", "~=", ">", "<", "-"):
                if separator in name:
                    name = name.split(separator)[0]
            return name.strip()
    return ""


def default_target() -> Path:
    if WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Programs" / APP
    return Path.home() / ".local" / "share" / "kestrel"


class Installer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Install {APP}")
        self.resizable(False, False)
        self.working = False
        self.python: Path | None = None
        # Things that went wrong but did not stop the install, said at the end
        # rather than swallowed.
        self.problems: list[str] = []

        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=APP, font=("", 17, "bold")).pack(anchor="w")
        ttk.Label(frame, wraplength=440, justify="left", text=(
            "A local AI agent that runs on your own machine. This downloads the "
            "latest version, installs what it needs, and adds a shortcut.")
        ).pack(anchor="w", pady=(4, 14))

        ttk.Label(frame, text="Install to").pack(anchor="w")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(4, 0))
        self.target = tk.StringVar(value=str(default_target()))
        ttk.Entry(row, textvariable=self.target, width=46).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse\u2026", command=self.pick).pack(
            side="left", padx=(8, 0))

        self.python_line = ttk.Label(frame, text="Looking for Python\u2026",
                                     foreground="#666666")
        self.python_line.pack(anchor="w", pady=(12, 0))

        self.llama = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, variable=self.llama,
            text="Also fetch llama.cpp (skip it if you already have one)"
        ).pack(anchor="w", pady=(10, 0))
        self.shortcut = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Add a desktop and Start menu shortcut",
                        variable=self.shortcut).pack(anchor="w")

        self.status = ttk.Label(frame, text="", foreground="#666666")
        self.status.pack(anchor="w", pady=(14, 4))
        self.bar = ttk.Progressbar(frame, mode="determinate", length=440)
        self.bar.pack(fill="x")

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(16, 0))
        self.go = ttk.Button(buttons, text="Install", command=self.start,
                             state="disabled")
        self.go.pack(side="right")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(
            side="right", padx=8)

        self.after(60, self.find_python)

    # -- python ------------------------------------------------------------
    def find_python(self) -> None:
        exe, note = usable_python()
        self.python = exe
        if exe:
            self.python_line.config(text=f"Using {note}", foreground="#2E7D5B")
            self.go.config(state="normal")
        else:
            # Not a dead end: the button becomes the way out of it.
            self.python_line.config(text=note, foreground="#B4453C")
            self.go.config(text="Get Python", state="normal",
                           command=self.open_python_page)

    def open_python_page(self) -> None:
        import webbrowser
        webbrowser.open("https://www.python.org/downloads/")
        messagebox.showinfo(
            "Python first",
            "Install Python, and tick \u201cAdd Python to PATH\u201d during "
            "setup.\n\nThen run this installer again.")

    # -- actions -----------------------------------------------------------
    def pick(self) -> None:
        chosen = filedialog.askdirectory(title=f"Where should {APP} go?")
        if not chosen:
            return
        folder = Path(chosen)
        # Into a folder of its own: "install to Documents" should not mean
        # sixty files loose in Documents.
        if folder.name.lower() != APP.lower():
            folder = folder / APP
        self.target.set(str(folder))

    def say(self, text: str, done: int | None = None) -> None:
        self.status.config(text=text)
        if done is not None:
            self.bar["value"] = done
        self.update_idletasks()

    def start(self) -> None:
        if self.working or not self.python:
            return
        self.working = True
        self.go.config(state="disabled")
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            target = self._install()
        except Exception as e:
            self.working = False
            self.go.config(state="normal")
            self.say("")
            messagebox.showerror("Could not install", str(e))
            return
        self.say("Done.", 100)
        if self.problems:
            messagebox.showwarning(
                f"{APP} installed, with a problem",
                f"{APP} is in\n{target}\n\n" + "\n\n".join(self.problems))
        else:
            messagebox.showinfo(
                APP, f"{APP} is installed in\n{target}\n\n"
                     "Open it from the shortcut.")
        self.destroy()

    # -- the work ----------------------------------------------------------
    def _install(self) -> Path:
        target = Path(self.target.get()).expanduser()
        self.say("Downloading Kestrel", 5)
        archive = self._download()

        self.say("Unpacking", 30)
        self._unpack(archive, target)

        self.say("Tidying up", 36)
        self._prune(target)

        self.say("Installing Python libraries", 40)
        self._dependencies(target)

        self.say("Building the launcher", 62)
        self._launcher(target)

        if self.llama.get():
            self.say("Downloading llama.cpp", 78)
            self._llama(target)

        self.say("Writing the uninstaller", 88)
        self._uninstaller(target)

        if self.shortcut.get():
            self.say("Making shortcuts", 94)
            self._shortcut(target)
            self.say("Checking the shortcuts", 97)
            self._verify(target)
        return target

    def _download(self) -> bytes:
        """The current source, from the branch.

        The branch rather than the latest release: a release is a snapshot from
        whenever it was tagged, so anything added since — a new skill, a fix —
        is simply absent, and the install looks incomplete for no visible
        reason.
        """
        try:
            request = urllib.request.Request(
                SOURCE, headers={"User-Agent": "Kestrel-Installer"})
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not download Kestrel.\n\n{e}\n\n"
                "Check the internet connection and try again.") from None

    def _unpack(self, archive: bytes, target: Path) -> None:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            names = bundle.namelist()
            # GitHub wraps everything in one folder named after the commit.
            # That folder is packaging, not part of the program.
            root = (names[0].split("/")[0] + "/") if names else ""
            if target.exists() and any(target.iterdir()):
                # Replaced, not merged: a module left over from an older
                # version is how a program half-updates and then fails in a way
                # nobody can reproduce.
                keep = target.with_name(target.name + ".old")
                shutil.rmtree(keep, ignore_errors=True)
                target.rename(keep)
            target.mkdir(parents=True, exist_ok=True)
            root_resolved = str(target.resolve())
            for name in names:
                if name.endswith("/") or not name.startswith(root):
                    continue
                destination = (target / name[len(root):]).resolve()
                # Never write outside the chosen folder, whatever paths the
                # archive claims to contain.
                if not str(destination).startswith(root_resolved):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, open(destination, "wb") as out:
                    shutil.copyfileobj(source, out)

    # Files that belong to developing Kestrel rather than running it. The
    # download is the whole repository, which is the simplest thing to fetch
    # and the wrong thing to leave lying in somebody's install.
    DEVELOPMENT_ONLY = (
        "install.bat", "install.sh", "build-exe.bat", "build-installer.bat",
        "Kestrel.spec", "Screenshot.png", ".gitignore", ".gitattributes",
        "models.json", "install-gui.py", "build-exe.py",
    )

    def _prune(self, target: Path) -> None:
        """Remove what only a checkout needs."""
        for name in self.DEVELOPMENT_ONLY:
            path = target / name
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass                  # not worth failing an install over
        for folder in (".github", "tests", "docs"):
            shutil.rmtree(target / folder, ignore_errors=True)

    def _dependencies(self, target: Path) -> None:
        """A virtual environment of Kestrel's own, and the libraries in it.

        Its own rather than the system's: installing PySide6 system-wide is
        somebody else's machine to break, and a folder that can be deleted is
        an uninstall.
        """
        venv = target / ".venv"
        self._run([str(self.python), "-m", "venv", str(venv)],
                  "Could not create the virtual environment")
        python = (venv / "Scripts" / "python.exe") if WINDOWS \
            else (venv / "bin" / "python")
        self._run([str(python), "-m", "pip", "install", "--upgrade", "pip"],
                  "Could not update pip", tolerate=True)

        self.say("Installing Python libraries", 46)
        requirements = target / "requirements.txt"
        command = [str(python), "-m", "pip", "install"]
        command += (["-r", str(requirements)] if requirements.is_file()
                    else ["PySide6", "requests", "psutil"])
        self._pip(command)

    def _launcher(self, target: Path) -> Path | None:
        """Build Kestrel.exe with PyInstaller, in the installed folder.

        This is the only reliable way to get Kestrel's own name and icon onto
        the taskbar. A shortcut that runs python — pythonw included — is, to
        Windows, Python running, and a Microsoft Store install carries an MSIX
        package identity that overrides anything the process asks for. The
        program cannot correct that from the inside; the identity belongs to
        the executable.

        Several minutes and a few hundred megabytes, once, at install time
        rather than every launch. The exe is a launcher: it loads Kestrel's
        modules from this folder, so updating stays a matter of replacing the
        source.
        """
        if not WINDOWS:
            return None
        python = target / ".venv" / "Scripts" / "python.exe"
        entry = target / "kestrel-run.py"
        if not entry.is_file():
            # Written rather than depended on. If the published source has not
            # got it yet, silently skipping the build is how you end up with
            # Python's icon and no explanation — the file is nine lines of
            # bootstrap, so the installer simply provides it.
            entry.write_text(run_entry_source(), "utf-8")

        got = subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "pyinstaller"],
            capture_output=True, text=True, creationflags=QUIET)
        if got.returncode != 0:
            tail = (got.stderr or got.stdout or "").strip().splitlines()[-3:]
            self.problems.append(
                "PyInstaller could not be installed, so no launcher was built "
                "and the taskbar will show Python's icon.\n\n"
                + "\n".join(tail))
            return None
        icon = target / "assets" / "kestrel.ico"
        command = [str(python), "-m", "PyInstaller", "--noconfirm", "--clean",
                   "--windowed", "--onedir", "--name", APP,
                   "--distpath", str(target / "launcher"),
                   "--workpath", str(target / "launcher-build"),
                   "--specpath", str(target / "launcher-build"),
                   # Where the package actually is. PyInstaller has to import
                   # kestrel to collect it, and the installed folder is not on
                   # the path of the Python doing the building.
                   "--paths", str(target),
                   "--collect-submodules", "kestrel"]
        if icon.is_file():
            command += ["--icon", str(icon)]
        command.append(str(entry))
        self.say("Building the launcher", 66)
        # Run from inside the installed folder, so relative imports and the
        # package itself resolve the way they will at runtime.
        result = subprocess.run(command, capture_output=True, text=True,
                                cwd=str(target), creationflags=QUIET)

        built = target / "launcher" / APP / f"{APP}.exe"
        if built.is_file():
            built = self._surface(target, built)
            shutil.rmtree(target / "launcher-build", ignore_errors=True)
            return built

        # Not silent. Falling back to pythonw leaves the taskbar showing
        # Python, which is the one thing the launcher exists to prevent — and
        # a failure nobody is told about looks exactly like the fix not
        # working.
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-4:]
        self.problems.append(
            "The launcher could not be built, so the shortcut runs Python "
            "instead and the taskbar will show Python's icon.\n\n"
            + "\n".join(tail)
            + "\n\nRunning this installer again over the same folder will "
              "try the build once more.")
        return None

    def _pip(self, command: list[str]) -> None:
        """Install, naming each library as pip reaches it.

        pip says what it is doing; the installer was throwing that away and
        showing one unchanging line for several minutes, which is
        indistinguishable from being stuck. Reading its output back means the
        slowest part of the install is also the part that visibly moves.
        """
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=QUIET)
        tail: list[str] = []
        done = 46
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            tail.append(line)
            del tail[:-8]              # only the end matters if it fails
            name = _pip_subject(line)
            if name:
                # Creeps towards the next stage rather than pretending to know
                # how many libraries are left: pip does not say up front.
                done = min(60, done + 1)
                self.say(f"Installing {name}", done)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Could not install the Python libraries\n\n"
                               + "\n".join(tail[-5:]))

    def _surface(self, target: Path, built: Path) -> Path:
        """Move the launcher up to the folder people actually open.

        PyInstaller puts a one-folder build two levels down. The exe cannot
        simply be lifted out on its own — it needs the `_internal` folder
        beside it, and separating them gives the "Failed to load Python DLL"
        error — so the whole contents move together.
        """
        produced = built.parent
        for item in produced.iterdir():
            destination = target / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
        shutil.rmtree(target / "launcher", ignore_errors=True)
        return target / f"{APP}.exe"

    def _llama(self, target: Path) -> None:
        script = target / ("install.bat" if WINDOWS else "install.sh")
        if not script.is_file():
            return
        command = ([str(script), "--yes", "--llama"] if WINDOWS
                   else ["bash", str(script), "--yes", "--llama"])
        self._run(command, "Could not install llama.cpp", tolerate=True)

    def _uninstaller(self, target: Path) -> None:
        """Make sure there is a way back out, in the folder.

        Newer sources ship uninstaller.bat and it arrives with the download;
        older ones do not, so a short one is written instead. Either way the
        folder contains the means to remove itself, which is the part people
        look for and the part that is usually missing.
        """
        if not WINDOWS:
            self._uninstaller_sh(target)
            return
        existing = target / "uninstaller.bat"
        if existing.is_file():
            return
        existing.write_text(UNINSTALL_BAT.replace("{APP}", APP), "utf-8")

    def _uninstaller_sh(self, target: Path) -> None:
        script = target / "uninstall.sh"
        if script.is_file():
            return
        script.write_text(UNINSTALL_SH, "utf-8")
        try:
            script.chmod(0o755)
        except OSError:
            pass

    def _shortcut(self, target: Path) -> None:
        """Kestrel's own shortcuts, and nothing else's.

        No other program's entries are read, moved or stamped. That is how
        pinned taskbar items get broken, and it is never worth it.
        """
        icon = target / "assets" / ("kestrel.ico" if WINDOWS else "kestrel.png")
        if WINDOWS:
            # The built launcher when there is one. Falling back to pythonw
            # leaves the taskbar showing Python, which is the whole reason the
            # launcher is built.
            built = target / f"{APP}.exe"
            runner = built if built.is_file() else \
                target / ".venv" / "Scripts" / "pythonw.exe"
            arguments = "" if built.is_file() else "-m kestrel"
            for link in self._shortcut_paths():
                link.parent.mkdir(parents=True, exist_ok=True)
                # The shortcut carries the same application identity the
                # program claims at startup. Windows matches a running window
                # to a pinned entry by that identity, not by the executable —
                # without it the taskbar shows a second button under Python's
                # name whatever icon the shortcut has.
                script = (
                    "$s = (New-Object -ComObject WScript.Shell)"
                    f".CreateShortcut('{link}'); "
                    f"$s.TargetPath = '{runner}'; "
                    + (f"$s.Arguments = '{arguments}'; " if arguments else "")
                    + f"$s.WorkingDirectory = '{target}'; "
                    + (f"$s.IconLocation = '{icon}'; " if icon.is_file() else "")
                    + "$s.Description = 'Kestrel'; $s.Save()")
                subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                                "-Command", script],
                               capture_output=True, creationflags=QUIET)
        else:
            runner = target / ".venv" / "bin" / "python"
            entry = (Path.home() / ".local" / "share" / "applications"
                     / "kestrel.desktop")
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(
                "[Desktop Entry]\nType=Application\nName=Kestrel\n"
                "Comment=A local AI agent\n"
                f"Exec={runner} -m kestrel\nPath={target}\n"
                f"Icon={icon}\nTerminal=false\nCategories=Development;\n",
                "utf-8")

    def _verify(self, target: Path) -> None:
        """Read the shortcuts back and check they point where intended.

        Writing a shortcut can succeed and still leave it pointing at the
        wrong thing — a path with a quote in it, a build that produced nothing.
        The symptom is the taskbar showing Python, which looks like the fix not
        working rather than the shortcut not being written. Reading it back
        turns that into something the installer can say out loud.
        """
        if not WINDOWS:
            return
        built = target / f"{APP}.exe"
        if not built.is_file():
            return                    # already reported when the build failed
        for link in self._shortcut_paths():
            if not link.is_file():
                continue
            script = ("(New-Object -COM WScript.Shell)"
                      f".CreateShortcut('{link}').TargetPath")
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, creationflags=QUIET)
            points_at = result.stdout.strip()
            if points_at and Path(points_at) != built:
                self.problems.append(
                    f"The {link.stem} shortcut points at\n{points_at}\n"
                    f"rather than\n{built}\n\n"
                    "The taskbar will show that program's icon instead of "
                    "Kestrel's.")

    def _shortcut_paths(self) -> list[Path]:
        places = []
        if WINDOWS:
            places.append(Path.home() / "Desktop" / f"{APP}.lnk")
            places.append(
                Path(os.environ.get("APPDATA", Path.home())) / "Microsoft"
                / "Windows" / "Start Menu" / "Programs" / f"{APP}.lnk")
        return places

    def _run(self, command: list[str], complaint: str,
             tolerate: bool = False) -> None:
        result = subprocess.run(command, capture_output=True, text=True,
                                creationflags=QUIET)
        if result.returncode != 0 and not tolerate:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            raise RuntimeError(complaint + "\n\n" + "\n".join(tail[-5:]))


# A short uninstaller, written only when the downloaded source did not bring
# one. Deliberately plain: it refuses any folder that is not a Kestrel install,
# asks before deleting, and leaves anything it did not put there.
# The launcher's entry point, carried here so the installer does not
# depend on the published source already containing it. Stored encoded
# only because it contains triple quotes of its own, which would end the
# string early — it is plain Python, written out verbatim.
RUN_ENTRY_B64 = (
    "IiIiRW50cnkgcG9pbnQgZm9yIHRoZSBidWlsdCBLZXN0cmVsLmV4ZS4KClR3byBqb2JzLCBib3Ro"
    "IG9mIHdoaWNoIGV4aXN0IGJlY2F1c2Ugb2YgaG93IFdpbmRvd3MgdHJlYXRzIGEgcHJvZ3JhbSdz"
    "CmlkZW50aXR5LgoKVGhlIGZpcnN0IGlzIHRoZSBpY29uLiBBIC5weSBmaWxlIGxhdW5jaGVkIHRo"
    "cm91Z2ggdGhlIFB5dGhvbiBpbnRlcnByZXRlciBpcywKYXMgZmFyIGFzIFdpbmRvd3MgaXMgY29u"
    "Y2VybmVkLCBQeXRob24gcnVubmluZyDigJQgdGhlIHRhc2tiYXIgc2hvd3MgUHl0aG9uJ3MKaWNv"
    "biBhbmQgZ3JvdXBzIEtlc3RyZWwncyB3aW5kb3cgd2l0aCBhbnkgb3RoZXIgUHl0aG9uIHByb2dy"
    "YW0uIE5vdGhpbmcgaW5zaWRlCnRoZSBhcHBsaWNhdGlvbiBjYW4gY29ycmVjdCB0aGF0LCBiZWNh"
    "dXNlIHRoZSBpZGVudGl0eSBiZWxvbmdzIHRvIHRoZSBwcm9jZXNzLApub3QgdGhlIHdpbmRvdy4g"
    "QSByZWFsIGV4ZWN1dGFibGUgd2l0aCBpdHMgb3duIGljb24gaXMgdGhlIG9ubHkgcmVsaWFibGUg"
    "Zml4LgoKVGhlIHNlY29uZCBpcyB1cGRhdGluZy4gQSBmcm96ZW4gZXhlY3V0YWJsZSBub3JtYWxs"
    "eSBjYXJyaWVzIGl0cyBvd24gY29weSBvZgpldmVyeSBtb2R1bGUsIHNvIHB1bGxpbmcgYSBuZXcg"
    "dmVyc2lvbiBvZiB0aGUgc291cmNlIHdvdWxkIGNoYW5nZSBub3RoaW5nIHVudGlsCnRoZSBleGUg"
    "d2FzIHJlYnVpbHQuIFRoZSBpbXBvcnQgaG9vayBiZWxvdyBwcmVmZXJzIGZpbGVzIG9uIGRpc2sg"
    "YmVzaWRlIHRoZQpleGVjdXRhYmxlLCBzbyBhbiB1cGRhdGUgaXMgYSBnaXQgcHVsbCBhcyBpdCBh"
    "bHdheXMgd2FzLCBhbmQgdGhlIGV4ZSBpcyBvbmx5CnJlYnVpbHQgd2hlbiB0aGUgbGF1bmNoZXIg"
    "aXRzZWxmIGNoYW5nZXMuCiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoK"
    "aW1wb3J0IG9zCmltcG9ydCBzeXMKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgoKZGVmIF9pbnN0"
    "YWxsX2RpcigpIC0+IFBhdGg6CiAgICAiIiJXaGVyZSBLZXN0cmVsJ3Mgb3duIGZpbGVzIGxpdmUs"
    "IGZyb3plbiBvciBub3QuIiIiCiAgICBpZiBnZXRhdHRyKHN5cywgImZyb3plbiIsIEZhbHNlKToK"
    "ICAgICAgICAjIFRoZSBmb2xkZXIgaG9sZGluZyB0aGUgZXhlLCBub3QgdGhlIHRlbXBvcmFyeSB1"
    "bnBhY2sgZGlyZWN0b3J5OiB0aGUKICAgICAgICAjIHBvaW50IGlzIHRvIGZpbmQgdGhlIHNvdXJj"
    "ZSB0aGUgdXNlciBjYW4gZWRpdCBhbmQgdXBkYXRlLgogICAgICAgIHJldHVybiBQYXRoKHN5cy5l"
    "eGVjdXRhYmxlKS5yZXNvbHZlKCkucGFyZW50CiAgICByZXR1cm4gUGF0aChfX2ZpbGVfXykucmVz"
    "b2x2ZSgpLnBhcmVudAoKCmNsYXNzIF9EaXNrRmlyc3Q6CiAgICAiIiJJbXBvcnQgS2VzdHJlbCdz"
    "IG1vZHVsZXMgZnJvbSBkaXNrIGluIHByZWZlcmVuY2UgdG8gdGhlIGJ1bmRsZS4KCiAgICBSZWdp"
    "c3RlcmVkIGFoZWFkIG9mIFB5SW5zdGFsbGVyJ3Mgb3duIGZpbmRlci4gT25seSBga2VzdHJlbGAg"
    "YW5kIGl0cwogICAgc3VibW9kdWxlcyBhcmUgYWZmZWN0ZWQg4oCUIGV2ZXJ5dGhpbmcgZWxzZSwg"
    "UHlTaWRlNiBpbmNsdWRlZCwgc3RpbGwgY29tZXMKICAgIGZyb20gdGhlIGJ1bmRsZSwgd2hpY2gg"
    "aXMgd2hhdCBrZWVwcyB0aGUgZXhlY3V0YWJsZSBzZWxmLWNvbnRhaW5lZC4KICAgICIiIgoKICAg"
    "IGRlZiBfX2luaXRfXyhzZWxmLCByb290OiBQYXRoKToKICAgICAgICBzZWxmLnJvb3QgPSByb290"
    "CgogICAgZGVmIGZpbmRfbW9kdWxlKHNlbGYsIG5hbWUsIHBhdGg9Tm9uZSk6ICAgICAgICAjIGxl"
    "Z2FjeSBBUEksIHN0aWxsIGNhbGxlZAogICAgICAgIHJldHVybiBOb25lCgogICAgZGVmIGZpbmRf"
    "c3BlYyhzZWxmLCBuYW1lLCBwYXRoPU5vbmUsIHRhcmdldD1Ob25lKToKICAgICAgICBpZiBuYW1l"
    "ICE9ICJrZXN0cmVsIiBhbmQgbm90IG5hbWUuc3RhcnRzd2l0aCgia2VzdHJlbC4iKToKICAgICAg"
    "ICAgICAgcmV0dXJuIE5vbmUKICAgICAgICBpbXBvcnQgaW1wb3J0bGliLm1hY2hpbmVyeSBhcyBt"
    "YWNoaW5lcnkKICAgICAgICBpbXBvcnQgaW1wb3J0bGliLnV0aWwgYXMgdXRpbAoKICAgICAgICBy"
    "ZWxhdGl2ZSA9IG5hbWUuc3BsaXQoIi4iKQogICAgICAgIGNhbmRpZGF0ZSA9IHNlbGYucm9vdC5q"
    "b2lucGF0aCgqcmVsYXRpdmUpCiAgICAgICAgcGFja2FnZSA9IGNhbmRpZGF0ZSAvICJfX2luaXRf"
    "Xy5weSIKICAgICAgICBtb2R1bGUgPSBjYW5kaWRhdGUud2l0aF9zdWZmaXgoIi5weSIpCiAgICAg"
    "ICAgaWYgcGFja2FnZS5pc19maWxlKCk6CiAgICAgICAgICAgIHNwZWMgPSB1dGlsLnNwZWNfZnJv"
    "bV9maWxlX2xvY2F0aW9uKAogICAgICAgICAgICAgICAgbmFtZSwgcGFja2FnZSwKICAgICAgICAg"
    "ICAgICAgIHN1Ym1vZHVsZV9zZWFyY2hfbG9jYXRpb25zPVtzdHIoY2FuZGlkYXRlKV0sCiAgICAg"
    "ICAgICAgICAgICBsb2FkZXI9bWFjaGluZXJ5LlNvdXJjZUZpbGVMb2FkZXIobmFtZSwgc3RyKHBh"
    "Y2thZ2UpKSkKICAgICAgICAgICAgcmV0dXJuIHNwZWMKICAgICAgICBpZiBtb2R1bGUuaXNfZmls"
    "ZSgpOgogICAgICAgICAgICByZXR1cm4gdXRpbC5zcGVjX2Zyb21fZmlsZV9sb2NhdGlvbigKICAg"
    "ICAgICAgICAgICAgIG5hbWUsIG1vZHVsZSwKICAgICAgICAgICAgICAgIGxvYWRlcj1tYWNoaW5l"
    "cnkuU291cmNlRmlsZUxvYWRlcihuYW1lLCBzdHIobW9kdWxlKSkpCiAgICAgICAgcmV0dXJuIE5v"
    "bmUKCgpkZWYgbWFpbigpIC0+IGludDoKICAgIHJvb3QgPSBfaW5zdGFsbF9kaXIoKQogICAgaWYg"
    "KHJvb3QgLyAia2VzdHJlbCIgLyAiX19pbml0X18ucHkiKS5pc19maWxlKCk6CiAgICAgICAgc3lz"
    "Lm1ldGFfcGF0aC5pbnNlcnQoMCwgX0Rpc2tGaXJzdChyb290KSkKICAgICAgICBvcy5jaGRpcihy"
    "b290KQoKICAgICMgV2luZG93cyBncm91cHMgdGFza2JhciBidXR0b25zIGJ5IGFwcGxpY2F0aW9u"
    "IGlkZW50aXR5LCBhbmQgYSBwcm9ncmFtCiAgICAjIHRoYXQgZG9lcyBub3QgY2xhaW0gb25lIGlu"
    "aGVyaXRzIHRoZSBpbnRlcnByZXRlcidzLiBTZXQgYmVmb3JlIGFueSB3aW5kb3cKICAgICMgZXhp"
    "c3RzLCBhbmQgb25seSBldmVyIHRvIEtlc3RyZWwncyBvd24g4oCUIHN0YW1waW5nIGFuIGlkZW50"
    "aXR5IG9udG8KICAgICMgYW55dGhpbmcgZWxzZSBpcyBob3cgb3RoZXIgYXBwbGljYXRpb25zJyBw"
    "aW5uZWQgc2hvcnRjdXRzIGdldCBicm9rZW4uCiAgICBpZiBzeXMucGxhdGZvcm0gPT0gIndpbjMy"
    "IjoKICAgICAgICB0cnk6CiAgICAgICAgICAgIGltcG9ydCBjdHlwZXMKICAgICAgICAgICAgY3R5"
    "cGVzLndpbmRsbC5zaGVsbDMyLlNldEN1cnJlbnRQcm9jZXNzRXhwbGljaXRBcHBVc2VyTW9kZWxJ"
    "RCgKICAgICAgICAgICAgICAgICJBbHRlcm5pVGVjaC5LZXN0cmVsIikKICAgICAgICBleGNlcHQg"
    "RXhjZXB0aW9uOgogICAgICAgICAgICBwYXNzCgogICAgZnJvbSBrZXN0cmVsLl9fbWFpbl9fIGlt"
    "cG9ydCBtYWluIGFzIGtlc3RyZWxfbWFpbgogICAgcmV0dXJuIGtlc3RyZWxfbWFpbigpCgoKaWYg"
    "X19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIHJhaXNlIFN5c3RlbUV4aXQobWFpbigpKQo="
)


def run_entry_source() -> str:
    import base64
    return base64.b64decode(RUN_ENTRY_B64).decode("utf-8")


UNINSTALL_BAT = r"""@echo off
setlocal EnableDelayedExpansion
set "TARGET=%~dp0"
if "%~1" neq "" set "TARGET=%~1"
if "!TARGET:~-1!"=="\" set "TARGET=!TARGET:~0,-1!"
echo.
echo   Uninstall {APP}
echo   Folder: !TARGET!
echo.
if not exist "!TARGET!\kestrel\__init__.py" (
    echo   That is not a {APP} installation. Nothing was changed.
    pause
    exit /b 1
)
set "WRITABLE=1"
2>nul ( >"!TARGET!\.write-test" echo. ) || set "WRITABLE=0"
if exist "!TARGET!\.write-test" del /f /q "!TARGET!\.write-test" >nul 2>&1
if "!WRITABLE!"=="0" (
    echo   This folder needs administrator rights.
    echo.
    echo   Close this window, then right-click this file and choose
    echo     Run as administrator
    echo.
    pause
    exit /b 1
)
echo   This removes the program, its Python libraries and its
echo   shortcuts. Your models, llama.cpp and your conversations
echo   are left alone.
echo.
set /p "SURE=  Type YES to remove {APP}: "
if /i not "!SURE!"=="YES" (
    echo   Nothing was changed.
    pause
    exit /b 0
)
taskkill /im {APP}.exe /f >nul 2>&1
del /f /q "%USERPROFILE%\Desktop\{APP}.lnk" >nul 2>&1
del /f /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\{APP}.lnk" >nul 2>&1
set "FAILED="
for %%D in (kestrel assets personas skills _internal launcher launcher-build .venv) do (
    if exist "!TARGET!\%%D" (
        rmdir /s /q "!TARGET!\%%D" >nul 2>&1
        if exist "!TARGET!\%%D" set "FAILED=!FAILED! %%D"
    )
)
for %%F in (kestrel-run.py installer.py install.bat install.sh run.bat run.sh ^
            node.bat node.sh requirements.txt version.txt README.md LICENSE ^
            Kestrel.exe) do (
    if exist "!TARGET!\%%F" (
        del /f /q "!TARGET!\%%F" >nul 2>&1
        if exist "!TARGET!\%%F" set "FAILED=!FAILED! %%F"
    )
)
rmdir "!TARGET!" >nul 2>&1
echo.
if defined FAILED (
    echo   Some things could not be removed:
    echo    !FAILED!
    echo   Usually a file still in use. Close {APP} and retry.
    pause
    exit /b 1
)
echo   {APP} is removed. Settings remain in:
echo     %APPDATA%\kestrel
echo.
pause
"""

UNINSTALL_SH = r"""#!/usr/bin/env bash
# Remove Kestrel from this folder. Models, llama.cpp and saved
# conversations are left alone.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$here/kestrel/__init__.py" ]; then
    echo "That is not a Kestrel installation. Nothing was changed."
    exit 1
fi
echo "This removes Kestrel from: $here"
printf "Type YES to remove it: "
read -r sure
[ "$sure" = "YES" ] || { echo "Nothing was changed."; exit 0; }
rm -rf "$here/kestrel" "$here/assets" "$here/personas" "$here/skills" \
       "$here/launcher" "$here/launcher-build" "$here/.venv"
rm -f "$here"/{kestrel-run.py,installer.py,install.sh,run.sh,node.sh} \
      "$here"/{requirements.txt,version.txt,README.md,LICENSE}
rm -f "$HOME/.local/share/applications/kestrel.desktop"
rmdir "$here" 2>/dev/null
echo "Kestrel is removed. Settings remain in ~/.config/kestrel"
"""


if __name__ == "__main__":
    Installer().mainloop()
