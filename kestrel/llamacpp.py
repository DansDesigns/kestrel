"""Finding llama.cpp, and fetching it when it isn't there.

Order of preference: something already installed, then an official prebuilt
release, then a source build. Prebuilt assets are matched by scoring their names
against the platform and the requested backend rather than by exact filename,
because llama.cpp's asset naming drifts between releases and a hardcoded pattern
would rot.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import stat
import time
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

REPO = "ggml-org/llama.cpp"
API = f"https://api.github.com/repos/{REPO}"
BACKENDS = ["auto", "cpu", "cuda", "vulkan", "hip", "metal", "sycl"]

# Order is preference. The dedicated server binary wins; the unified `llama`
# dispatcher is accepted as a fallback and invoked as `llama serve`.
SERVER_NAMES = ["llama-server", "llama-server.exe", "server", "server.exe",
                "llama", "llama.exe"]
RPC_NAMES = ["ggml-rpc-server", "rpc-server", "ggml-rpc-server.exe", "rpc-server.exe"]

Progress = Callable[[str], None]


# ------------------------------------------------------------------ finding --
def search_roots() -> list[Path]:
    home = Path.home()
    roots = [
        install_dir(),
        home / "llama.cpp" / "build" / "bin",
        home / "llama.cpp",
        home / ".local" / "bin",
        home / "Downloads" / "llama.cpp",
        Path("/usr/local/bin"), Path("/usr/bin"), Path("/opt/llama.cpp/bin"),
        Path("/opt/homebrew/bin"), Path("/snap/bin"),
        Path.cwd() / "llama.cpp" / "build" / "bin",
        Path.cwd() / "bin",
    ]
    if os.name == "nt":
        for var in ("LOCALAPPDATA", "PROGRAMFILES", "USERPROFILE"):
            base = os.environ.get(var)
            if base:
                roots += [Path(base) / "llama.cpp", Path(base) / "llama.cpp" / "build" / "bin"]
    return roots


def install_dir() -> Path:
    """Where Kestrel puts llama.cpp when it installs it itself."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "kestrel" / "llama.cpp"


def _executable(p: Path) -> bool:
    return p.is_file() and (os.name == "nt" or os.access(p, os.X_OK))


def find(names, hint: str = "", deep: bool = True) -> str:
    """Locate a binary: explicit hint, then PATH, then the usual places."""
    if hint:
        h = Path(hint).expanduser()
        if _executable(h):
            return str(h)
        for n in names:                       # hint may be a directory
            if _executable(h / n):
                return str(h / n)
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    for root in search_roots():
        for n in names:
            p = root / n
            if _executable(p):
                return str(p)
    if deep:
        for root in (install_dir(), Path.home() / "llama.cpp"):
            if not root.is_dir():
                continue
            try:
                for p in root.rglob("*"):
                    if p.name in names and _executable(p):
                        return str(p)
            except OSError:
                continue
    return ""


def find_server(hint: str = "") -> str:
    return find(SERVER_NAMES, hint)


def find_rpc(hint: str = "") -> str:
    return find(RPC_NAMES, hint)


@dataclass
class Found:
    server: str = ""
    rpc: str = ""
    version: str = ""
    working: bool = False        # the binary was actually executed successfully
    managed: bool = False        # Kestrel installed it, so Kestrel may remove it
    problem: str = ""            # why it does not run, when it does not

    @property
    def ok(self) -> bool:
        """Present *and* functional. A binary that will not execute is not a
        usable installation, and treating it as one is how a broken setup
        silently persists."""
        return bool(self.server) and self.working

    @property
    def present(self) -> bool:
        return bool(self.server)


def is_managed(path: str) -> bool:
    """Did Kestrel install this? Only then is it ours to delete."""
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(install_dir().resolve())
        return True
    except (ValueError, OSError):
        return False


# Windows reports these as process exit codes rather than messages, so an
# unexplained eight-digit number is all the user would otherwise see.
NT_STATUS = {
    3221225781: "a required DLL is missing (0xC0000135 STATUS_DLL_NOT_FOUND)",
    3221225785: "a DLL entry point is missing (0xC0000139)",
    3221225477: "access violation (0xC0000005)",
    3221225595: "a DLL failed to initialise (0xC0000142)",
    3221225727: "the CPU lacks an instruction this build uses (0xC00001BF)",
}


def verify(binary: str) -> tuple[bool, str]:
    """Actually run it.

    Existence on disk proves nothing: the common failures are a CUDA build on a
    machine with no CUDA runtime, a half-finished extraction, an architecture
    mismatch, or a missing shared library. All of these leave a file that looks
    fine and fails the moment it is executed.
    """
    if not binary:
        return False, "not found"
    p = Path(binary)
    if not p.is_file():
        return False, "the recorded path no longer exists"
    if os.name != "nt" and not os.access(p, os.X_OK):
        return False, "not executable (permissions)"
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True,
                              timeout=25, errors="replace")
    except subprocess.TimeoutExpired:
        return False, "timed out on startup"
    except OSError as e:
        return False, str(e)
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    low = text.lower()
    if proc.returncode in NT_STATUS:
        return False, NT_STATUS[proc.returncode]
    for marker in ("error while loading shared libraries", "cannot open shared object",
                   "is not a valid win32", "symbol lookup error", "illegal instruction",
                   "no such file or directory", "dll load failed",
                   "cannot execute binary file"):
        if marker in low:
            return False, text.splitlines()[0][:180] if text else marker
    if proc.returncode != 0 and not re.search(r"\bb?\d{3,5}\b|version", low):
        return False, (text.splitlines()[0][:180] if text
                       else f"exited with code {proc.returncode}")
    return True, text.splitlines()[0][:120] if text else "ok"


def scan(server_hint: str = "", rpc_hint: str = "") -> Found:
    server = find_server(server_hint)
    working, detail = (verify(server) if server else (False, ""))

    # A stale configured path can point at something that no longer runs. Rather
    # than reporting that as the state of the machine, look for a working
    # install elsewhere before giving up.
    if server and not working and server_hint:
        alternative = find_server("")
        if alternative and alternative != server:
            alt_ok, alt_detail = verify(alternative)
            if alt_ok:
                server, working, detail = alternative, True, alt_detail

    f = Found(server=server, rpc=find_rpc(rpc_hint), working=working)
    if f.server:
        f.managed = is_managed(f.server)
        if working:
            f.version = version_of(f.server)
        else:
            f.problem = detail
    return f


def installed_paths(include_source: bool = True) -> list[Path]:
    """Everything Kestrel has put on disk for llama.cpp, whether or not any of
    it currently works. Removal has to be offered on the strength of the
    directory existing, not on finding a functioning binary in it — a partial
    extraction has no working binary and is exactly what needs clearing out."""
    root = install_dir()
    out = []
    for name in (["current"] + (["src"] if include_source else [])):
        if (root / name).exists():
            out.append(root / name)
    if root.is_dir():
        for leftover in root.glob("*.zip"):
            out.append(leftover)
        for leftover in root.glob("*.part"):
            out.append(leftover)
        for leftover in root.glob("*.tar.gz"):
            out.append(leftover)
    return out


def _force_remove(path: Path) -> str:
    """Delete a file or tree, clearing read-only attributes on the way.

    Windows refuses to unlink read-only files, and archive extraction routinely
    produces them, so a plain rmtree leaves a directory that looks undeletable.
    """
    def handler(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:
            pass

    for attempt in range(3):
        try:
            if path.is_dir():
                try:
                    shutil.rmtree(path, onexc=handler)      # Python 3.12+
                except TypeError:
                    shutil.rmtree(path, onerror=handler)
            else:
                try:
                    os.chmod(path, stat.S_IWRITE)
                except OSError:
                    pass
                path.unlink()
        except Exception as e:
            last = e
        if not path.exists():
            return ""
        time.sleep(0.4 * (attempt + 1))
    try:
        return str(last)
    except NameError:
        return "still present after three attempts"


def uninstall(progress: Progress | None = None, include_source: bool = True
              ) -> tuple[list[str], list[str]]:
    """Remove the installation Kestrel made. Returns (removed, failures).

    Deliberately confined to Kestrel's own directory. A llama.cpp from a package
    manager, Homebrew, or the user's own build is not ours to delete, and doing
    so would be a surprising thing for an installer to do.
    """
    say = progress or (lambda m: None)
    removed: list[str] = []
    failures: list[str] = []
    targets = installed_paths(include_source)
    if not targets:
        say("nothing to remove — no Kestrel-managed installation found")
        return removed, failures

    for target in targets:
        say(f"removing {target}")
        problem = _force_remove(target)
        if problem:
            failures.append(f"{target}: {problem}")
            say(f"could not remove {target}: {problem}")
        else:
            removed.append(str(target))

    if failures:
        say("Something is still holding those files open. Close Kestrel and any "
            "running llama-server, then try again.")
    return removed, failures


def version_of(binary: str) -> str:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             timeout=15)
        text = (out.stdout or "") + (out.stderr or "")
        m = re.search(r"\bb?(\d{3,5})\b", text)
        return m.group(0) if m else text.strip().splitlines()[0][:60] if text.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------- platform --
def detect_backend() -> str:
    """Best guess at the accelerator available on this machine."""
    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocminfo") or Path("/opt/rocm").exists():
        return "hip"
    if shutil.which("vulkaninfo"):
        return "vulkan"
    return "cpu"


def platform_tokens() -> tuple[list[str], list[str]]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        os_tokens = ["macos", "darwin", "osx"]
    elif system == "windows":
        os_tokens = ["win", "windows"]
    else:
        os_tokens = ["ubuntu", "linux"]
    if machine in ("arm64", "aarch64"):
        arch_tokens = ["arm64", "aarch64"]
    elif machine in ("x86_64", "amd64", "x64"):
        arch_tokens = ["x64", "x86_64", "amd64"]
    else:
        arch_tokens = [machine]
    return os_tokens, arch_tokens


def score_asset(name: str, backend: str) -> int:
    """Higher is better; negative means unusable on this machine."""
    low = name.lower()
    if not low.endswith((".zip", ".tar.gz", ".tgz")):
        return -100
    if "cudart" in low or low.startswith("cudart"):
        return -100          # the CUDA runtime bundle, not the binaries
    if "source" in low or low.endswith((".sha256", ".txt")):
        return -100

    os_tokens, arch_tokens = platform_tokens()
    score = 0
    if any(t in low for t in os_tokens):
        score += 40
    else:
        return -100          # wrong operating system
    if any(t in low for t in arch_tokens):
        score += 20
    elif any(t in low for t in ("x64", "arm64", "x86_64", "aarch64")):
        return -100          # explicitly a different architecture

    if backend and backend not in ("auto", "cpu"):
        if backend in low:
            score += 30
        else:
            score -= 10
    else:
        if any(b in low for b in ("cuda", "hip", "sycl", "vulkan")):
            score -= 15      # do not hand a CPU box a CUDA build
        else:
            score += 10
    if "avx512" in low:
        score -= 2           # widely supported but not universally
    return score


def pick_asset(names: list[str], backend: str) -> str:
    scored = [(score_asset(n, backend), n) for n in names]
    scored = [(s, n) for s, n in scored if s > 0]
    if not scored:
        return ""
    scored.sort(key=lambda t: (-t[0], len(t[1])))
    return scored[0][1]


# ---------------------------------------------------------------- installing --
def latest_release(timeout: float = 30.0) -> dict:
    r = requests.get(f"{API}/releases", params={"per_page": 5}, timeout=timeout,
                     headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        raise RuntimeError(data.get("message") or "unexpected response from GitHub")
    for rel in data:
        if rel.get("assets"):
            return rel
    raise RuntimeError("no release with prebuilt assets found")


def rpc_available(root: Path | str) -> bool:
    """Does this installation include the RPC worker binary?"""
    return bool(find(RPC_NAMES, str(root)))


def install_prebuilt(backend: str = "auto", dest: Path | None = None,
                     progress: Progress | None = None,
                     cancel: Callable[[], bool] | None = None,
                     with_rpc: bool = True) -> str:
    """Download and unpack an official build. Returns the llama-server path."""
    say = progress or (lambda m: None)
    dest = Path(dest or install_dir())
    if backend in ("", "auto"):
        backend = detect_backend()
        say(f"detected backend: {backend}")

    say("asking GitHub for the latest release…")
    rel = latest_release()
    tag = rel.get("tag_name", "?")
    assets = {a["name"]: a for a in rel.get("assets", [])}
    name = pick_asset(list(assets), backend)
    if not name:
        raise RuntimeError(
            f"No prebuilt binary for {platform.system()} {platform.machine()} "
            f"with backend '{backend}' in release {tag}. Build from source instead."
        )
    asset = assets[name]
    say(f"release {tag}: {name} ({asset.get('size', 0) // 1048576} MB)")
    if backend not in ("cpu", "auto") and backend not in name.lower():
        # llama.cpp ships no Linux CUDA/HIP prebuilt, for instance. Taking the
        # CPU build silently would look like a mysterious 100x slowdown later.
        say(f"note: no prebuilt {backend} build exists for this platform, so this "
            f"is a CPU build. For GPU acceleration, install from source instead.")

    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "current"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    _fetch_and_unpack(asset, dest, target, say, cancel)
    _make_executable(target)
    server = find(SERVER_NAMES, str(target))
    if not server:
        raise RuntimeError(f"unpacked {name} but found no llama-server inside")

    # On Windows the backend archives carry only the backend library; the base
    # ggml and llama DLLs live in the CPU package. Extracting one without the
    # other produces binaries that exist, look correct, and fail to start with
    # STATUS_DLL_NOT_FOUND. Overlay the base package and try again.
    ok, detail = verify(server)
    if not ok and os.name == "nt":
        base_name = pick_asset(list(assets), "cpu")
        if base_name and base_name != name:
            say(f"it will not start: {detail}")
            say(f"adding the base runtime from {base_name} — Windows backend "
                "archives do not include it")
            try:
                _fetch_and_unpack(assets[base_name], dest, target, say, cancel)
                _make_executable(target)
                ok, detail = verify(server)
            except Exception as e:
                say(f"could not add the base runtime: {e}")
    if not ok:
        say(f"warning: the installed binary does not run — {detail}")
    if with_rpc and not rpc_available(target):
        # Whether rpc-server ships in a prebuilt varies by platform and release.
        # Saying so is better than letting the Cluster tab fail later with no
        # explanation of why no worker binary exists.
        say("note: this prebuilt does not include rpc-server, so this machine "
            "cannot act as a cluster worker. Build from source with RPC enabled "
            "if you need that.")
    say(f"installed {tag} to {target}")
    return server


def _fetch_and_unpack(asset: dict, dest: Path, target: Path,
                      say: Progress, cancel: Callable[[], bool] | None) -> None:
    name = asset["name"]
    archive = dest / name
    with requests.get(asset["browser_download_url"], stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = step = 0
        with open(archive, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if cancel and cancel():
                    raise InterruptedError("cancelled")
                f.write(chunk)
                done += len(chunk)
                step += 1
                if total and step % 8 == 0:
                    say(f"downloading {name}… {100 * done // total}%")
    say(f"unpacking {name}…")
    _unpack(archive, target)
    try:
        archive.unlink()
    except OSError:
        pass


def _unpack(archive: Path, target: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    continue                      # never write outside target
                z.extract(member, target)
    else:
        with tarfile.open(archive) as t:
            safe = [m for m in t.getmembers()
                    if not m.name.startswith("/") and ".." not in Path(m.name).parts]
            t.extractall(target, members=safe)


def _make_executable(root: Path) -> None:
    if os.name == "nt":
        return
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix == "" or p.name in SERVER_NAMES + RPC_NAMES):
            try:
                p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError:
                pass


def build_from_source(backend: str = "auto", dest: Path | None = None,
                      progress: Progress | None = None,
                      with_rpc: bool = True) -> str:
    """Clone and build. Slower, but works where no prebuilt asset fits."""
    say = progress or (lambda m: None)
    dest = Path(dest or install_dir()) / "src"
    if backend in ("", "auto"):
        backend = detect_backend()

    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is required to build from source but is not installed")

    if not (dest / ".git").exists():
        say("cloning llama.cpp…")
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", f"https://github.com/{REPO}.git", str(dest)], say)
    else:
        say("updating existing checkout…")
        _run(["git", "-C", str(dest), "pull", "--ff-only"], say)

    flags = [f"-DGGML_RPC={'ON' if with_rpc else 'OFF'}", "-DLLAMA_CURL=OFF"]
    flags += {
        "cuda": ["-DGGML_CUDA=ON"], "hip": ["-DGGML_HIP=ON"],
        "vulkan": ["-DGGML_VULKAN=ON"], "sycl": ["-DGGML_SYCL=ON"],
        "metal": ["-DGGML_METAL=ON"],
    }.get(backend, [])
    say(f"configuring ({backend}); RPC "
        + ("enabled — this machine can also act as a cluster worker"
           if with_rpc else "disabled"))
    _run(["cmake", "-B", str(dest / "build"), "-S", str(dest),
          "-DCMAKE_BUILD_TYPE=Release", *flags], say)
    say("building — this takes a while")
    _run(["cmake", "--build", str(dest / "build"), "--config", "Release",
          "-j", str(max(1, (os.cpu_count() or 4)))], say)

    server = find(SERVER_NAMES, str(dest / "build" / "bin"))
    if not server:
        raise RuntimeError("build finished but llama-server was not produced")
    say(f"built {server}")
    return server


def _run(cmd: list[str], say: Progress) -> None:
    say("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1)
    tail: list[str] = []
    for line in proc.stdout:                      # type: ignore[union-attr]
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        if line:
            say("  " + line[:160])
    if proc.wait() != 0:
        raise RuntimeError("command failed:\n" + "\n".join(tail[-12:]))


def ensure(cfg, progress: Progress | None = None, allow_install: bool = False,
           backend: str = "auto", source: bool = False,
           with_rpc: bool | None = None, remove_first: bool = False) -> Found:
    """Scan, and optionally install if nothing is present.

    Writes what it finds back into the config, so this is the single entry point
    for both the startup check and the button in the interface.
    """
    say = progress or (lambda m: None)
    if remove_first and allow_install:
        uninstall(progress=say)
        cfg.llama_server_bin = ""
        cfg.rpc_bin = ""
    found = scan(cfg.llama_server_bin, cfg.rpc_bin)
    if found.present and not found.working:
        say(f"found {found.server} but it does not run: {found.problem}")
    if found.ok:
        say(f"found llama-server at {found.server}"
            + (f" ({found.version})" if found.version else ""))
    elif allow_install:
        if remove_first:
            uninstall(progress=say)[0]
            cfg.llama_server_bin = ""
            cfg.rpc_bin = ""
        say("no working llama.cpp found — installing")
        rpc = getattr(cfg, "llama_with_rpc", True) if with_rpc is None else with_rpc
        server = (build_from_source(backend, progress=say, with_rpc=rpc) if source
                  else install_prebuilt(backend, progress=say, with_rpc=rpc))
        found = scan(server, cfg.rpc_bin)
    else:
        say("no llama.cpp found")
        return found

    if found.ok:
        # Only a verified binary is worth remembering; recording a broken path
        # makes the next run believe the machine is already set up.
        cfg.llama_server_bin = found.server
        if found.rpc:
            cfg.rpc_bin = found.rpc
        try:
            cfg.save()
        except Exception:
            pass
    return found
