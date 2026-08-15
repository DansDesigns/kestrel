"""Spreading one model across several machines.

llama.cpp already knows how to do this: each worker runs `rpc-server`, the head
node runs `llama-server --rpc host:port,host:port` and the scheduler splits the
weights and KV cache across every backend it can see, local and remote. Kestrel
handles the awkward parts — finding the workers, checking they answer, deriving
sensible split proportions from their memory pools, and building the command.

Two caveats worth repeating from upstream, because they bite people:

  * Every machine must run the *same* llama.cpp build. Mismatched versions hang
    at the handshake or crash mid-inference. Pin a tag everywhere.
  * The RPC protocol has no authentication whatsoever. Keep it on a LAN or a
    VPN; never expose a worker to the internet.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import Config, Node

BEACON_MAGIC = "kestrel-node/1"
SERVER_BINARIES = ["llama-server", "llama-server.exe", "server", "server.exe"]
RPC_BINARIES = ["ggml-rpc-server", "rpc-server", "ggml-rpc-server.exe", "rpc-server.exe"]


# ---------------------------------------------------------------- binaries --
def find_binary(names: Iterable[str], extra_dirs: Iterable[str] = ()) -> str:
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    roots = [Path(d) for d in extra_dirs if d]
    roots += [Path.home() / "llama.cpp", Path.home() / "llama.cpp" / "build" / "bin",
              Path("/usr/local/bin"), Path("/opt/llama.cpp/bin"),
              Path.cwd() / "llama.cpp" / "build" / "bin"]
    for r in roots:
        for n in names:
            p = r / n
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
    return ""


def find_server_binary(hint: str = "") -> str:
    if hint and Path(hint).is_file():
        return hint
    return find_binary(SERVER_BINARIES, [str(Path(hint).parent)] if hint else [])


def find_rpc_binary(hint: str = "") -> str:
    if hint and Path(hint).is_file():
        return hint
    return find_binary(RPC_BINARIES, [str(Path(hint).parent)] if hint else [])


# ------------------------------------------------------------------ probes --
@dataclass
class Probe:
    node: Node
    up: bool
    latency_ms: float = 0.0
    error: str = ""


def probe(node: Node, timeout: float = 1.5) -> Probe:
    """rpc-server speaks a raw ggml protocol, so a TCP handshake is the honest
    check — anything more would need a matching client build."""
    started = time.time()
    try:
        with socket.create_connection((node.host, int(node.port)), timeout=timeout):
            return Probe(node, True, (time.time() - started) * 1000)
    except OSError as e:
        return Probe(node, False, 0.0, str(e))


def probe_all(nodes: list[Node], timeout: float = 1.5) -> list[Probe]:
    results: list[Probe | None] = [None] * len(nodes)
    threads = []

    def work(i: int, n: Node):
        results[i] = probe(n, timeout)

    for i, n in enumerate(nodes):
        t = threading.Thread(target=work, args=(i, n), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout + 0.5)
    return [r or Probe(nodes[i], False, error="timed out") for i, r in enumerate(results)]


# --------------------------------------------------------------- discovery --
def discover(port: int = 50051, seconds: float = 3.0,
             on_found: Callable[[Node], None] | None = None) -> list[Node]:
    """Listen for beacons from `kestrel-node` workers on the local network."""
    found: dict[str, Node] = {}
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        sock.bind(("", int(port)))
    except OSError:
        return []
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8", "replace"))
            except ValueError:
                continue
            if msg.get("magic") != BEACON_MAGIC:
                continue
            node = Node(host=msg.get("host") or addr[0],
                        port=int(msg.get("rpc_port") or 50052),
                        label=str(msg.get("label") or ""),
                        mem_mb=int(msg.get("mem_mb") or 0))
            if node.addr not in found:
                found[node.addr] = node
                if on_found:
                    on_found(node)
    finally:
        sock.close()
    return list(found.values())


def broadcast(port: int, payload: dict, interval: float, stop: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    blob = json.dumps(payload).encode("utf-8")
    while not stop.is_set():
        try:
            sock.sendto(blob, ("255.255.255.255", int(port)))
        except OSError:
            pass
        stop.wait(interval)
    sock.close()


# ------------------------------------------------------------ split weights --
def tensor_split(cfg: Config, local_mem_mb: int = 0) -> str:  # noqa: D401
    """Proportions in device order, local devices first then each RPC node.

    llama.cpp defaults to an even split, which is wrong the moment your machines
    differ — a 24 GB card and an 8 GB laptop should not carry the same load.
    """
    if cfg.runtime.tensor_split.strip():
        return cfg.runtime.tensor_split.strip()
    nodes = cfg.active_nodes()
    if not nodes or not any(n.mem_mb for n in nodes):
        return ""
    weights = []
    if local_mem_mb > 0:
        weights.append(local_mem_mb)
    for n in nodes:
        weights.append(n.mem_mb or 1024)
    total = sum(weights) or 1
    return ",".join(f"{w / total:.3f}" for w in weights)


# ---------------------------------------------------------- server process --
def build_command(cfg: Config, host: str = "", port: int = 0,
                  with_model: bool = True) -> list[str]:
    """Full llama-server command line, including RPC workers and their split."""
    from .runtime import build_command as _build

    if host:
        cfg.runtime.host = host
    if port:
        cfg.runtime.port = port
    rpc = cfg.rpc_arg()
    if rpc and not cfg.runtime.tensor_split:
        split = tensor_split(cfg)
        if split:
            cfg.runtime.tensor_split = split
    return _build(cfg, rpc=rpc, with_model=with_model)


def port_in_use(host: str, port: int, timeout: float = 0.6) -> bool:
    target = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((target, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def endpoint_alive(url: str, timeout: float = 2.5) -> bool:
    """Is something already serving here? Starting a second server on an
    occupied port silently fails, and the health check then passes against the
    process that was already there — which looks like success."""
    import requests
    for path in ("/health", "/v1/models"):
        try:
            r = requests.get(url.rstrip("/") + path, timeout=timeout)
            if r.status_code < 500:
                return True
        except Exception:
            continue
    return False


class ServerProcess:
    """Supervises a locally launched llama-server."""

    def __init__(self, on_log: Callable[[str], None] | None = None):
        self.proc: subprocess.Popen | None = None
        self.on_log = on_log or (lambda line: None)
        self.command: list[str] = []
        self.command_host = ""
        self.command_port = 0
        self.tail: list[str] = []
        self.exit_code: int | None = None
        self._reader: threading.Thread | None = None
        self.adopted = False     # something was already serving; we did not spawn

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cfg: Config, host: str = "", port: int = 0,
              adopt: bool = True, with_model: bool = True) -> bool:
        """Start the server. Returns True if a process was spawned, False if an
        existing server was adopted instead.

        Binding is checked first: if another process already holds the port,
        spawning ours produces a process that dies immediately while the health
        check succeeds against the incumbent.
        """
        if self.running:
            # Replacing our own server is the normal way to load a different
            # model, so it is done rather than refused.
            self.on_log("[stopping the running server first]")
            if not self.stop():
                raise RuntimeError(
                    "The running server did not stop. Close it from the Cluster "
                    "tab, or change the port under Params \u2192 Runtime.")
        self.command = build_command(cfg, host, port, with_model=with_model)
        self.command_host = cfg.runtime.host
        self.command_port = cfg.runtime.port
        self.tail = []
        self.exit_code = None
        self.adopted = False

        url = cfg.runtime.url()
        if port_in_use(cfg.runtime.host, cfg.runtime.port):
            if adopt and endpoint_alive(url):
                self.adopted = True
                self.on_log(f"[a server is already responding on {url} — using it "
                            "rather than starting another]")
                return False
            raise RuntimeError(
                f"Port {cfg.runtime.port} is already in use on {cfg.runtime.host}, "
                "but nothing is answering there. Stop whatever holds it, or change "
                "the port under Models \u2192 Runtime."
            )
        self.on_log("$ " + " ".join(self.command))
        creation = 0
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1, creationflags=creation,
        )
        self._reader = threading.Thread(target=self._pump, args=(self.proc,),
                                        daemon=True)
        self._reader.start()
        return True

    def _pump(self, proc=None) -> None:
        # The process is captured rather than read from the attribute: stop()
        # clears it, and this thread outlives that by however long the pipe
        # takes to drain.
        proc = proc or self.proc
        if proc is None:
            return
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                self.tail = (self.tail + [line])[-40:]
                self.on_log(line)
        except Exception:
            pass
        # poll() straight after the pipe closes can still return None; wait()
        # gives the real status, which is what makes a startup failure legible.
        try:
            self.exit_code = proc.wait(timeout=10)
        except Exception:
            try:
                self.exit_code = proc.poll()
            except Exception:
                self.exit_code = None
        self.on_log(f"[llama-server exited with code {self.exit_code}]")

    def failure_summary(self) -> str:
        lines = [l for l in self.tail if l.strip()][-6:]
        head = f"llama-server exited with code {self.exit_code}."
        return head + ("\n" + "\n".join(lines) if lines else "")

    def wait_healthy(self, url: str, timeout: float = 300.0) -> bool:
        """Loading a big model over RPC is slow; be patient but not infinite."""
        import requests
        if self.adopted:
            return endpoint_alive(url)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.running:
                return False
            try:
                r = requests.get(url.rstrip("/") + "/health", timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1.0)
        return False

    def stop(self, wait_for_port: bool = True) -> bool:
        """Stop the server and wait until it has actually gone.

        Terminating is a request, not an outcome: a server mid-load can take
        seconds to notice, and the previous version neither waited after the
        kill nor checked the port. Returning while the process still held the
        port is what made the next load fail with "a server is already
        running" — against the very process Kestrel had just tried to stop.
        """
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            self._await_port_release() if wait_for_port else None
            return True
        try:
            proc.terminate()
            proc.wait(timeout=6)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=4)
            except Exception:
                pass
        gone = proc.poll() is not None
        if not gone:
            self.on_log("[the server did not exit; it may still hold the port]")
        elif wait_for_port and not self._await_port_release():
            self.on_log("[the server exited but the port is still held]")
            gone = False
        self.on_log("[server stopped]")
        return gone

    def _await_port_release(self, timeout: float = 6.0) -> bool:
        """A closed process can leave the socket briefly in TIME_WAIT."""
        host = self.command_host or "127.0.0.1"
        port = self.command_port
        if not port:
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not port_in_use(host, port, timeout=0.4):
                return True
            time.sleep(0.3)
        return False
