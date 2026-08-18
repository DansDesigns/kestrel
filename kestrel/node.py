"""Run this on every machine that should contribute memory and compute.

    python -m kestrel.node --mem 8192

It starts llama.cpp's rpc-server and broadcasts a small UDP beacon so the
Kestrel head node can find it without you typing IP addresses.

The RPC protocol is unauthenticated. Run this on a trusted LAN only.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading

from . import llamacpp
from .cluster import (BEACON_MAGIC, binary_build, broadcast, broadcast_targets,
                      firewall_hint)
from .config import Config


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kestrel-node", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=50052, help="rpc-server port")
    ap.add_argument("--host", default="0.0.0.0", help="interface to bind (0.0.0.0 for LAN)")
    ap.add_argument("--mem", type=int, default=0,
                    help="memory pool to advertise, in MB. Set this to the free VRAM "
                         "or RAM you want to donate.")
    ap.add_argument("--label", default=socket.gethostname(), help="friendly name")
    ap.add_argument("--bin", default="", help="path to rpc-server / ggml-rpc-server")
    ap.add_argument("--cache", action="store_true",
                    help="enable the local tensor cache (much faster reloads)")
    ap.add_argument("--beacon-port", type=int, default=50051)
    ap.add_argument("--no-beacon", action="store_true")
    ap.add_argument("--install", action="store_true",
                    help="install llama.cpp with RPC support if it is not found")
    ap.add_argument("--config", default="")
    args = ap.parse_args(argv)

    # The same search the rest of Kestrel uses, and the same configuration file,
    # so a worker picks up whatever the installer put in place without being
    # told where it went.
    cfg = Config.load(args.config or None)
    binary = llamacpp.find_rpc(args.bin or cfg.rpc_bin)
    if not binary and args.install:
        print("no rpc-server found — installing llama.cpp with RPC support")
        try:
            llamacpp.build_from_source(cfg.llama_backend,
                                       progress=lambda m: print(f"  {m}"),
                                       with_rpc=True)
        except Exception as e:
            print(f"install failed: {e}", file=sys.stderr)
        binary = llamacpp.find_rpc()
    if not binary:
        print("Could not find rpc-server.\n"
              "  It ships with llama.cpp built using -DGGML_RPC=ON.\n"
              "  Run the installer on this machine:  bash install.sh --llama --source\n"
              "  or pass a path:                     --bin /path/to/rpc-server\n"
              "  or let this command build it:       --install", file=sys.stderr)
        return 2

    ok, detail = llamacpp.verify(binary)
    if not ok:
        print(f"Found {binary} but it does not run: {detail}", file=sys.stderr)
        return 2
    if binary != cfg.rpc_bin:
        cfg.rpc_bin = binary
        try:
            cfg.save()      # so the head node and future runs agree on the path
        except Exception:
            pass
    print(f"using {binary}")

    cmd = [binary, "-H", args.host, "-p", str(args.port)]
    if args.mem:
        cmd += ["-m", str(args.mem)]
    if args.cache:
        cmd.append("-c")

    build = binary_build(binary)
    print(f"rpc-server: {binary}" + (f"  (llama.cpp build {build})" if build else ""))
    print("Every machine in a cluster must run the same llama.cpp build — the "
          "RPC protocol changes between them, and a mismatch connects and then "
          "drops straight away.")
    hint = firewall_hint(args.port, args.beacon_port)
    if hint:
        print()
        print(hint)
    print()

    stop = threading.Event()
    if not args.no_beacon:
        payload = {
            "magic": BEACON_MAGIC,
            "host": local_ip(),
            "rpc_port": args.port,
            "label": args.label,
            "mem_mb": args.mem,
            "pid": os.getpid(),
        }
        threading.Thread(target=broadcast,
                         args=(args.beacon_port, payload, 3.0, stop, print),
                         daemon=True).start()
        print(f"beacon: {payload['host']}:{args.port} as '{args.label}' "
              f"({args.mem or 'auto'} MB) on udp/{args.beacon_port}")

    print("$ " + " ".join(cmd), flush=True)
    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
        return proc.returncode or 0
    except KeyboardInterrupt:
        return 0
    except OSError as e:
        print(f"Could not start rpc-server: {e}", file=sys.stderr)
        return 2
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
