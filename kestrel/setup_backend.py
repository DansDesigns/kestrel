"""Backend installer, invoked by install.sh / install.bat and usable directly.

    python -m kestrel.setup_backend                    scan, install if absent
    python -m kestrel.setup_backend --backend vulkan   pick the accelerator
    python -m kestrel.setup_backend --source --no-rpc  build without clustering
    python -m kestrel.setup_backend --check            report only, change nothing
    python -m kestrel.setup_backend --reinstall        remove and install again
    python -m kestrel.setup_backend --uninstall        remove Kestrel's copy
"""
from __future__ import annotations

import argparse
import sys

from . import llamacpp
from .config import Config


def _clean(value: str, batch: bool) -> str:
    """Make a value safe to embed in a generated script.

    Diagnostics come from other programs, so they can contain anything. In cmd,
    `%` and `!` are expanded and `"` ends the assignment; in sh, a single quote
    ends it. Strip rather than escape — this is a status line, not data.
    """
    out = " ".join(str(value or "").split())
    if batch:
        out = out.replace("%", "pct").replace("!", "").replace('"', "'")
        out = out.replace("^", "").replace("&", "and").replace("<", "").replace(">", "")
    else:
        out = out.replace("'", "")
    return out[:300]


def _emit(config: str, batch: bool) -> int:
    """Print detection results as assignments the installers can evaluate.

    Generating a fragment avoids the delimiter-collapsing that both `for /f` and
    `read` perform, which silently shifts every field after an empty one.
    """
    cfg = Config.load(config or None)
    f = llamacpp.scan(cfg.llama_server_bin, cfg.rpc_bin)
    fields = {
        "HAVE_SERVER": f.server,
        "HAVE_RPC": f.rpc,
        "HAVE_VER": f.version,
        "HAVE_OK": "1" if f.working else "0",
        "HAVE_MANAGED": "1" if f.managed else "0",
        "HAVE_PROBLEM": f.problem,
        "DETECTED": llamacpp.detect_backend(),
    }
    for key, value in fields.items():
        value = _clean(value, batch)
        if batch:
            print(f'set "{key}={value}"')
        else:
            print(f"{key}='{value}'")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kestrel.setup_backend", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", choices=llamacpp.BACKENDS,
                    help="accelerator to build or download for; auto inspects the machine")
    ap.add_argument("--source", action="store_true",
                    help="compile from source rather than downloading a release")
    ap.add_argument("--no-rpc", dest="rpc", action="store_false", default=True,
                    help="omit the RPC backend; the machine cannot then be a cluster worker")
    ap.add_argument("--check", action="store_true", help="report what is present and exit")
    ap.add_argument("--force", action="store_true",
                    help="install even if a working build is already present")
    ap.add_argument("--reinstall", action="store_true",
                    help="remove the Kestrel-managed installation first, then install")
    ap.add_argument("--uninstall", action="store_true",
                    help="remove the Kestrel-managed installation and exit")
    ap.add_argument("--keep-source", action="store_true",
                    help="when removing, leave the source checkout in place")
    ap.add_argument("--no-fallback", dest="fallback", action="store_false", default=True,
                    help="do not retry with a CPU build if the chosen backend will not run")
    ap.add_argument("--emit-batch", action="store_true",
                    help=argparse.SUPPRESS)   # used by install.bat
    ap.add_argument("--emit-sh", action="store_true",
                    help=argparse.SUPPRESS)   # used by install.sh
    ap.add_argument("--config", default="")
    args = ap.parse_args(argv)

    if args.emit_batch or args.emit_sh:
        return _emit(args.config, batch=args.emit_batch)

    cfg = Config.load(args.config or None)
    backend = args.backend
    if backend == "auto":
        backend = llamacpp.detect_backend()
        print(f"detected accelerator: {backend}")
    cfg.llama_backend = args.backend
    cfg.llama_with_rpc = args.rpc

    if args.uninstall:
        removed = llamacpp.uninstall(progress=lambda m: print(f"  {m}"),
                                     include_source=not args.keep_source)
        if removed:
            cfg.llama_server_bin = ""
            cfg.rpc_bin = ""
            cfg.save()
            print(f"Removed {len(removed)} item(s).")
        else:
            other = llamacpp.scan()
            if other.present:
                print(f"\nNote: {other.server} is still on this machine, but Kestrel "
                      "did not install it,\nso it has been left alone. Remove it with "
                      "your package manager if you want it gone.")
        return 0

    found = llamacpp.scan(cfg.llama_server_bin, cfg.rpc_bin)
    if found.present and not found.working:
        print(f"Found {found.server}, but it does not run:")
        print(f"  {found.problem}")
        print("  " + ("This is a Kestrel-managed install and can be replaced."
                      if found.managed else
                      "Kestrel did not install this, so it will be left in place; "
                      "a fresh copy will be installed for Kestrel to use."))
    if found.ok and not args.force and not args.reinstall:
        print(f"llama-server: {found.server}" + (f"  ({found.version})" if found.version else ""))
        print(f"rpc-server:   {found.rpc or 'not present — clustering unavailable'}")
        if not args.check:
            cfg.llama_server_bin = found.server
            if found.rpc:
                cfg.rpc_bin = found.rpc
            cfg.save()
            print("Already installed. Pass --force to reinstall.")
        return 0

    if args.check:
        print("llama.cpp not found.")
        return 1

    if args.source and backend in ("cuda", "hip") and not args.rpc:
        print("note: building without RPC means this machine cannot contribute to a "
              "cluster. Use --rpc if you intend to add it to one later.")

    if args.reinstall:
        print("Removing the existing Kestrel-managed installation first.")
    print(f"Installing llama.cpp — backend {backend}, RPC "
          f"{'enabled' if args.rpc else 'disabled'}, "
          f"{'from source' if args.source else 'prebuilt release'}")
    try:
        found = llamacpp.ensure(cfg, progress=lambda m: print(f"  {m}"),
                                allow_install=True, backend=args.backend,
                                source=args.source, with_rpc=args.rpc,
                                remove_first=args.reinstall)
    except KeyboardInterrupt:
        print("\ncancelled")
        return 130
    except Exception as e:
        print(f"failed: {e}", file=sys.stderr)
        print("You can still point Kestrel at an existing build from the Backend tab.",
              file=sys.stderr)
        return 1

    if not found.present:
        print("installation did not produce a llama-server binary", file=sys.stderr)
        return 1
    if not found.working and args.fallback and backend != "cpu" and not args.source:
        # The goal of an installer is a working backend, not a particular one.
        # A GPU build that will not start is worth nothing; a CPU build always
        # runs, and the accelerator can be revisited from the Backend tab.
        print(f"\nThe {backend} build will not run: {found.problem}")
        print("Retrying with a CPU build, which has no driver or runtime "
              "dependencies.\n")
        try:
            found = llamacpp.ensure(cfg, progress=lambda m: print(f"  {m}"),
                                    allow_install=True, backend="cpu",
                                    source=False, with_rpc=args.rpc,
                                    remove_first=True)
        except Exception as e:
            print(f"fallback failed: {e}", file=sys.stderr)
        if found.working:
            print(f"\nInstalled a CPU build instead: {found.server}")
            print(f"The {backend} build failed on this machine. If you want GPU "
                  "acceleration, check that")
            print(f"the {backend} runtime and drivers are installed, then use the "
                  "Backend tab to retry.")

    if not found.working:
        print(f"\nInstalled to {found.server}, but it still does not run:",
              file=sys.stderr)
        print(f"  {found.problem}", file=sys.stderr)
        if backend != "cpu":
            print("  Try a CPU build, which has no runtime dependencies:",
                  file=sys.stderr)
            print("  python -m kestrel.setup_backend --reinstall --backend cpu",
                  file=sys.stderr)
        else:
            print("  This usually means a missing system library. On Windows, "
                  "install the", file=sys.stderr)
            print("  Microsoft Visual C++ Redistributable; on Linux, check "
                  "libgomp is present.", file=sys.stderr)
        return 1
    print(f"\nllama-server: {found.server}")
    print(f"rpc-server:   {found.rpc or 'not present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
