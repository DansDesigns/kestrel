"""Headless mode: the same agent loop without Qt. Useful over SSH, and for
checking a setup before opening the window."""
from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .config import Config
from .llm import LlamaClient, LLMError


def _approver(mode: str):
    def ask(tool, args) -> bool:
        preview = ", ".join(f"{k}={str(v)[:60]}" for k, v in args.items())
        sys.stdout.write(f"\n  run {tool.name}({preview})? [y/N] ")
        sys.stdout.flush()
        try:
            return input().strip().lower().startswith("y")
        except (EOFError, KeyboardInterrupt):
            return False
    return None if mode == "never" else ask


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kestrel --cli")
    ap.add_argument("--cli", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", default="")
    ap.add_argument("--url", default="", help="override the server URL")
    ap.add_argument("--workspace", default="")
    ap.add_argument("--dialect", choices=["auto", "native", "text"], default="")
    ap.add_argument("--profile", choices=["", "nano", "small", "standard", "large"], default="")
    ap.add_argument("--yes", action="store_true", help="skip approval prompts")
    ap.add_argument("--model", default="", help="GGUF path to load (implies --serve)")
    ap.add_argument("--serve", action="store_true",
                    help="start llama-server before running, and stop it after")
    ap.add_argument("--think", choices=["auto", "on", "off"], default="",
                    help="reasoning mode")
    ap.add_argument("--think-budget", type=int, default=-1,
                    help="cap the reasoning trace, in tokens")
    ap.add_argument("--temp", type=float, default=-1.0)
    ap.add_argument("--preset", choices=["deterministic", "precise", "balanced", "creative"],
                    default="", help="sampling preset")
    ap.add_argument("--no-memory", action="store_true", help="disable long-term memory")
    ap.add_argument("--memories", action="store_true",
                    help="print what is remembered and exit")
    ap.add_argument("prompt", nargs="*", help="run one task and exit")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config or None)
    if args.url:
        cfg.server_url = args.url
    if args.workspace:
        cfg.workspace = args.workspace
    if args.dialect:
        cfg.tool_dialect = args.dialect
    if args.profile:
        cfg.profile_override = args.profile
    if args.yes:
        cfg.approval = "never"
    if args.model:
        cfg.model_path = args.model
    if args.think:
        cfg.thinking.mode = args.think
    if args.think_budget >= 0:
        cfg.thinking.budget = args.think_budget
    if args.preset:
        cfg.sampling.preset(args.preset)
    if args.temp >= 0:
        cfg.sampling.temperature = args.temp
    if args.no_memory:
        cfg.memory.enabled = False

    if args.memories:
        from .memory import MemoryStore
        store = MemoryStore(cfg.memory.db_path, cfg.memory_scope())
        items = store.all()
        print(f"{len(items)} memories ({cfg.memory.db_path})\n")
        for m in items:
            print(f"  #{m.id:<4} {m.line()}")
        store.close()
        return 0

    server = None
    # Auto-start when nothing is serving, matching the interface. An endpoint
    # that already answers is left alone.
    if not (args.serve or args.model) and cfg.auto_start_server:
        from .cluster import endpoint_alive
        if (cfg.model_path and cfg.llama_server_bin
                and not endpoint_alive(cfg.server_url)
                and not endpoint_alive(cfg.runtime.url())):
            print(f"nothing serving on {cfg.runtime.url()} — starting llama-server")
            args.serve = True

    if args.serve or args.model:
        from .cluster import ServerProcess
        server = ServerProcess(on_log=lambda ln: print("  |", ln))
        try:
            spawned = server.start(cfg)
        except Exception as e:
            print(f"Could not start llama-server: {e}")
            return 1
        cfg.server_url = cfg.runtime.url()
        if spawned:
            print(f"waiting for {cfg.server_url} …")
        if not server.wait_healthy(cfg.server_url) or not (
                server.adopted or server.running):
            print("llama-server did not start:\n" + server.failure_summary())
            server.stop()
            return 1

    client = LlamaClient(cfg.server_url, cfg.api_key)
    # Stream the model's prose but swallow the tool block itself — the call is
    # reported on its own line a moment later, so echoing the JSON is just noise.
    state = {"raw": "", "printed": 0}

    def flush() -> None:
        raw = state["raw"]
        cut = raw.find("<tool")
        visible = raw if cut < 0 else raw[:cut]
        if len(visible) > state["printed"]:
            sys.stdout.write(visible[state["printed"]:])
            sys.stdout.flush()
            state["printed"] = len(visible)

    def newline() -> None:
        if state["printed"]:
            sys.stdout.write("\n")
            sys.stdout.flush()
        state["raw"] = ""
        state["printed"] = 0

    def emit(kind: str, data: dict) -> None:
        if kind == "token":
            state["raw"] += data["text"]
            flush()
            return
        if kind == "step":
            newline()
        elif kind == "assistant":
            newline()
            print(data["text"] + "\n")
        elif kind == "tool_call":
            newline()
            if data["name"] == "finish":
                return
            preview = ", ".join(f"{k}={str(v)[:70]}" for k, v in data["args"].items())
            print(f"  \u2192 {data['name']}({preview})")
        elif kind == "tool_result":
            mark = "    " if data["ok"] else "  ! "
            for ln in data["shown"].splitlines()[:6]:
                print(f"{mark}{ln[:160]}")
        elif kind == "thinking_done":
            newline()
            print(f"  [thought {data['tokens']} tokens, not resent]")
        elif kind == "memory_recall":
            newline()
            print("  [recalled] " + "; ".join(m.text[:60] for m in data["memories"][:3]))
        elif kind == "memory_saved":
            newline()
            for _mid, text in data["items"]:
                print(f"  [remembered] {text[:80]}")
        elif kind == "context":
            u = data["usage"]
            print(f"  [ctx {u.used}/{u.n_ctx}, {u.free} free, "
                  f"{data['compactions']} compaction(s)]")
        elif kind == "error":
            newline()
            print(f"  ! {data['message']}")

    agent = Agent(cfg, client, emit=emit)
    agent.approver = _approver(cfg.approval)
    try:
        info = agent.prepare()
    except LLMError as e:
        print(f"Cannot reach {cfg.server_url}: {e}")
        print("Start llama-server, or pass --url.")
        return 1

    print(f"Kestrel  model={info['model'] or 'unknown'}  ctx={info['n_ctx']} "
          f"({info['profile']})  dialect={info['dialect']}  "
          f"tools={info['tools']}  skills={info['skills']}")
    b = info["budget"]
    print(f"budget:  system {b.system}  memory {b.memory}  history {b.history}  "
          f"output {b.output}  tool preview {b.tool_preview}")
    print(f"memory:  {info.get('memories', 0)} stored  |  thinking: {info.get('thinking')}")

    if args.prompt:
        agent.run(" ".join(args.prompt))
        if server:
            server.stop()
        return 0

    print("Type a task, or /reset, /memories, /quit.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            if server:
                server.stop()
            return 0
        if line == "/memories":
            if agent.memory:
                for m in agent.memory.all():
                    print(f"  #{m.id:<4} {m.line()}")
            else:
                print("  (memory disabled)")
            continue
        if line == "/reset":
            agent.reset()
            print("(cleared)")
            continue
        try:
            agent.run(line)
        except KeyboardInterrupt:
            agent.cancel()
            print("\n(cancelled)")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
