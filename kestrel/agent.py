"""The agent loop.

Two dialects:

  native — the model emits OpenAI-style tool_calls. Needs a chat template with a
           tools section and costs full JSON schemas in the prompt.
  text   — the model emits a <tool>{...}</tool> block. Works with literally any
           instruct model, costs about a fifth of the prompt tokens, and lets us
           set a stop sequence so generation ends the moment the call is closed.

`auto` picks text below 16k or when the server refuses a tools payload, which is
why a 4k model can run this harness at all.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import persona as personamod, prompts, reasoning, skills as skillmod
from .context import Budget, ContextManager, budget_for
from .llm import ChatResult, LlamaClient, LLMError
from .memory import CAPTURE_PROMPT, MemoryStore, parse_capture
from .tokens import TokenCounter
from .todo import TodoList
from .tools import Registry, ToolResult

def _memory_path(cfg) -> str:
    """Resolve the store location, falling back to the config directory when a
    Config was built directly rather than loaded from disk."""
    from .config import config_dir
    path = (cfg.memory.db_path or "").strip()
    if not path:
        path = str(config_dir() / "memory.db")
        cfg.memory.db_path = path
    return path


TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*(?:</tool>|$)", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json|tool)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class Call:
    name: str
    args: dict
    id: str = ""
    raw: str = ""


def parse_text_calls(text: str) -> list[Call]:
    """Pull tool calls out of free text. Deliberately forgiving — small models
    put the JSON in fences, forget the closing tag, or add a stray comma."""
    out: list[Call] = []
    seen: set[str] = set()
    for rx in (TOOL_RE, FENCE_RE):
        for m in rx.finditer(text):
            blob = m.group(1)
            if blob in seen:
                continue
            obj = _loads(blob)
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
                if isinstance(args, str):
                    args = _loads(args) or {}
                if isinstance(args, dict):
                    seen.add(blob)
                    out.append(Call(str(obj["name"]), args, raw=blob))
        if out:
            return out
    # last resort: a bare JSON object that looks like a call
    stripped = text.strip()
    if stripped.startswith("{"):
        obj = _loads(stripped)
        if isinstance(obj, dict) and obj.get("name") and isinstance(
                obj.get("arguments", obj.get("parameters", {})), dict):
            out.append(Call(str(obj["name"]),
                            obj.get("arguments", obj.get("parameters", {})), raw=stripped))
    return out


def _loads(blob: str) -> Any:
    try:
        return json.loads(blob)
    except ValueError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", blob)          # trailing commas
    fixed = re.sub(r"'(\w+)'\s*:", r'"\1":', fixed)       # single-quoted keys
    try:
        return json.loads(fixed)
    except ValueError:
        return None


def collapse_repeats(text: str) -> str:
    """Fold immediately repeated lines or sentences into one.

    Base and completion models — anything not tuned for chat — will answer a
    greeting with the same sentence several times over. The duplication carries
    no information, so it is not worth showing or speaking.
    """
    if not text:
        return text
    lines, seen_last = [], ""
    for line in text.splitlines():
        norm = " ".join(line.split()).lower()
        if norm and norm == seen_last:
            continue
        lines.append(line)
        seen_last = norm
    out = "\n".join(lines)

    parts = re.split(r"(?<=[.!?])\s+", out.strip())
    if len(parts) > 1:
        deduped, previous = [], ""
        for part in parts:
            norm = " ".join(part.split()).lower()
            if norm and norm == previous:
                continue
            deduped.append(part)
            previous = norm
        out = " ".join(deduped)
    return out.strip()


def strip_calls(text: str) -> str:
    text = TOOL_RE.sub("", text)
    return text.strip()


class Agent:
    """Owns one conversation. Not thread-safe; run it on one worker thread."""

    def __init__(self, cfg, client: LlamaClient, emit: Callable[[str, dict], None] | None = None):
        self.cfg = cfg
        self.client = client
        self.emit = emit or (lambda kind, data: None)
        self.history: list[dict] = []
        self.cancelled = threading.Event()
        self.paused = threading.Event()
        self._recent: list[str] = []       # signatures, for loop detection
        self.skills: list[skillmod.Skill] = []
        self.registry: Registry | None = None
        self.dialect = "text"
        self.budget: Budget = budget_for(4096)
        self.counter = TokenCounter()
        self.ctx: ContextManager | None = None
        self.approver: Callable[[Any, dict], bool] | None = None
        self.memory: MemoryStore | None = None
        self.todo: TodoList | None = None
        self.persona: personamod.Persona | None = None
        self._system_cache = ""
        self._turn_log: list[str] = []

    # -- setup ----------------------------------------------------------------
    def prepare(self) -> dict:
        """Probe the endpoint, load skills, size the budget. Safe to re-run."""
        from .tools import build_registry

        self.counter = TokenCounter(tokenize=self._safe_tokenize)
        self.client.reset_cache()
        ok, why = self.client.health()
        if not ok:
            raise LLMError(why)

        n_ctx = self.client.n_ctx() or self.cfg.runtime.ctx_size or 4096
        self.budget = budget_for(n_ctx, self.cfg.profile_override, self.cfg.thinking)

        if self.cfg.memory.enabled:
            try:
                if self.memory is None:
                    self.memory = MemoryStore(_memory_path(self.cfg), self.cfg.memory_scope())
                self.memory.scope = self.cfg.memory_scope()
            except Exception as e:
                # A broken memory store must not take the session down.
                self.memory = None
                self.emit("error", {"message": f"Long-term memory unavailable: {e}"})
        elif self.memory is not None:
            self.memory.close()
            self.memory = None

        want = self.cfg.tool_dialect
        if want == "auto":
            # Schemas are only worth their tokens on a roomy window.
            self.dialect = "native" if (n_ctx >= 16384 and self.client.supports_tools(self.cfg.model)) else "text"
        else:
            self.dialect = want

        self.todo = TodoList.load(self.cfg.workspace_path())
        self.persona = self._load_persona()
        self.skills = skillmod.discover(self.cfg.skills_dirs)
        self.registry = build_registry(self.cfg, lambda: self.skills,
                                       approver=self._approve,
                                       memory_provider=lambda: self.memory,
                                       todo_provider=lambda: self.todo,
                                       persona_provider=lambda: self.persona)
        self.ctx = ContextManager(self.counter, self.budget, summarizer=self._summarize)
        self._system_cache = ""

        info = {
            "n_ctx": n_ctx,
            "profile": self.budget.profile,
            "dialect": self.dialect,
            "skills": len(self.skills),
            "tools": len(self.registry.tools),
            "model": self.client.model_name(),
            "budget": self.budget,
            "memories": self.memory.count() if self.memory else 0,
            "thinking": self.cfg.thinking.mode if self.cfg.thinking.enabled else "off",
            "persona": (self.persona.name if self.persona and self.persona.any_content()
                        else ""),
            "todo": self.todo,
        }
        self.emit("ready", info)
        return info

    def _safe_tokenize(self, text: str) -> int:
        return self.client.tokenize(text)

    def _approve(self, tool, args) -> bool:
        if self.approver is None:
            return True
        return bool(self.approver(tool, args))

    # Some chat templates refuse a conversation that does not open with a system
    # message, and others reject two messages from the same role in a row. The
    # main loop always satisfies both; these one-off helper calls did not.
    HELPER_SYSTEM = "You are a careful assistant. Follow the instruction exactly."

    def _helper_messages(self, prompt: str) -> list[dict]:
        return [{"role": "system", "content": self.HELPER_SYSTEM},
                {"role": "user", "content": prompt}]

    def _summarize(self, blob: str, cap: int) -> str:
        """Ask the model to compact old turns. Skipped on tiny windows, where the
        generation would cost more than the text it saves."""
        if self.budget.n_ctx < 6000:
            return ""
        prompt = prompts.SUMMARY_PROMPT.format(cap=cap, blob=blob[:20000])
        try:
            res = self.client.chat(
                self._helper_messages(prompt),
                temperature=0.1, max_tokens=min(cap, 400), stream=False,
                model=self.cfg.model,
            )
            return res.content
        except Exception:
            return ""

    def _load_persona(self):
        if self.cfg.persona_file:
            p = personamod.load_file(self.cfg.persona_file)
            if p is not None:
                return p
        if self.cfg.persona.strip():
            return personamod.parse(self.cfg.persona)
        return None

    def _persona_text(self) -> str:
        if self.persona is None or not self.persona.any_content():
            return ""
        level = (self.cfg.persona_level if self.cfg.persona_level >= 0
                 else self.budget.verbosity)
        # Voice gets a slice of the system allowance, never the whole file.
        chars = self.counter.budget_chars(max(24, int(self.budget.system * 0.22)))
        return self.persona.compile(level, chars)

    def _extra_payload(self) -> dict:
        extra = dict(self.cfg.sampling.payload())
        extra.update(self.cfg.thinking.payload())
        return extra

    def _memory_block(self, query: str) -> str:
        if self.memory is None or not self.cfg.memory.enabled or self.ctx is None:
            return ""
        try:
            block, kept = self.memory.block(query, self.counter, self.budget.memory,
                                            limit=self.cfg.memory.recall)
        except Exception:
            return ""
        if kept:
            self.emit("memory_recall", {"memories": kept})
        return block

    def system_prompt(self) -> str:
        if self._system_cache:
            return self._system_cache
        assert self.registry is not None
        listing = self.registry.listing(self.budget.verbosity) if self.dialect == "text" \
            else self.registry.listing(0)
        self._system_cache = prompts.build_system(
            workspace=str(self.cfg.workspace_path()),
            tool_listing=listing,
            skills=self.skills,
            budget=self.budget,
            dialect=self.dialect,
            persona=self._persona_text(),
            has_plan_tools=bool(self.registry and "plan" in self.registry.tools),
        )
        return self._system_cache

    # -- running --------------------------------------------------------------
    def reload_persona(self):
        """Re-read the persona and rebuild the prompt.

        The system prompt is cached, so clearing it is all that is needed for
        the next turn to speak in the new voice — no reconnection, and the
        conversation so far is kept.
        """
        self.persona = self._load_persona()
        self._system_cache = ""
        return self.persona

    def reload_skills(self) -> list:
        """Re-read the skill folders and rebuild the system prompt.

        Cheap enough to run whenever the folders change: the prompt is cached,
        so clearing it is all that is needed for the next turn to see them.
        """
        self.skills = skillmod.discover(self.cfg.skills_dirs)
        self._system_cache = ""
        return self.skills

    def load_history(self, messages: list[dict], digest: str = "") -> None:
        """Adopt a saved conversation. The transcript is restored as-is and will
        be re-fitted to the budget on the next turn, so a session saved on a
        large window still opens on a small one."""
        self.history = [dict(m) for m in messages]
        if self.ctx is not None:
            self.ctx.digest = digest
            self.ctx.compactions = 0

    def reset(self) -> None:
        self.history.clear()
        if self.ctx:
            self.ctx.digest = ""
            self.ctx.compactions = 0

    def cancel(self) -> None:
        self.cancelled.set()
        self.paused.clear()

    def pause(self) -> None:
        self.paused.set()

    def resume(self) -> None:
        self.paused.clear()

    def _wait_if_paused(self) -> None:
        """Hold between steps. Pausing mid-generation would abandon a partial
        tool call, so the check sits at the step boundary where stopping is
        clean and the plan can be edited safely."""
        if not self.paused.is_set():
            return
        self.emit("paused", {"paused": True})
        while self.paused.is_set() and not self.cancelled.is_set():
            time.sleep(0.15)
        self.emit("paused", {"paused": False})

    def run(self, user_text: str) -> str:
        if self.registry is None or self.ctx is None:
            self.prepare()
        assert self.registry is not None and self.ctx is not None

        self.cancelled.clear()
        self.history.append({"role": "user", "content": user_text})
        self._turn_log = [f"user: {user_text}"]
        self._recent = []
        memory_block = self._memory_block(user_text)
        self._autoplan(user_text)
        final = ""

        for step in range(1, self.cfg.max_steps + 1):
            if self.cancelled.is_set():
                self.emit("cancelled", {})
                return final or "(stopped)"
            self._wait_if_paused()
            if self.cancelled.is_set():
                self.emit("cancelled", {})
                return final or "(stopped)"

            plan_block = (self.todo.render(self.counter, self.budget.plan)
                          if self.todo else "")
            messages = self.ctx.assemble(self.system_prompt(), self.history,
                                         memory_block, plan_block)
            self.emit("context", {"usage": self.ctx.usage, "budget": self.budget,
                                  "compactions": self.ctx.compactions, "step": step})
            if self.todo:
                self.emit("todo", {"todo": self.todo})

            tools = self.registry.schemas() if self.dialect == "native" else None
            stop = [prompts.TOOL_BLOCK_CLOSE] if self.dialect == "text" else None
            self.emit("step", {"step": step, "max": self.cfg.max_steps})

            try:
                res: ChatResult = self.client.chat(
                    messages, tools=tools, model=self.cfg.model,
                    temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                    max_tokens=self.budget.output, stop=stop, stream=True,
                    on_token=lambda t: self.emit("token", {"text": t}),
                    on_reasoning=lambda t: self.emit("thinking", {"text": t}),
                    cancel=self.cancelled.is_set,
                    extra=self._extra_payload(),
                )
            except LLMError as e:
                self.emit("error", {"message": str(e)})
                return f"The model endpoint failed: {e}"

            self.emit("gen", {"tps": res.tokens_per_sec, "tokens": res.completion_tokens})

            content, trace = reasoning.merge(res)
            res.content = content
            if trace:
                self.emit("thinking_done", {"text": trace,
                                            "tokens": self.counter.count(trace)})
                if reasoning.truncated(trace, content, res.finish_reason):
                    msg = ("The model spent its entire generation budget thinking and "
                           "produced no answer. Cap the trace with a thinking budget, "
                           "or switch thinking off, in the Params tab.")
                    self.emit("error", {"message": msg})
                    self.history.append({"role": "assistant", "content": msg})
                    return msg

            calls = self._extract(res)

            if not calls:
                # Plain prose ends the turn.
                answer = collapse_repeats(strip_calls(content).strip())
                if not answer:
                    answer = "(the model returned nothing usable)"
                self.history.append({"role": "assistant", "content": answer})
                self._turn_log.append(f"assistant: {answer}")
                self.emit("assistant", {"text": answer})
                self._capture()
                return answer

            self._record_assistant(content, calls)

            for call in calls:
                if self.cancelled.is_set():
                    break
                self.emit("tool_call", {"name": call.name, "args": call.args})
                if call.name == "finish":
                    final = collapse_repeats(
                        str(call.args.get("answer") or "").strip()) or "Done."
                    self.history.append({"role": "assistant", "content": final})
                    self._turn_log.append(f"assistant: {final}")
                    self.emit("assistant", {"text": final})
                    self._capture()
                    return final
                signature = f"{call.name}:{json.dumps(call.args, sort_keys=True)[:200]}"
                repeats = self._recent[-3:].count(signature)
                self._recent.append(signature)
                if repeats >= 2:
                    # Small models get stuck repeating a call verbatim. Running
                    # it a third time cannot teach them anything the first two
                    # did not, so answer with the observation instead.
                    result = ToolResult(
                        f"You have already called {call.name} with exactly these "
                        "arguments twice, and the result will not change. Either "
                        "do something different, or call finish with what you have.",
                        ok=False)
                    self.emit("loop", {"name": call.name, "count": repeats + 1})
                    if repeats >= 3:
                        msg = (f"Stopped: {call.name} was called identically "
                               f"{repeats + 1} times without progress.")
                        self.emit("assistant", {"text": msg})
                        self.history.append({"role": "assistant", "content": msg})
                        return msg
                else:
                    result = self.registry.call(call.name, call.args)
                shown = self._spool(call.name, result)
                if self.todo:
                    if call.name in ("plan", "todo", "plan_add"):
                        self.emit("todo", {"todo": self.todo})
                    else:
                        self.todo.stale_steps += 1
                        shown += self.todo.nudge()
                # Only that a tool ran, never what it returned. Feeding results
                # into the capture pass is what fills memory with the date, the
                # contents of a directory, and the text of whatever was read.
                self._turn_log.append(f"[used {call.name}]")
                self.emit("tool_result", {"name": call.name, "ok": result.ok,
                                          "text": result.display, "shown": shown})
                self._record_tool(call, shown)

        msg = (f"Stopped after {self.cfg.max_steps} steps without finishing. "
               "Raise the step limit in Settings, or narrow the task.")
        self.emit("assistant", {"text": msg})
        self.history.append({"role": "assistant", "content": msg})
        return msg

    # -- planning --------------------------------------------------------------
    def _autoplan(self, user_text: str) -> None:
        """Seed the checklist for a fresh multi-step task.

        The model is asked to plan the same way it would be asked anything else,
        so the checklist reflects its own reading of the task. Skipped when a
        plan is already in flight, when the window is too small to spend a
        generation on it, or when the request is plainly a single action.
        """
        if (self.todo is None or not self.cfg.todo_enabled or not self.cfg.auto_plan
                or self.budget.n_ctx < 6000):
            return
        if self.todo.items and not self.todo.complete:
            return          # a plan is already running; let the model revise it
        if len(user_text.split()) < 6:
            return          # "list the files" does not need a checklist
        # Streamed, so the checklist fills in line by line rather than
        # appearing all at once after a silent pause.
        buffer: list[str] = []
        drafted: list[str] = []
        title = " ".join(user_text.split())[:60]

        def on_token(chunk: str) -> None:
            buffer.append(chunk)
            text = "".join(buffer)
            if "\n" not in text:
                return
            complete, _, rest = text.rpartition("\n")
            buffer[:] = [rest]
            for line in complete.splitlines():
                if line.strip():
                    drafted.append(line)
            if drafted:
                self.todo.set_plan(drafted, title=title)
                self.emit("todo", {"todo": self.todo})

        try:
            res = self.client.chat(
                self._helper_messages(prompts.PLAN_PROMPT % user_text[:2000]),
                temperature=0.2, max_tokens=300, stream=True, model=self.cfg.model,
                extra=self.cfg.thinking.payload(), on_token=on_token,
                cancel=self.cancelled.is_set)
        except Exception:
            self.todo.clear()
            return
        visible = collapse_repeats(reasoning.split(res.content)[0])
        steps = [ln for ln in visible.splitlines() if ln.strip()]
        if len(steps) < 2:
            self.todo.clear()   # one step is not a plan; do not clutter the prompt
            self.emit("todo", {"todo": self.todo})
            return
        self.todo.set_plan(visible, title=title)
        self.emit("todo", {"todo": self.todo})

    # -- memory ---------------------------------------------------------------
    def _capture(self) -> None:
        """Ask the model what from this task is worth keeping.

        Skipped on very small windows, where the extra generation costs more
        than the memories are worth. The `remember` tool still works there.
        """
        if (self.memory is None or not self.cfg.memory.auto_capture
                or self.budget.n_ctx < 6000 or len(self._turn_log) < 2):
            return
        blob = "\n".join(self._turn_log)[:12000]
        try:
            res = self.client.chat(
                self._helper_messages(CAPTURE_PROMPT % (5, blob)),
                temperature=0.1, max_tokens=400, stream=False, model=self.cfg.model)
        except Exception:
            return
        saved = []
        rejected = 0
        for item in parse_capture(reasoning.split(res.content)[0])[:5]:
            if item["importance"] < 3:
                rejected += 1
                continue          # auto-capture keeps only what it rated useful
            try:
                mid, created = self.memory.remember(
                    item["text"], item["kind"], item["importance"], source="capture")
                if created:
                    saved.append((mid, item["text"]))
            except ValueError:
                rejected += 1     # failed the durability filter
            except Exception:
                rejected += 1
        if rejected:
            self.emit("memory_rejected", {"count": rejected})
        if saved:
            self.emit("memory_saved", {"items": saved})
            try:
                self.memory.prune(self.cfg.memory.max_items)
            except Exception:
                pass

    # -- helpers --------------------------------------------------------------
    def _extract(self, res: ChatResult) -> list[Call]:
        if res.tool_calls:
            out = []
            for tc in res.tool_calls:
                fn = tc.get("function") or {}
                args = _loads(fn.get("arguments") or "{}") or {}
                if not isinstance(args, dict):
                    args = {}
                out.append(Call(str(fn.get("name") or ""), args, id=str(tc.get("id") or "")))
            return [c for c in out if c.name]
        return parse_text_calls(res.content)

    def _record_assistant(self, content: str, calls: list[Call]) -> None:
        if self.dialect == "native":
            msg: dict = {"role": "assistant", "content": strip_calls(content) or None,
                         "tool_calls": [{"id": c.id or f"call_{i}", "type": "function",
                                         "function": {"name": c.name,
                                                      "arguments": json.dumps(c.args)}}
                                        for i, c in enumerate(calls)]}
        else:
            body = content.strip()
            if calls and prompts.TOOL_BLOCK_OPEN not in body:
                body = (body + "\n" if body else "") + \
                    f"{prompts.TOOL_BLOCK_OPEN}\n{json.dumps({'name': calls[0].name, 'arguments': calls[0].args})}\n{prompts.TOOL_BLOCK_CLOSE}"
            elif calls and prompts.TOOL_BLOCK_CLOSE not in body:
                body += f"\n{prompts.TOOL_BLOCK_CLOSE}"
            msg = {"role": "assistant", "content": body}
        self.history.append(msg)

    def _record_tool(self, call: Call, text: str) -> None:
        if self.dialect == "native":
            self.history.append({"role": "tool", "tool_call_id": call.id or "call_0",
                                 "name": call.name, "content": text})
        else:
            self.history.append({"role": "user", "content": f"[{call.name} result]\n{text}"})

    def _spool(self, name: str, result: ToolResult) -> str:
        """Keep a preview in context; put the rest on disk.

        This is the single biggest saving in the harness. A build log or a long
        file costs a fixed ~150 tokens no matter how big it is.
        """
        assert self.registry is not None and self.ctx is not None
        preview = self.ctx.clip(result.text, self.budget.tool_preview)
        if preview == result.text:
            return result.text
        path = self.registry.spool(result.full or result.text, label=name)
        try:
            rel = path.relative_to(self.registry.workspace)
        except ValueError:
            rel = path
        return (f"{preview}\n[full output saved to {rel} "
                f"({len(result.full or result.text)} chars) — read_file it if you need more]")
