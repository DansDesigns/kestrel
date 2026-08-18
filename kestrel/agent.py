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
from . import handover as handovermod
from .thoughts import ThoughtLog
from .todo import DONE, TodoList
from .tools import Registry, ToolResult, build_registry

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


def find_tool_blobs(text: str) -> list[tuple[int, int, str]]:
    """Locate every <tool> block, as (start, end, json).

    Scanned rather than matched by expression. A regular expression has to
    decide where the JSON ends, and `{.*?}` ends at the first closing brace —
    which is inside `arguments`, not after it. That only ever worked because a
    trailing `</tool>` forced it wider, so a model that omitted the closing tag
    had its calls rendered into the chat as text instead of being run.

    Braces are counted instead, ignoring those inside strings, which handles a
    missing closing tag, several blocks in one message, and a block cut short
    by the token limit.
    """
    found: list[tuple[int, int, str]] = []
    for match in re.finditer(r"<tool>", text, re.IGNORECASE):
        cursor = match.end()
        while cursor < len(text) and text[cursor] in " \t\r\n`":
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            continue
        depth, in_string, escaped = 0, False, False
        end = -1
        for i in range(cursor, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            continue                      # truncated: not a call yet
        stop = end
        tail = re.match(r"\s*(?:```)?\s*</tool>", text[end:], re.IGNORECASE)
        if tail:
            stop = end + tail.end()
        found.append((match.start(), stop, text[cursor:end]))
    return found


CANVAS_TAG_RE = re.compile(
    r"<canvas(?:\s+[^>]*)?>\s*(.*?)\s*</canvas>", re.DOTALL | re.IGNORECASE)
CANVAS_OPEN_RE = re.compile(r"<canvas(?:\s+[^>]*)?>\s*(.*)$", re.DOTALL | re.IGNORECASE)


def parse_canvas_tags(text: str) -> list[Call]:
    """Accept <canvas>…</canvas> as a canvas_write.

    Some models decide the canvas must be an XML tag rather than a tool, write
    the file inside one, and then call canvas_save on an empty buffer. The
    intent is unmistakable and the content is right there, so it is taken as
    the write it was meant to be rather than left on the floor.
    """
    out = []
    for match in CANVAS_TAG_RE.finditer(text):
        body = _strip_fence(match.group(1))
        if body.strip():
            out.append(Call("canvas_write", {"text": body}, raw=match.group(0)))
    if not out:
        # An unclosed tag, which is the usual shape when the reply was cut off.
        opened = CANVAS_OPEN_RE.search(text)
        if opened and len(opened.group(1).strip()) > 40:
            out.append(Call("canvas_write", {"text": _strip_fence(opened.group(1))},
                            raw=opened.group(0)))
    return out


def _strip_fence(body: str) -> str:
    body = body.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", body)
        body = re.sub(r"\n?```\s*$", "", body)
    return body


def parse_text_calls(text: str) -> list[Call]:
    """Pull tool calls out of free text. Deliberately forgiving — small models
    put the JSON in fences, forget the closing tag, or add a stray comma."""
    out: list[Call] = []
    seen: set[str] = set()
    for _start, _stop, blob in find_tool_blobs(text):
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
    tagged = parse_canvas_tags(text)
    if tagged:
        return tagged
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


# Conversational framing, wherever in the line it starts. Anchoring only at the
# end missed "Sure! Here's the plan:" and "Let me know if you'd like changes".
PLAN_NOISE = re.compile(
    r"^\s*(sure|certainly|of course|okay|ok|alright|here(?:'s| is| are)|"
    r"i will|i'll|let me know|let me|feel free|below (?:is|are)|"
    r"the plan|this plan|plan|steps?|breakdown|hope this|anything else)\b",
    re.I)
PLAN_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|step\s*\d+[:.)]?)\s*", re.I)


FENCE_BLOCK_RE = re.compile(
    r"```([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)
LIFT_MIN_LINES = 4
LIFT_MIN_CHARS = 160
# Prose formats that are not code and belong where they were written.
PROSE_FENCES = {"", "text", "txt", "md", "markdown", "log", "output", "console"}


def lift_code(answer: str, buffer) -> tuple[str, int]:
    """Move code out of a reply and into the canvas. Returns (reply, blocks).

    Telling a model where to put code only works when it listens. Some do not,
    and the result is a wall of source in the chat that cannot be edited, run
    or saved — while the canvas beside it sits empty. So the harness moves it:
    the reply keeps its explanation, the code goes where it can be worked on.

    Prose fences are left alone, and so is anything short — a two-line snippet
    quoted mid-sentence is part of the explanation, not a file.
    """
    blocks: list[tuple[str, str]] = []

    def take(match) -> str:
        language, body = match.group(1).strip().lower(), match.group(2)
        stripped = body.strip("\n")
        if language in PROSE_FENCES and stripped.count("\n") + 1 < 12:
            return match.group(0)
        if (stripped.count("\n") + 1 < LIFT_MIN_LINES
                and len(stripped) < LIFT_MIN_CHARS):
            return match.group(0)
        blocks.append((language, stripped))
        lines = stripped.count("\n") + 1
        return f"[{lines} lines of {language or 'code'} written to the canvas]"

    rewritten = FENCE_BLOCK_RE.sub(take, answer)
    if not blocks:
        return answer, 0
    if len(blocks) == 1:
        language, body = blocks[0]
    else:
        # Several blocks in one reply are usually one file in pieces, or a set
        # of small files; they are kept in order with a rule between them.
        language = blocks[0][0]
        body = "\n\n".join(
            f"# ---- part {n} ----\n{code}" if language in ("python", "py", "sh")
            else f"/* ---- part {n} ---- */\n{code}"
            for n, (_lang, code) in enumerate(blocks, 1))
    existing = buffer.text.strip()
    if existing and body.strip() in existing:
        return rewritten, 0          # the model already put it there
    buffer.set(body, language=language or buffer.language)
    return rewritten, len(blocks)


def clean_plan_lines(text: str) -> list[str]:
    """Turn whatever the model wrote into a list of steps.

    Models introduce a list before giving it, number it in several styles, and
    close with a summary. Insisting on one bare step per line throws away a
    perfectly good plan because of its packaging, which is why some models
    appeared not to plan at all.
    """
    out: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip().strip("`")
        if not line or PLAN_NOISE.match(line):
            continue
        if line.startswith("#") or line.startswith(">"):
            continue
        line = PLAN_BULLET.sub("", line).strip()
        line = re.sub(r"^\[[ x>!]\]\s*", "", line)      # a checkbox, if given
        line = line.strip(" .;")
        if len(line) < 3 or len(line.split()) > 40:
            continue
        # A step is an instruction, not a remark about the plan.
        if PLAN_NOISE.match(line) or line.endswith((":", "?")):
            continue
        if line.lower() in {s.lower() for s in out}:
            continue
        out.append(line[:160])
    return out[:12]


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
    """Remove tool blocks from what the reader sees.

    Spans come from the same scanner that finds the calls, so anything that ran
    is also hidden — and a block that was cut short mid-JSON is hidden too,
    rather than being left on screen as a wall of escaped quotes.
    """
    spans = find_tool_blobs(text)
    if spans:
        out, cursor = [], 0
        for start, stop, _blob in spans:
            out.append(text[cursor:start])
            cursor = stop
        out.append(text[cursor:])
        text = "".join(out)
    text = TOOL_RE.sub("", text)
    # A block truncated by the token limit leaves an opening tag behind.
    text = re.sub(r"<tool>\s*\{.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?tool>", "", text, flags=re.IGNORECASE)
    text = CANVAS_TAG_RE.sub("", text)
    text = CANVAS_OPEN_RE.sub("", text)
    text = re.sub(r"</?canvas[^>]*>", "", text, flags=re.IGNORECASE)
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
        self.thoughts = ThoughtLog()       # replaced with the project's log on prepare
        self.handover = handovermod.Handover()
        self._prose_streak = 0
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
        self.thoughts = ThoughtLog.load(self.cfg.workspace_path())
        self.handover = handovermod.load(self.cfg.workspace_path())
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
            "tool_list": [
                {"name": t.name, "summary": t.summary, "danger": t.danger,
                 "signature": t.signature(), "detail": t.detail,
                 "params": [(p.name, p.type, p.required, p.desc) for p in t.params]}
                for t in self.registry.tools.values()],
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
            has_canvas=bool(self.registry and "canvas_write" in self.registry.tools),
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

    def name_session(self, label: str) -> None:
        """Tell the thinking log which conversation it is recording."""
        self.thoughts.start_session(label)

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
        self.thoughts.start_task("")      # a new conversation recalls nothing yet
        if self.ctx:
            self.ctx.digest = ""
            self.ctx.compactions = 0

    def forget_project_memories(self) -> int:
        """Drop what was learnt about this project, keeping the rest.

        Starting again means starting again: the project tier goes, while what
        Kestrel knows about the machine and about the person stays, because
        neither of those became untrue.
        """
        if self.memory is None:
            return 0
        from .memory import PROJECT
        return self.memory.clear_tier(PROJECT)

    def rebind(self, workspace: str) -> None:
        """Point the agent at a different project.

        Clearing the transcript is not enough. The checklist, the thinking log
        and the memory scope are all tied to a folder, and leaving them attached
        to the previous one is how a new project arrives already carrying
        another project's plan, reasoning and recollections.
        """
        self.cfg.workspace = workspace
        self.reset()
        self.todo = TodoList.load(workspace) if self.cfg.todo_enabled else None
        self.thoughts = ThoughtLog.load(workspace)
        self.handover = handovermod.load(workspace)
        if self.memory is not None:
            self.memory.scope = self.cfg.memory_scope()
        # The registry closes over the workspace for its file sandbox, so it is
        # rebuilt rather than left pointing at the previous project's folder.
        self.registry = build_registry(self.cfg, lambda: self.skills,
                                       approver=self._approve,
                                       memory_provider=lambda: self.memory,
                                       todo_provider=lambda: self.todo,
                                       persona_provider=lambda: self.persona)
        self._system_cache = ""

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
        self._prose_streak = 0
        self.thoughts.start_task(user_text)
        briefing = self._plan_briefing()
        thoughts = self._thought_block()
        # Only at the start of a conversation: mid-conversation the transcript
        # already says everything the handover would.
        resume = (self.handover.block()
                  if len(self.history) <= 1 and not handovermod.stale(self.handover)
                  else "")
        opening = "\n\n".join(x for x in (resume, briefing, thoughts) if x)
        if opening:
            # A dozen lines of what has already been considered, for a model
            # whose own trace was discarded after the turn that produced it.
            self.history.append({"role": "user", "content": opening})
        self._finish_blocks = 0
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
            self._handover_if_compacted()
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
                self._remember_thought(trace, step)
                if reasoning.truncated(trace, content, res.finish_reason):
                    msg = ("The model spent its entire generation budget thinking and "
                           "produced no answer. Cap the trace with a thinking budget, "
                           "or switch thinking off, in the Params tab.")
                    self.emit("error", {"message": msg})
                    self.history.append({"role": "assistant", "content": msg})
                    return msg

            calls = self._extract(res)

            if not calls:
                # Prose ends the turn only when there is nothing left to do.
                # Without this a model answers the first step and stops, leaving
                # a checklist it wrote itself half finished.
                nudge = self._unfinished_business(content)
                if nudge:
                    self._prose_streak += 1
                    self.history.append({"role": "assistant",
                                         "content": strip_calls(content).strip()
                                         or "(continuing)"})
                    self.history.append({"role": "user", "content": nudge})
                    self.emit("continuing", {"reason": nudge, "step": step})
                    continue
                answer = collapse_repeats(strip_calls(content).strip())
                if not answer:
                    answer = "(the model returned nothing usable)"
                answer = self._lift_code(answer)
                self._handover_if_unfinished()
                self.history.append({"role": "assistant", "content": answer})
                self._turn_log.append(f"assistant: {answer}")
                self.emit("assistant", {"text": answer})
                self._capture()
                return answer

            self._prose_streak = 0
            self._record_assistant(content, calls)

            for call in calls:
                if self.cancelled.is_set():
                    break
                self.emit("tool_call", {"name": call.name, "args": call.args})
                if call.name == "finish" and self._premature_finish():
                    done, total = self.todo.progress
                    result = ToolResult(
                        f"Not finished: {total - done} step(s) are still open. "
                        "Close them, mark them blocked with a reason, or remove "
                        "them from the plan, then call finish again.", ok=False)
                    shown = self._spool(call.name, result) + self._plan_status()
                    self.emit("tool_result", {"name": call.name, "ok": False,
                                              "text": result.display, "shown": shown})
                    self._record_tool(call, shown)
                    continue
                if call.name == "finish":
                    final = collapse_repeats(
                        str(call.args.get("answer") or "").strip()) or "Done."
                    final = self._lift_code(final)
                    self._handover_if_unfinished()
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
                    # Every result carries the plan state, so the model is never
                    # more than one message away from knowing where it is.
                    shown += self._plan_status()
                # Only that a tool ran, never what it returned. Feeding results
                # into the capture pass is what fills memory with the date, the
                # contents of a directory, and the text of whatever was read.
                self._turn_log.append(f"[used {call.name}]")
                self.emit("tool_result", {"name": call.name, "ok": result.ok,
                                          "text": result.display, "shown": shown})
                self._record_substep(call, result)
                self._record_tool(call, shown)

        msg = (f"Stopped after {self.cfg.max_steps} steps without finishing. "
               "Raise the step limit in Settings, or narrow the task.")
        self.emit("assistant", {"text": msg})
        self.history.append({"role": "assistant", "content": msg})
        return msg

    # -- keeping going ---------------------------------------------------------
    def _unfinished_business(self, content: str) -> str:
        """Why the turn should continue, or an empty string if it should not.

        The checklist is the definition of done. If the model has written one
        and steps remain open, a prose reply is a status update rather than an
        answer, and the work carries on.
        """
        if not self.cfg.plan_driven or self.todo is None or not self.todo.items:
            return ""
        if self.todo.complete:
            return ""
        # Three prose replies in a row means it is talking rather than working;
        # continuing past that produces a monologue, not progress.
        if self._prose_streak >= 3:
            return ""
        # Only a real question stops the work. "Let me know if you want me to
        # continue" is the stall this exists to override, not a request for a
        # decision — and a model that genuinely cannot proceed should mark the
        # step blocked rather than stop the turn.
        said = strip_calls(content).strip()
        tail = said[-240:].lower()
        asking = "?" in tail and any(
            word in tail for word in ("which", "should i", "do you want",
                                      "would you prefer", "confirm", "or should"))
        if asking:
            return ""
        current = self.todo.current
        if current is None:
            return ""
        done, total = self.todo.progress
        return (f"{done} of {total} steps are done. Step {current.id} is still "
                f"open: {current.text}\n"
                "Carry on with it now using your tools. Update the checklist as "
                "you go, and call finish only once every step is closed or "
                "blocked.")

    def _premature_finish(self) -> bool:
        """Is the model calling finish with work still on the checklist?

        Allowed twice: the plan may genuinely have been overtaken by events, and
        refusing forever would trap the turn.
        """
        if not self.cfg.plan_driven or self.todo is None or not self.todo.items:
            return False
        if self.todo.complete:
            return False
        self._finish_blocks = getattr(self, "_finish_blocks", 0) + 1
        return self._finish_blocks <= 2

    def _plan_status(self) -> str:
        """A one-line reminder appended to each tool result."""
        if self.todo is None or not self.todo.items or self.todo.complete:
            return ""
        done, total = self.todo.progress
        current = self.todo.current
        where = f"; on step {current.id}: {current.text[:60]}" if current else ""
        return f"\n[plan {done}/{total}{where}]"

    # -- planning --------------------------------------------------------------
    def _remember_thought(self, trace: str, step: int) -> None:
        """Keep one line of each reasoning trace, and notice a loop."""
        if not trace.strip():
            return
        fresh, seen = self.thoughts.add(step, trace)
        if not fresh and seen >= 3:
            self.emit("thought_loop", {"count": seen,
                                       "text": self.thoughts.looping()})

    def _thought_block(self) -> str:
        return self.thoughts.block()

    # Tools whose whole job is bookkeeping: recording them as work would fill
    # the plan with entries about the plan.
    QUIET_TOOLS = {"plan", "todo", "plan_add", "finish", "skill_find",
                   "canvas_read", "recall"}
    MAX_SUBSTEPS = 8

    def _record_substep(self, call, result) -> None:
        """Log what a tool did as a sub-step of the stage in flight.

        The two-level plan is only useful if something fills the second level,
        and a small model will not reliably do it while also doing the work.
        The harness knows exactly what happened, so it records it: the stage
        stays the model's, the detail underneath is written from fact.
        """
        if self.todo is None or not self.todo.items:
            return
        if call.name in self.QUIET_TOOLS or not result.ok:
            return
        stage = self.todo.current
        if stage is None:
            return
        stage = self.todo.stage_of(stage) or stage
        if len(self.todo.children(stage.id)) >= self.MAX_SUBSTEPS:
            return
        detail = ""
        for key in ("path", "file", "query", "pattern", "command", "name"):
            value = call.args.get(key)
            if value:
                detail = f" {str(value)[:48]}"
                break
        text = f"{call.name}{detail}"
        if any(k.text == text for k in self.todo.children(stage.id)):
            return
        item = self.todo.add(text, parent=stage.id, auto=True)
        item.status = DONE          # a record of what ran, already finished
        self.todo.save()
        self.emit("todo", {"todo": self.todo})

    def _lift_code(self, answer: str) -> str:
        """Put any code in the reply into the canvas instead."""
        if not self.registry or "canvas_write" not in self.registry.tools:
            return answer
        from .canvas import BUFFER
        try:
            rewritten, moved = lift_code(answer, BUFFER)
        except Exception:
            return answer
        if moved:
            self.emit("canvas", {"blocks": moved})
        return rewritten

    def _handover_if_unfinished(self) -> None:
        """A turn ending with the checklist open is a natural handover point."""
        if self.todo is None or not self.todo.items or self.todo.complete:
            return
        self.write_handover("the task is unfinished")

    def _handover_if_compacted(self) -> None:
        """The moment the transcript is summarised is the moment to write one.

        After compaction the detail is gone for good; a handover captures what
        mattered while the model can still see it.
        """
        if self.ctx is None:
            return
        if self.ctx.compactions > getattr(self, "_compactions_seen", 0):
            self._compactions_seen = self.ctx.compactions
            self.write_handover("the context was compacted")

    def write_handover(self, reason: str = "") -> str:
        """Summarise the state of the work, and keep it beside the project.

        Called when the context is compacted — the moment the detail stops
        being available — and when a turn ends with the checklist still open.
        Both are points at which the next turn would otherwise start knowing
        less than this one did.
        """
        if not self.history:
            return ""
        plan = self.todo.render() if self.todo and self.todo.items else "No plan."
        recent = "\n".join(self._turn_log[-14:])[:4000]
        task = self.thoughts.task or self._first_user_message()
        prompt = handovermod.PROMPT.format(task=task or "(not stated)",
                                           plan=plan, recent=recent)
        try:
            res = self.client.chat(self._helper_messages(prompt),
                                   temperature=0.1, max_tokens=400, stream=False,
                                   model=self.cfg.model)
        except Exception:
            return ""
        body = collapse_repeats(reasoning.split(res.content)[0]).strip()
        if len(body) < 40:
            return ""
        self.handover = handovermod.Handover(
            task=task, body=body, when=time.time(), turns=len(self.history) // 2)
        handovermod.save(self.cfg.workspace_path(), self.handover)
        self.emit("handover", {"reason": reason, "text": body})
        return body

    def _first_user_message(self) -> str:
        for message in self.history:
            if message.get("role") == "user":
                return " ".join(str(message.get("content") or "").split())[:120]
        return ""

    def _plan_briefing(self) -> str:
        """Aim the model at a checklist it did not write.

        A plan typed in by hand arrives with no conversation behind it, so the
        model has no reason to treat it as instructions unless it is told. This
        is a single line, injected once per turn, naming the step to start on.
        """
        if self.todo is None or not self.todo.items or self.todo.complete:
            return ""
        current = self.todo.current
        if current is None:
            return ""
        done, total = self.todo.progress
        return (f"[checklist: {done} of {total} done. Start with step "
                f"{current.id}: {current.text}. Work through the steps in order, "
                "marking each done once you have checked it. Do not call plan; "
                "the checklist already exists.]")

    def _autoplan(self, user_text: str) -> None:
        """Seed the checklist for a fresh multi-step task.

        The model is asked to plan the same way it would be asked anything else,
        so the checklist reflects its own reading of the task. Skipped when a
        plan is already in flight, when the window is too small to spend a
        generation on it, or when the request is plainly a single action.
        """
        if (self.todo is None or not self.cfg.todo_enabled or not self.cfg.auto_plan
                or self.budget.n_ctx < 4000):
            return
        if self.todo.items and not self.todo.complete:
            return          # a plan is already running; let the model revise it
        if len(user_text.split()) < 4:
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
            for line in clean_plan_lines(complete):
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
            # Whatever was drafted while streaming stands. Clearing here threw
            # away a usable plan because the request was interrupted.
            if len(drafted) < 2:
                self.todo.clear()
                self.emit("todo", {"todo": self.todo})
            return

        visible = collapse_repeats(reasoning.split(res.content)[0])
        steps = clean_plan_lines(visible)
        # A reasoning model can put the whole plan in its trace and return an
        # empty answer, so the final parse finds nothing. The plan built line by
        # line while streaming is the better record, and it is never discarded
        # in favour of a worse one — that is what made a plan appear and then
        # vanish, leaving the model with nothing to work through.
        if len(steps) < len(drafted):
            steps = drafted
            visible = "\n".join(drafted)
        if len(steps) < 2:
            if len(self.todo.items) < 2:
                self.todo.clear()   # one step is not a plan; do not clutter it
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
