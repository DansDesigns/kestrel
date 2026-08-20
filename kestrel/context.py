"""Context budgeting.

The reason harnesses need 64k is that they hand the model everything on every
turn: full JSON tool schemas, a personality file, every skill definition, and an
unpruned transcript. Kestrel instead treats the window as a fixed budget and
spends it, with four rules:

  1. Nothing is sent at full fidelity unless it fits. Tool descriptions, skill
     listings and system text all have their own allowance and are rendered at
     whatever verbosity that allowance permits.
  2. Tool output lands on disk first. Context gets a head/tail preview plus a
     path, so a 40k-line log costs ~120 tokens instead of blowing the session.
  3. History is a rolling window with summarisation. Old turns collapse into a
     compact digest rather than being silently truncated mid-sentence.
  4. There is always an output reserve. Truncated tool calls are the classic
     small-context failure, so generation headroom is carved out first.

The practical floor is about 2048 tokens. It is happier from 4096 up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .tokens import TokenCounter

# name -> (upper bound of n_ctx, sys share, memory share, history share,
#          min output reserve, tool-result preview tokens, skills listed, verbosity)
PROFILES: list[tuple] = [
    ("nano",     4096,   0.22, 0.06, 0.50, 320,  180,  8,  0),
    ("small",    16384,  0.20, 0.07, 0.58, 640,  500,  20, 1),
    ("standard", 65536,  0.18, 0.08, 0.62, 1024, 1200, 48, 2),
    ("large",    10**9,  0.15, 0.08, 0.65, 2048, 3000, 128, 3),
]


@dataclass
class Budget:
    profile: str
    n_ctx: int
    system: int          # tokens for system prompt incl. tool + skill listings
    memory: int          # tokens for recalled long-term memories
    plan: int            # tokens for the task checklist
    history: int         # tokens for the running transcript
    output: int          # generation reserve, including any reasoning trace
    tool_preview: int    # tokens of a tool result kept inline
    max_skills: int      # skills named in the system prompt
    verbosity: int       # 0 terse .. 3 full
    thinking: bool = False

    @property
    def input_total(self) -> int:
        return self.system + self.memory + self.plan + self.history


def pick_profile(n_ctx: int, override: str = "") -> tuple:
    if override:
        for row in PROFILES:
            if row[0] == override:
                return row
    for row in PROFILES:
        if n_ctx <= row[1]:
            return row
    return PROFILES[-1]


def budget_for(n_ctx: int, override: str = "", thinking=None) -> Budget:
    """Split the window. Reasoning models get a bigger output reserve, because a
    trace of two thousand tokens before the first tool call is normal and a
    truncated call wastes the whole step."""
    n_ctx = max(1024, int(n_ctx or 4096))
    (name, _cap, sys_share, mem_share, hist_share,
     min_out, preview, max_skills, verbosity) = pick_profile(n_ctx, override)

    think_on = bool(thinking is not None and getattr(thinking, "enabled", False))
    output = max(min_out, int(n_ctx * 0.18))
    if think_on:
        budgeted = getattr(thinking, "budget", 0) or 0
        if budgeted:
            output = max(output, budgeted + max(256, min_out // 2))
        else:
            output = int(output * thinking.reserve_multiplier())
    output = min(output, int(n_ctx * (0.55 if think_on else 0.35)))

    usable = n_ctx - output
    total_share = sys_share + mem_share + hist_share
    system = int(usable * sys_share / total_share)
    memory = int(usable * mem_share / total_share)
    # The checklist is small and fixed: it must fit even at nano, because it is
    # the state that survives compaction.
    plan = max(48, min(320, int(usable * 0.05)))
    history = usable - system - memory - plan
    return Budget(
        profile=name, n_ctx=n_ctx, system=system, memory=memory, plan=plan,
        history=history,
        output=output, tool_preview=min(preview, max(64, history // 4)),
        max_skills=max_skills, verbosity=verbosity, thinking=think_on,
    )


@dataclass
class Usage:
    system: int = 0
    memory: int = 0
    plan: int = 0
    history: int = 0
    output: int = 0
    n_ctx: int = 4096

    @property
    def used(self) -> int:
        return self.system + self.memory + self.plan + self.history

    @property
    def free(self) -> int:
        return max(0, self.n_ctx - self.used - self.output)


class ContextManager:
    """Fits a system prompt plus transcript into the budget, compacting as needed."""

    def __init__(self, counter: TokenCounter, budget: Budget,
                 summarizer: Callable[[str, int], str] | None = None):
        self.counter = counter
        self.budget = budget
        self.summarizer = summarizer
        self.digest = ""            # rolling summary of dropped turns
        self.compactions = 0
        self.clipped_system = 0
        self.usage = Usage(n_ctx=budget.n_ctx)

    def set_budget(self, budget: Budget) -> None:
        self.budget = budget
        self.usage.n_ctx = budget.n_ctx

    # -- truncation helpers ---------------------------------------------------
    def clip(self, text: str, tokens: int, tail_share: float = 0.25) -> str:
        """Trim to a token allowance, keeping the head and (optionally) the tail.

        Errors and stack traces live at the end of output, so we always keep a
        slice of the tail unless asked not to.
        """
        if tokens <= 0 or not text:
            return ""
        if self.counter.count(text) <= tokens:
            return text
        chars = self.counter.budget_chars(tokens)
        if chars >= len(text):
            return text
        if tail_share <= 0:
            return text[:chars] + f"\n... [+{len(text) - chars} chars trimmed]"
        tail = int(chars * tail_share)
        head = chars - tail - 40
        if head <= 0:
            return text[:chars]
        omitted = len(text) - head - tail
        return f"{text[:head]}\n... [{omitted} chars trimmed] ...\n{text[-tail:]}"

    # -- transcript assembly --------------------------------------------------
    def assemble(self, system: str, history: list[dict], memory: str = "",
                 plan: str = "") -> list[dict]:
        """Return the message list to send, compacting history in place if needed."""
        sys_tokens = self.counter.count(system) + 4
        if sys_tokens > self.budget.system:
            # Cutting the system prompt truncates the tool listing mid-entry,
            # and a model handed half an instruction produces half an answer or
            # none. It still has to fit, but it must not happen quietly: the
            # symptom is "the model returned nothing usable", which looks like
            # a broken model rather than a prompt cut in two.
            self.clipped_system = sys_tokens - self.budget.system
            system = self.clip(system, self.budget.system, tail_share=0.0)
            sys_tokens = self.counter.count(system) + 4
        else:
            self.clipped_system = 0

        allowance = self.budget.history
        if plan:
            allowance -= self.counter.count(plan) + 8
        if self.digest:
            allowance -= self.counter.count(self.digest) + 12

        kept = self._fit_history(history, max(128, allowance))

        if memory:
            memory = self.clip(memory, self.budget.memory, tail_share=0.0)
        if plan:
            plan = self.clip(plan, self.budget.plan, tail_share=0.0)

        msgs: list[dict] = [{"role": "system", "content": system}]
        if memory:
            msgs.append({"role": "system", "content": memory})
        if plan:
            msgs.append({"role": "system", "content": plan})
        if self.digest:
            msgs.append({"role": "system",
                         "content": "Earlier in this session:\n" + self.digest})
        msgs.extend(kept)

        msgs = coalesce(msgs)
        self.usage.system = sys_tokens + (self.counter.count(self.digest) if self.digest else 0)
        self.usage.memory = self.counter.count(memory) + 4 if memory else 0
        self.usage.plan = self.counter.count(plan) + 4 if plan else 0
        self.usage.history = self.counter.count_messages(kept)
        self.usage.output = self.budget.output
        return msgs

    def _fit_history(self, history: list[dict], allowance: int) -> list[dict]:
        if not history:
            return []
        total = self.counter.count_messages(history)
        if total <= allowance:
            return list(history)

        # Walk backwards keeping whole turns; never orphan a tool result from
        # the assistant message that requested it.
        kept: list[dict] = []
        used = 0
        for msg in reversed(history):
            cost = self.counter.count_messages([msg])
            if used + cost > allowance and kept:
                break
            kept.insert(0, msg)
            used += cost
        while kept and kept[0].get("role") == "tool":
            dropped = kept.pop(0)
            used -= self.counter.count_messages([dropped])

        dropped_msgs = history[: len(history) - len(kept)]
        if dropped_msgs:
            self._absorb(dropped_msgs)
        if not kept:
            last = dict(history[-1])
            last["content"] = self.clip(str(last.get("content") or ""), allowance)
            kept = [last]
        return kept

    def _absorb(self, dropped: list[dict]) -> None:
        """Fold dropped turns into the rolling digest."""
        self.compactions += 1
        cap = max(96, self.budget.history // 5)
        raw = []
        for m in dropped:
            role = m.get("role", "?")
            body = str(m.get("content") or "")
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {})
                body += f" [called {fn.get('name')}({str(fn.get('arguments'))[:120]})]"
            if body.strip():
                raw.append(f"{role}: {body}")
        blob = "\n".join(raw)
        if not blob:
            return

        merged = (self.digest + "\n" + blob).strip() if self.digest else blob
        summary = ""
        if self.summarizer is not None:
            try:
                summary = self.summarizer(merged, cap).strip()
            except Exception:
                summary = ""
        if not summary:
            summary = _extractive(merged, self.counter, cap)
        self.digest = self.clip(summary, cap, tail_share=0.0)


def readable(content) -> str:
    """The text of a message that may also carry pictures.

    Images are counted as a fixed, small cost rather than by their base64
    length: the encoder turns one into a few hundred tokens whatever the file
    size, and counting the data URL would reserve megabytes of budget for it.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                parts.append(" " * 1200)      # roughly 300 tokens for an image
        return "".join(parts)
    return str(content or "")


def coalesce(messages: list[dict]) -> list[dict]:
    """Merge neighbouring messages that share a role.

    Several chat templates reject two turns from the same role in a row, and the
    text dialect produces exactly that when one step runs two tools: each result
    comes back as a user message. Merging them changes nothing the model sees
    and keeps the template happy.
    """
    out: list[dict] = []
    for message in messages:
        previous = out[-1] if out else None
        mergeable = (
            previous is not None
            and previous.get("role") == message.get("role")
            and message.get("role") in ("user", "system")
            and not previous.get("tool_calls") and not message.get("tool_calls")
            and isinstance(previous.get("content"), str)
            and isinstance(message.get("content"), str)
        )
        if mergeable:
            out[-1] = dict(previous)
            out[-1]["content"] = previous["content"] + "\n\n" + message["content"]
        else:
            out.append(message)
    return out


def _extractive(text: str, counter: TokenCounter, cap: int) -> str:
    """Offline fallback digest: keep first and last lines of each speaker run.

    Not clever, but it never fails and never needs a model call, which matters
    when the model is small and every generation is expensive.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    picked: list[str] = []
    budget_chars = counter.budget_chars(cap)
    for ln in lines:
        picked.append(ln if len(ln) <= 220 else ln[:200] + "…")
    out: list[str] = []
    size = 0
    # Prefer the newest material; older context is what we are shedding.
    for ln in reversed(picked):
        if size + len(ln) > budget_chars:
            break
        out.insert(0, ln)
        size += len(ln) + 1
    return "\n".join(out)
