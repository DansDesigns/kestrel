"""Reasoning traces.

Depending on the model and how llama-server was started, a thinking model's
trace arrives one of three ways: in a separate `reasoning_content` field
(`--reasoning-format deepseek`), inline between `<think>` tags
(`--reasoning-format none`), or as an unterminated trace when generation is cut
off by the token limit mid-thought.

All three end up in the same place here. The trace is shown but never fed back:
resending past reasoning is the fastest way to fill a small window with text the
model has already acted on.
"""
from __future__ import annotations

import re

TAGS = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reason>", "</reason>"),
    ("<reasoning>", "</reasoning>"),
    ("<|thinking|>", "<|/thinking|>"),
]
_PATTERNS = [re.compile(re.escape(o) + r"(.*?)" + re.escape(c), re.DOTALL | re.IGNORECASE)
             for o, c in TAGS]
_OPEN = re.compile("|".join(re.escape(o) for o, _ in TAGS), re.IGNORECASE)


def split(text: str) -> tuple[str, str]:
    """Return (visible_text, reasoning). Handles an unterminated trailing trace,
    which is what a truncated generation leaves behind."""
    if not text:
        return "", ""
    traces: list[str] = []

    def take(m):
        traces.append(m.group(1).strip())
        return ""

    out = text
    for rx in _PATTERNS:
        out = rx.sub(take, out)

    m = _OPEN.search(out)
    if m:
        # opened but never closed: everything after the tag is reasoning
        traces.append(out[m.end():].strip())
        out = out[:m.start()]

    return out.strip(), "\n\n".join(t for t in traces if t).strip()


def merge(result) -> tuple[str, str]:
    """Combine a ChatResult's separate reasoning field with any inline tags."""
    visible, inline = split(result.content or "")
    trace = (result.reasoning or "").strip()
    if inline:
        trace = (trace + "\n\n" + inline).strip() if trace else inline
    return visible, trace


def truncated(trace: str, visible: str, finish_reason: str) -> bool:
    """A trace that ate the whole generation with nothing to show for it."""
    return bool(trace) and not visible.strip() and finish_reason == "length"


class StreamSplitter:
    """Separates reasoning from answer as tokens arrive.

    Two problems appear only in a stream. The first is that a model may emit
    `<think>` inline in its content rather than in a separate field, so the tags
    reach the transcript as text. The second is worse: a tag can be split across
    chunks — "<thi" then "nk>" — and matching each chunk on its own never sees
    it, so the state never flips and fragments of markup appear in the reply
    while pieces of the reply appear as thinking.

    The fix for both is to hold back a short tail: text is only released once it
    cannot possibly be the start of a tag. That costs a few characters of
    latency and removes the whole class of fault.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.thinking = False
        self._longest = max(len(t) for pair in TAGS for t in pair)

    def feed(self, chunk: str) -> tuple[str, str]:
        """Take a chunk, return (visible, reasoning) that are safe to emit."""
        self.buffer += chunk or ""
        visible, thought = [], []
        while True:
            if self.thinking:
                index, tag = self._find_closing()
                if index < 0:
                    break
                thought.append(self.buffer[:index])
                self.buffer = self.buffer[index + len(tag):]
                self.thinking = False
            else:
                index, tag = self._find_opening()
                if index < 0:
                    break
                visible.append(self.buffer[:index])
                self.buffer = self.buffer[index + len(tag):]
                self.thinking = True
        # Whatever is left might be the beginning of a tag, so only the part
        # that certainly is not gets released.
        hold = self._longest - 1
        if len(self.buffer) > hold:
            safe, self.buffer = self.buffer[:-hold], self.buffer[-hold:]
            (thought if self.thinking else visible).append(safe)
        return "".join(visible), "".join(thought)

    def flush(self) -> tuple[str, str]:
        """Release the tail at the end of a response."""
        rest, self.buffer = self.buffer, ""
        # An unterminated trace is what a truncated generation leaves behind;
        # it belongs with the reasoning rather than the answer.
        return ("", rest) if self.thinking else (rest, "")

    # -- scanning ------------------------------------------------------------
    def _find_opening(self) -> tuple[int, str]:
        best, found = -1, ""
        for open_tag, _close in TAGS:
            at = self.buffer.lower().find(open_tag.lower())
            if at >= 0 and (best < 0 or at < best):
                best, found = at, self.buffer[at:at + len(open_tag)]
        return best, found

    def _find_closing(self) -> tuple[int, str]:
        best, found = -1, ""
        for _open, close_tag in TAGS:
            at = self.buffer.lower().find(close_tag.lower())
            if at >= 0 and (best < 0 or at < best):
                best, found = at, self.buffer[at:at + len(close_tag)]
        return best, found
