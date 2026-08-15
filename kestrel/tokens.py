"""Token accounting.

Everything in Kestrel is budgeted, so counting has to be cheap and never fatal.
We use llama-server's /tokenize when it answers and a calibrated character
estimate when it doesn't. The estimator self-corrects: every real count we get
back adjusts the chars-per-token ratio.
"""
from __future__ import annotations

import threading
from typing import Callable

_DEFAULT_CPT = 3.7  # chars per token, roughly right for English + code


class TokenCounter:
    def __init__(self, tokenize: Callable[[str], int] | None = None, cache_size: int = 4096):
        self._tokenize = tokenize
        self._cache: dict[int, int] = {}
        self._order: list[int] = []
        self._cache_size = cache_size
        self._cpt = _DEFAULT_CPT
        self._samples = 0
        self._lock = threading.Lock()
        self.exact = tokenize is not None

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self._cpt) + 1)

    def count(self, text: str) -> int:
        if not text:
            return 0
        key = hash(text)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        n = None
        if self._tokenize is not None and len(text) < 60000:
            try:
                n = self._tokenize(text)
            except Exception:
                n = None
        if n is None:
            return self.estimate(text)
        with self._lock:
            self._cache[key] = n
            self._order.append(key)
            if len(self._order) > self._cache_size:
                self._cache.pop(self._order.pop(0), None)
            if n > 0:
                ratio = len(text) / n
                self._samples += 1
                w = min(0.25, 1.0 / self._samples)
                self._cpt = (1 - w) * self._cpt + w * ratio
        return n

    def count_messages(self, messages: list[dict]) -> int:
        """Chat messages cost more than their text: role tags and turn delimiters."""
        total = 0
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, list):  # multimodal parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            total += self.count(content) + 4
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {})
                total += self.count(fn.get("name", "")) + self.count(fn.get("arguments", "")) + 8
        return total + 3

    @property
    def chars_per_token(self) -> float:
        return self._cpt

    def budget_chars(self, tokens: int) -> int:
        """How many characters fit in `tokens`, conservatively."""
        return max(0, int(tokens * self._cpt * 0.92))
