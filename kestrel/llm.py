"""Client for a llama.cpp server (llama-server / ggml-org llama.cpp HTTP API).

Speaks the OpenAI-compatible endpoints so it also works against LM Studio,
llamafile, Unsloth's server, vLLM, Ollama's /v1 shim, or a hosted endpoint.
Uses llama.cpp's native extras (/props, /tokenize, /health) when present and
degrades quietly when they aren't.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import requests


from . import reasoning as reasoningmod


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.completion_tokens / self.elapsed if self.elapsed > 0 else 0.0


def _sse_lines(response, chunk_size: int = 8192):
    """Yield lines from an event stream, decoded as UTF-8.

    `iter_lines(decode_unicode=True)` cannot be used here. It decodes with the
    response's charset, and for `text/event-stream` with no charset declared
    requests falls back to latin-1 — so every multi-byte character arrives as
    mojibake, and a character split across two network chunks is corrupted
    outright.

    Bytes are therefore buffered and split on newlines here, and only complete
    lines are decoded. Decoding is incremental, so a character split across a
    chunk boundary is held until the rest of it arrives rather than being
    replaced with a question mark.
    """
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    for chunk in response.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        buffer += decoder.decode(chunk)
        while True:
            index = buffer.find("\n")
            if index < 0:
                break
            line, buffer = buffer[:index], buffer[index + 1:]
            yield line.rstrip("\r")
    buffer += decoder.decode(b"", final=True)
    for line in buffer.split("\n"):
        if line:
            yield line.rstrip("\r")


class LlamaClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 900.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self._props: dict | None = None
        self._tool_support: bool | None = None
        self._lock = threading.Lock()

    # -- plumbing -------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def reset_cache(self) -> None:
        with self._lock:
            self._props = None
            self._tool_support = None

    # -- introspection --------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        for path in ("/health", "/v1/models"):
            try:
                r = self.session.get(self._url(path), headers=self._headers(), timeout=5)
                if r.status_code < 500:
                    return True, "ok"
            except requests.RequestException as e:
                last = str(e)
                continue
            except Exception as e:  # pragma: no cover
                last = str(e)
        return False, "no response from " + self.base

    def props(self, force: bool = False) -> dict:
        with self._lock:
            if self._props is not None and not force:
                return self._props
        data: dict = {}
        try:
            r = self.session.get(self._url("/props"), headers=self._headers(),
                                 timeout=(5, 60))
            if r.ok:
                data = r.json()
        except Exception:
            data = {}
        with self._lock:
            self._props = data
        return data

    def n_ctx(self) -> int:
        """Context window the server was actually *loaded* with, not what the
        GGUF claims it supports. This is the number that matters."""
        p = self.props()
        for key in ("default_generation_settings", "generation_settings"):
            gs = p.get(key)
            if isinstance(gs, dict):
                for k in ("n_ctx", "n_ctx_per_seq"):
                    if isinstance(gs.get(k), int) and gs[k] > 0:
                        return int(gs[k])
        for k in ("n_ctx", "n_ctx_per_seq", "context_size"):
            if isinstance(p.get(k), int) and p[k] > 0:
                return int(p[k])
        return 0

    def model_name(self) -> str:
        p = self.props()
        for k in ("model_path", "model", "default_generation_settings"):
            v = p.get(k)
            if isinstance(v, str) and v:
                return v.replace("\\", "/").split("/")[-1]
            if isinstance(v, dict) and isinstance(v.get("model"), str):
                return v["model"].replace("\\", "/").split("/")[-1]
        try:
            r = self.session.get(self._url("/v1/models"), headers=self._headers(), timeout=8)
            if r.ok:
                items = r.json().get("data") or []
                if items:
                    return str(items[0].get("id", ""))
        except Exception:
            pass
        return ""

    def list_models(self) -> list[str]:
        try:
            r = self.session.get(self._url("/v1/models"), headers=self._headers(), timeout=8)
            if r.ok:
                return [str(m.get("id", "")) for m in (r.json().get("data") or [])]
        except Exception:
            pass
        return []

    def tokenize(self, text: str) -> int:
        r = self.session.post(self._url("/tokenize"), headers=self._headers(),
                              data=json.dumps({"content": text}),
                              timeout=(10, 180))
        r.raise_for_status()
        toks = r.json().get("tokens", [])
        return len(toks)

    def supports_tools(self, model: str = "") -> bool:
        """Probe once whether the chat template can emit native tool calls.

        Many GGUFs have no tool section in their Jinja template, and llama-server
        answers with a 500 when you pass `tools`. Rather than let that kill a
        session we find out up front and fall back to the text protocol.
        """
        with self._lock:
            if self._tool_support is not None:
                return self._tool_support
        payload = {
            "model": model or "local",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        }
        ok = False
        try:
            r = self.session.post(self._url("/v1/chat/completions"), headers=self._headers(),
                                  data=json.dumps(payload), timeout=(10, 120))
            ok = r.ok
        except Exception:
            ok = False
        with self._lock:
            self._tool_support = ok
        return ok

    # -- generation -----------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str = "",
        temperature: float = 0.4,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: Iterable[str] | None = None,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        extra: dict | None = None,
    ) -> ChatResult:
        import time

        payload: dict[str, Any] = {
            "model": model or "local",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
        }
        if stop:
            payload["stop"] = list(stop)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if extra:
            # sampling and reasoning fields; llama-server accepts these on the
            # OpenAI endpoint and ignores what it does not recognise
            payload.update({k: v for k, v in extra.items() if v is not None})

        started = time.time()
        try:
            r = self.session.post(
                self._url("/v1/chat/completions"), headers=self._headers(),
                # (connect, read). The read timeout is the gap between bytes,
                # not the length of the answer — but a busy machine can take
                # minutes to process a long prompt before the first token
                # appears, and killing the request then throws away work that
                # was going to arrive.
                data=json.dumps(payload), timeout=(15, self.timeout),
                stream=bool(stream),
            )
        except requests.RequestException as e:
            raise LLMError(f"cannot reach {self.base}: {e}") from e

        if not r.ok:
            detail = ""
            try:
                detail = r.text[:400]
            except Exception:
                pass
            raise LLMError(f"server returned {r.status_code}: {detail}")

        if not stream:
            body = r.json()
            choice = (body.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            usage = body.get("usage") or {}
            return ChatResult(
                content=msg.get("content") or "",
                reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                tool_calls=list(msg.get("tool_calls") or []),
                finish_reason=choice.get("finish_reason") or "",
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                elapsed=time.time() - started,
            )

        content: list[str] = []
        reasoning: list[str] = []
        splitter = reasoningmod.StreamSplitter()
        calls: dict[int, dict] = {}
        finish = ""
        usage: dict = {}
        for raw in _sse_lines(r):
            if cancel is not None and cancel():
                try:
                    r.close()
                except Exception:
                    pass
                finish = "cancelled"
                break
            if not raw:
                continue
            if raw.startswith("data:"):
                raw = raw[5:].strip()
            if raw == "[DONE]":
                break
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = ch.get("delta") or ch.get("message") or {}
            think = delta.get("reasoning_content") or delta.get("reasoning")
            if think:
                reasoning.append(think)
                if on_reasoning:
                    on_reasoning(think)
            piece = delta.get("content")
            if piece:
                # A model may put its reasoning inline in the content rather
                # than in a field of its own. Splitting it here keeps the tags
                # out of the transcript and, more importantly, survives a tag
                # arriving in two pieces.
                visible, thought = splitter.feed(piece)
                if thought:
                    reasoning.append(thought)
                    if on_reasoning:
                        on_reasoning(thought)
                if visible:
                    content.append(visible)
                    if on_token:
                        on_token(visible)
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", len(calls)))
                slot = calls.setdefault(idx, {"id": "", "type": "function",
                                              "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]

        tail_visible, tail_thought = splitter.flush()
        if tail_thought:
            reasoning.append(tail_thought)
            if on_reasoning:
                on_reasoning(tail_thought)
        if tail_visible:
            content.append(tail_visible)
            if on_token:
                on_token(tail_visible)

        text = "".join(content)
        elapsed = time.time() - started
        return ChatResult(
            content=text,
            reasoning="".join(reasoning),
            tool_calls=[calls[k] for k in sorted(calls)],
            finish_reason=finish,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0) or 0,
            elapsed=elapsed,
        )
