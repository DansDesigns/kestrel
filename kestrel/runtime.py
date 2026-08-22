"""Full control over the llama.cpp backend.

Three groups, matching how llama.cpp itself splits them:

  Runtime   load-time flags. Changing any of these requires a server restart.
  Sampling  per-request flags, sent in the chat payload. Live.
  Thinking  reasoning control, which straddles both — the format is a load-time
            flag, the budget and toggle are per-request.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

CACHE_TYPES = ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]
SPLIT_MODES = ["", "layer", "row", "none"]
ROPE_SCALING = ["", "none", "linear", "yarn"]
REASONING_FORMATS = ["auto", "none", "deepseek"]


@dataclass
class Runtime:
    """Load-time flags for llama-server."""

    host: str = "127.0.0.1"
    port: int = 8080

    ctx_size: int = 8192            # 0 = llama.cpp default
    n_gpu_layers: int = -1        # -1 = as many as fit; 0 = CPU only; 999 = all
    threads: int = 0
    threads_batch: int = 0
    batch_size: int = 0             # -b, logical batch
    ubatch_size: int = 0            # -ub, physical batch
    n_predict: int = 0

    flash_attn: str = "auto"        # auto | on | off
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    no_mmap: bool = False
    mlock: bool = False
    no_kv_offload: bool = False

    parallel: int = 0               # concurrent slots
    cont_batching: bool = True

    split_mode: str = ""
    main_gpu: int = 0
    tensor_split: str = ""
    gpu_budget_mb: int = 0        # 0 = ask the device; otherwise an override
    cpu_moe: bool = False         # keep mixture-of-experts weights on the CPU
    # What the recovery ladder changed to make a model fit. Kept because the
    # model needs it, recorded because these settings cost quality or speed and
    # nobody would guess they were still on a week later.
    recovered: list = field(default_factory=list)

    rope_scaling: str = ""
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    yarn_orig_ctx: int = 0

    chat_template: str = ""         # named template, e.g. chatml
    chat_template_file: str = ""
    jinja: bool = True

    seed: int = -1
    keep: int = 0
    extra_args: str = ""

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def args(self) -> list[str]:
        a: list[str] = []
        if self.ctx_size:
            a += ["-c", str(self.ctx_size)]
        # Resolved by build_command, which knows the model and the device.
        if self.n_gpu_layers >= 0:
            a += ["-ngl", str(self.n_gpu_layers)]
        if self.cpu_moe:
            # A mixture-of-experts model activates a fraction of its weights per
            # token, so the idle experts are the cheapest thing to leave in
            # system RAM.
            a += ["--cpu-moe"]
        if self.threads:
            a += ["-t", str(self.threads)]
        if self.threads_batch:
            a += ["-tb", str(self.threads_batch)]
        if self.batch_size:
            a += ["-b", str(self.batch_size)]
        if self.ubatch_size:
            a += ["-ub", str(self.ubatch_size)]
        if self.n_predict:
            a += ["-n", str(self.n_predict)]
        if self.flash_attn in ("on", "off", "auto"):
            a += ["-fa", self.flash_attn]
        if self.cache_type_k and self.cache_type_k != "f16":
            a += ["--cache-type-k", self.cache_type_k]
        if self.cache_type_v and self.cache_type_v != "f16":
            a += ["--cache-type-v", self.cache_type_v]
        if self.no_mmap:
            a.append("--no-mmap")
        if self.mlock:
            a.append("--mlock")
        if self.no_kv_offload:
            # The cache lives in system RAM while the weights stay on the GPU:
            # graphics memory goes to what benefits most from being there, and
            # the cache sits where there is room for it.
            a.append("-nkvo")
        if self.parallel:
            a += ["-np", str(self.parallel)]
        if not self.cont_batching:
            a.append("--no-cont-batching")
        if self.split_mode:
            a += ["-sm", self.split_mode]
        if self.main_gpu:
            a += ["-mg", str(self.main_gpu)]
        if self.tensor_split:
            a += ["-ts", self.tensor_split]
        if self.rope_scaling:
            a += ["--rope-scaling", self.rope_scaling]
        if self.rope_freq_base:
            a += ["--rope-freq-base", str(self.rope_freq_base)]
        if self.rope_freq_scale:
            a += ["--rope-freq-scale", str(self.rope_freq_scale)]
        if self.yarn_orig_ctx:
            a += ["--yarn-orig-ctx", str(self.yarn_orig_ctx)]
        if self.chat_template:
            a += ["--chat-template", self.chat_template]
        if self.chat_template_file:
            a += ["--chat-template-file", self.chat_template_file]
        if self.jinja:
            a.append("--jinja")
        if self.seed >= 0:
            a += ["--seed", str(self.seed)]
        if self.keep:
            a += ["--keep", str(self.keep)]
        return a


@dataclass
class Sampling:
    """Per-request flags. llama-server accepts all of these on the OpenAI
    endpoint alongside the standard fields."""

    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1.0
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = -1
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    seed: int = -1

    def payload(self) -> dict:
        """Only send what differs from llama.cpp's own defaults, so a plain
        setup produces a plain request."""
        d: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
        }
        if self.typical_p != 1.0:
            d["typical_p"] = self.typical_p
        if self.repeat_penalty != 1.0:
            d["repeat_penalty"] = self.repeat_penalty
            d["repeat_last_n"] = self.repeat_last_n
        if self.presence_penalty:
            d["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty:
            d["frequency_penalty"] = self.frequency_penalty
        if self.mirostat:
            d["mirostat"] = self.mirostat
            d["mirostat_tau"] = self.mirostat_tau
            d["mirostat_eta"] = self.mirostat_eta
        if self.dry_multiplier:
            d["dry_multiplier"] = self.dry_multiplier
            d["dry_base"] = self.dry_base
            d["dry_allowed_length"] = self.dry_allowed_length
            d["dry_penalty_last_n"] = self.dry_penalty_last_n
        if self.xtc_probability:
            d["xtc_probability"] = self.xtc_probability
            d["xtc_threshold"] = self.xtc_threshold
        if self.seed >= 0:
            d["seed"] = self.seed
        return d

    def preset(self, name: str) -> None:
        """Reasonable starting points, since raw sampler defaults suit
        creative writing more than tool use."""
        if name == "precise":
            self.temperature, self.top_p, self.top_k, self.min_p = 0.15, 0.9, 20, 0.05
            self.repeat_penalty = 1.05
        elif name == "balanced":
            self.temperature, self.top_p, self.top_k, self.min_p = 0.4, 0.95, 40, 0.05
            self.repeat_penalty = 1.1
        elif name == "creative":
            self.temperature, self.top_p, self.top_k, self.min_p = 0.9, 0.98, 80, 0.02
            self.repeat_penalty = 1.15
        elif name == "deterministic":
            self.temperature, self.top_p, self.top_k, self.min_p = 0.0, 1.0, 1, 0.0
            self.repeat_penalty = 1.0
            self.seed = 0


@dataclass
class Thinking:
    """Reasoning control.

    Thinking models are the sharpest edge in a small window: a model can burn
    two thousand tokens reasoning before it writes a single tool call. Kestrel
    reserves room for that explicitly, keeps the trace out of the transcript it
    resends, and can cap or disable it outright.
    """

    mode: str = "auto"              # auto (leave to the model) | on | off
    budget: int = 0                 # token cap on the trace; 0 = uncapped
    reasoning_format: str = "auto"  # auto | none (inline <think>) | deepseek
    effort: str = ""                # "", low, medium, high
    show: bool = True               # display the trace in the interface
    keep_in_history: bool = False   # resend past traces (expensive, rarely useful)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def server_args(self) -> list[str]:
        a: list[str] = []
        if self.reasoning_format and self.reasoning_format != "auto":
            a += ["--reasoning-format", self.reasoning_format]
        if self.mode == "off":
            a += ["--reasoning-budget", "0"]
        elif self.budget > 0:
            a += ["--reasoning-budget", str(self.budget)]
        return a

    def payload(self) -> dict:
        d: dict = {}
        kwargs: dict = {}
        if self.mode == "on":
            kwargs["enable_thinking"] = True
        elif self.mode == "off":
            kwargs["enable_thinking"] = False
        if kwargs:
            d["chat_template_kwargs"] = kwargs
        if self.effort:
            d["reasoning_effort"] = self.effort
        return d

    def reserve_multiplier(self) -> float:
        """How much extra output headroom thinking needs."""
        if self.mode == "off":
            return 1.0
        if self.budget:
            return 1.0
        return 2.2 if self.mode == "on" else 1.6


UNIFIED_NAMES = {"llama", "llama.exe"}


# Set when a hand-set layer count had to be reduced, so the interface can say
# so rather than silently disagreeing with the number on screen.
LAST_CAP: list = []


def cap_gpu_layers(cfg, model_path: str = "") -> tuple[int, str]:
    """Trim a hand-set layer count that leaves the driver no room.

    The headroom calculation only ran when the layer count was automatic, so a
    number entered by hand — or left behind by an earlier recovery — could fill
    the budget and fail at exactly the point the automatic path avoids. A
    figure you chose is respected unless it cannot work, and then it is reduced
    rather than allowed to fail.
    """
    wanted = cfg.runtime.n_gpu_layers
    if wanted <= 0:
        return wanted, ""
    fits = resolve_gpu_layers(cfg, model_path)
    if fits <= 0 or wanted <= fits:
        return wanted, ""
    return fits, (f"{wanted} layers would leave the driver no room for its "
                  f"buffers; using {fits}")


def resolve_gpu_layers(cfg, model_path: str = "") -> int:
    """Decide the split between GPU and system RAM.

    Everything on the GPU is the wrong default for integrated graphics and for
    any model larger than the card: llama.cpp fails the allocation rather than
    falling back, which is why a model that loads elsewhere refuses to load
    here. The rest of the model runs from system RAM.
    """
    from . import gguf, sysmon

    path = model_path or cfg.model_path
    if not path:
        return 0
    info = gguf.read(path, want_template=False)
    if not info.n_layer:
        return 0
    try:
        gpus = sysmon.Monitor().gpus()
    except Exception:
        gpus = []
    if not gpus:
        return 0                       # nothing detected: run on the CPU
    best = max(gpus, key=lambda g: g.budget_mb)
    vram = cfg.runtime.gpu_budget_mb or best.budget_mb
    if not vram:
        return 0
    system = 0
    try:
        system = sysmon.Monitor().sample().mem_total_mb
    except Exception:
        system = 0
    ctx = cfg.runtime.ctx_size or 4096
    bits = 8 if cfg.runtime.cache_type_k.startswith("q8") else 16

    # Room left for the driver's own allocations: compiled shaders, the
    # attention scratch, the output buffer. Filling the budget with weights
    # leaves nothing to build a compute pipeline in, and the load fails naming
    # a shader rather than the memory.
    #
    # This is not wasted space. On integrated graphics every allocation comes
    # from the same system RAM either way, so a layer left on the CPU costs
    # little beyond the compute units it would have used — while a pipeline
    # that cannot be built costs the whole load.
    headroom = _driver_headroom_mb(info, cfg, best.integrated)
    vram = max(0, vram - headroom)
    return gguf.layers_that_fit(info, vram, ctx, bits,
                                integrated=best.integrated, system_mb=system)


def _driver_headroom_mb(info, cfg, integrated: bool) -> int:
    """Memory to keep free for shaders and scratch, in MB."""
    from . import gguf

    batch = cfg.runtime.batch_size or 2048
    # The output buffer is the largest single piece and is knowable: batch
    # times vocabulary times four bytes.
    logits = gguf.logits_buffer_mb(info, batch)
    # Shaders and attention scratch. Flash attention compiles more of them, so
    # leaving it on needs more room, not less.
    shaders = 700 if cfg.runtime.flash_attn != "off" else 350
    # The embedding and output tensors sit outside the per-layer arithmetic
    # and are resident whatever the offload is set to.
    embeddings = int(gguf.embedding_bytes(info) / (1024 ** 2))
    reserve = logits + shaders + embeddings
    # An integrated GPU shares one pool with the operating system, so its
    # ceiling is a driver limit rather than a physical one and is easier to
    # walk into.
    return int(reserve * (1.25 if integrated else 1.0))


def _basename(path: str) -> str:
    """Last path component, treating both separators as such.

    A Windows path handed to a POSIX Path object keeps its backslashes inside
    `.name`, so the binary would go unrecognised anywhere but Windows — which is
    exactly where this code is hardest to test.
    """
    return re.split(r"[\\/]", str(path))[-1].lower()


def server_argv(binary: str) -> list[str]:
    """Recent llama.cpp ships a single `llama` dispatcher with subcommands, so
    the server is `llama serve` rather than `llama-server`. Passing `-m` to the
    dispatcher fails with `unknown command '-m'`, which reads like a bad flag
    rather than a missing subcommand."""
    if _basename(binary) in UNIFIED_NAMES:
        return [binary, "serve"]
    return [binary]


def is_unified(binary: str) -> bool:
    return _basename(binary) in UNIFIED_NAMES


def build_command(cfg, model_path: str = "", rpc: str = "",
                  with_model: bool = True) -> list[str]:
    """Assemble the server command line.

    `with_model` false starts the server with no model loaded. llama.cpp comes
    up in router mode, ready to accept one — which is what should happen when
    the application opens: the backend running and waiting, rather than tens of
    gigabytes read into memory before anyone has asked for it.
    """
    from .cluster import find_server_binary

    binary = find_server_binary(cfg.llama_server_bin)
    if not binary:
        raise FileNotFoundError(
            "Could not find llama-server. Set its path in Settings, or put the "
            "llama.cpp build/bin directory on PATH."
        )
    rt = cfg.runtime
    cmd = server_argv(binary) + ["--host", rt.host, "--port", str(rt.port)]
    if with_model:
        if rt.n_gpu_layers < 0:
            cmd += ["-ngl", str(resolve_gpu_layers(cfg, model_path))]
        else:
            # A hand-set count is respected unless it cannot work: filling the
            # budget with layers is what leaves nothing to build the compute
            # buffer in, and the failure names the buffer rather than the
            # layers that took its room.
            capped, why = cap_gpu_layers(cfg, model_path)
            if why:
                LAST_CAP.append(why)
            cmd += ["-ngl", str(capped)]

    if with_model:
        model = model_path or cfg.model_path
        if not model:
            raise ValueError("No model selected.")
        if not Path(model).exists():
            raise FileNotFoundError(f"Model file not found: {model}")
        cmd += ["-m", str(model)]
        # The projector goes with the weights. Without it llama.cpp loads the
        # language model alone and the result behaves as a text model — which
        # is exactly how a vision model comes to be reported as one.
        try:
            from . import gguf as _gguf
            info = _gguf.read(model, want_template=False)
            if info.projector and Path(info.projector).exists():
                cmd += ["--mmproj", str(info.projector)]
        except Exception:
            pass
    else:
        # Router mode: no model, so none of the model-loading flags apply. The
        # unified `llama serve` rejects flags it does not recognise and exits,
        # which is indistinguishable from a crash — so only host, port and any
        # explicit extra arguments are passed.
        if rt.extra_args.strip():
            cmd += shlex.split(rt.extra_args.strip())
        return cmd
    cmd += rt.args()
    cmd += cfg.thinking.server_args()
    if rpc:
        cmd += ["--rpc", rpc]
    if rt.extra_args.strip():
        cmd += shlex.split(rt.extra_args.strip())
    return cmd
