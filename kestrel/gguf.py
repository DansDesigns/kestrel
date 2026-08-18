"""Read GGUF metadata without loading the model.

A GGUF file opens with a magic, a version, tensor and KV counts, then the
metadata key/value block. Everything worth showing in a model browser lives in
that block — architecture, trained context length, quantisation, and the chat
template — and it sits in the first few hundred kilobytes. We stop before the
tensor index, so inspecting a 40 GB model costs one short read.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = 0x46554747  # 'GGUF'

# value type tags
(U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARRAY, U64, I64, F64) = range(13)

_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
        F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}
_SIZE = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4, BOOL: 1,
         U64: 8, I64: 8, F64: 8}

# LLAMA_FTYPE values, which is what general.file_type holds
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "MXFP4", 40: "Q4_0_4_4",
}
QUANT_IN_NAME = re.compile(
    r"(MXFP4(?:_MOE)?|IQ\d[A-Z_]*|Q\d_K_[SML]|Q\d_K|Q\d_\d|BF16|F16|F32|TQ\d_\d)",
    re.IGNORECASE)
PART_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


class GGUFError(ValueError):
    pass


@dataclass
class GGUFInfo:
    path: Path
    name: str = ""
    architecture: str = ""
    quant: str = ""
    size_label: str = ""
    n_ctx_train: int = 0
    n_layer: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_head_kv: int = 0          # fewer than n_head under grouped-query attention
    head_dim: int = 0
    n_params: int = 0
    tensor_count: int = 0
    file_size: int = 0
    parts: int = 1
    vision: bool = False        # the weights include an image encoder
    projector: str = ""         # a separate mmproj file that supplies one
    chat_template: str = ""
    error: str = ""
    kv: dict = field(default_factory=dict)

    # -- derived ---------------------------------------------------------
    @property
    def supports_tools(self) -> bool:
        """Whether the bundled chat template has a tools branch. Saves probing
        the server, and tells you before loading whether native tool calls will
        work or Kestrel will fall back to its text protocol."""
        t = self.chat_template
        return bool(t) and ("tools" in t or "tool_calls" in t)

    @property
    def supports_thinking(self) -> bool:
        t = self.chat_template
        return bool(t) and ("<think>" in t or "enable_thinking" in t
                            or "reasoning_content" in t)

    @property
    def size_gb(self) -> float:
        return self.file_size / (1024 ** 3)

    @property
    def label(self) -> str:
        bits = [self.name or self.path.stem]
        if self.quant:
            bits.append(self.quant)
        return "  ".join(bits)

    def summary(self) -> str:
        rows = []
        if self.architecture:
            rows.append(f"architecture   {self.architecture}")
        if self.quant:
            rows.append(f"quantisation   {self.quant}")
        if self.n_ctx_train:
            rows.append(f"trained ctx    {self.n_ctx_train:,}")
        if self.n_layer:
            rows.append(f"layers         {self.n_layer}")
        if self.n_params:
            rows.append(f"parameters     {self.n_params / 1e9:.1f} B")
        rows.append(f"file size      {self.size_gb:.2f} GB"
                    + (f"  ({self.parts} parts)" if self.parts > 1 else ""))
        rows.append(f"tool calling   {'yes' if self.supports_tools else 'no (text protocol)'}")
        rows.append(f"thinking       {'yes' if self.supports_thinking else 'not in template'}")
        if self.error:
            rows.append(f"note           {self.error}")
        return "\n".join(rows)


def _read(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise GGUFError("file ended early")
    return b


def _scalar(f, t: int):
    if t == STR:
        n = struct.unpack("<Q", _read(f, 8))[0]
        if n > 64 * 1024 * 1024:
            raise GGUFError("implausible string length")
        return _read(f, n).decode("utf-8", "replace")
    if t in _FMT:
        return struct.unpack(_FMT[t], _read(f, _SIZE[t]))[0]
    raise GGUFError(f"unknown value type {t}")


def _value(f, t: int, keep_arrays: bool):
    if t != ARRAY:
        return _scalar(f, t)
    elem, count = struct.unpack("<IQ", _read(f, 12))
    if elem == STR:
        out = []
        for _ in range(count):
            s = _scalar(f, STR)
            if keep_arrays and len(out) < 32:
                out.append(s)
        return out
    if elem == ARRAY:
        return [_value(f, ARRAY, False) for _ in range(count)]
    size = _SIZE.get(elem)
    if size is None:
        raise GGUFError(f"unknown array element type {elem}")
    raw = _read(f, size * count)
    if not keep_arrays or count > 64:
        return f"<{count} values>"
    return list(struct.unpack(f"<{count}{_FMT[elem][1]}", raw))


def read(path: str | Path, want_template: bool = True) -> GGUFInfo:
    """Parse a GGUF header. Never raises — problems land in `.error` so a bad
    file in a folder full of good ones doesn't break the browser."""
    p = Path(path)
    info = GGUFInfo(path=p)
    try:
        info.file_size = p.stat().st_size
    except OSError:
        pass
    m = PART_RE.search(p.name)
    if m:
        info.parts = int(m.group(2))
        # A sharded model's first file is a fraction of the whole; reporting it
        # as the size makes a 60 GB model look like 5 GB.
        stem = p.name[:m.start()]
        total = 0
        try:
            for shard in p.parent.glob(f"{stem}-*-of-*.gguf"):
                total += shard.stat().st_size
        except OSError:
            total = 0
        if total:
            info.file_size = total

    try:
        with open(p, "rb") as f:
            magic, version = struct.unpack("<II", _read(f, 8))
            if magic != MAGIC:
                info.error = "not a GGUF file"
                return info
            if version < 2 or version > 3:
                info.error = f"GGUF version {version} may not parse correctly"
            n_tensors, n_kv = struct.unpack("<QQ", _read(f, 16))
            info.tensor_count = n_tensors
            if n_kv > 100_000:
                info.error = "implausible metadata count"
                return info
            kv: dict = {}
            for _ in range(n_kv):
                klen = struct.unpack("<Q", _read(f, 8))[0]
                key = _read(f, klen).decode("utf-8", "replace")
                vtype = struct.unpack("<I", _read(f, 4))[0]
                skip_big = key.startswith("tokenizer.ggml.") and key.endswith(
                    ("tokens", "scores", "token_type", "merges"))
                val = _value(f, vtype, keep_arrays=not skip_big)
                if key == "tokenizer.chat_template" and not want_template:
                    val = ""
                kv[key] = val
    except (OSError, GGUFError, struct.error, MemoryError) as e:
        info.error = f"{type(e).__name__}: {e}"
        return info

    info.kv = {k: v for k, v in kv.items() if k != "tokenizer.chat_template"}
    detect_vision(info, kv)
    arch = str(kv.get("general.architecture") or "")
    info.architecture = arch
    info.name = str(kv.get("general.name") or p.stem)
    info.size_label = str(kv.get("general.size_label") or "")
    info.chat_template = str(kv.get("tokenizer.chat_template") or "")

    # Prefer a name from the filename over an unrecognised numeric file_type:
    # new quantisations appear faster than the enum can be updated, and
    # "ftype 38" tells the user nothing.
    ft = kv.get("general.file_type")
    named = FILE_TYPES.get(ft) if isinstance(ft, int) else None
    from_name = QUANT_IN_NAME.search(p.name)
    if named:
        info.quant = named
    elif from_name:
        info.quant = from_name.group(1).upper()
    elif isinstance(ft, int):
        info.quant = f"ftype {ft}"

    for key, attr in ((f"{arch}.context_length", "n_ctx_train"),
                      (f"{arch}.block_count", "n_layer"),
                      (f"{arch}.embedding_length", "n_embd"),
                      (f"{arch}.attention.head_count", "n_head"),
                      (f"{arch}.attention.head_count_kv", "n_head_kv"),
                      (f"{arch}.attention.key_length", "head_dim")):
        v = kv.get(key)
        if isinstance(v, int):
            setattr(info, attr, v)

    for key in ("general.parameter_count", f"{arch}.parameter_count"):
        v = kv.get(key)
        if isinstance(v, int):
            info.n_params = v
            break
    if not info.n_params and info.size_label:
        m = re.match(r"([\d.]+)\s*([BbMm])", info.size_label)
        if m:
            info.n_params = int(float(m.group(1)) * (1e9 if m.group(2).lower() == "b" else 1e6))
    return info


def kv_bytes(info: GGUFInfo, n_ctx: int, cache_bits: int = 16) -> int:
    """Bytes of KV cache for this model at this context length.

    Grouped-query attention is the whole point of doing this properly: a model
    with 64 query heads but 8 key/value heads needs an eighth of the cache the
    embedding width alone would suggest. Assuming otherwise overstates the
    requirement several times over on exactly the modern models people run.
    """
    if not (info.n_layer and n_ctx):
        return 0
    if info.n_head_kv and info.n_head and info.n_embd:
        head_dim = info.head_dim or (info.n_embd // max(1, info.n_head))
        per_token = 2 * info.n_layer * info.n_head_kv * head_dim
    else:
        per_token = 2 * info.n_layer * info.n_embd      # no GQA data: assume none
    return int(per_token * n_ctx * (cache_bits / 8))


def estimate_vram_mb(info: GGUFInfo, n_ctx: int, cache_bits: int = 16) -> int:
    """Rough memory needed to hold weights plus KV cache, in MB."""
    overhead = 300 * 1024 * 1024
    return int((info.file_size + kv_bytes(info, n_ctx, cache_bits) + overhead)
               / (1024 ** 2))


def context_that_fits(info: GGUFInfo, budget_mb: int, cache_bits: int = 16,
                      floor: int = 2048) -> int:
    """The largest context whose weights and cache fit in `budget_mb`.

    Used to answer the question a failed load raises: not "why", but "what
    would work". Rounded down to a multiple of 1024 because odd context
    lengths help nobody.
    """
    spare = budget_mb * 1024 * 1024 - info.file_size - 300 * 1024 * 1024
    if spare <= 0 or not info.n_layer:
        return 0
    per_token = kv_bytes(info, 1, cache_bits)
    if per_token <= 0:
        return 0
    fits = int(spare / per_token)
    return max(floor, (fits // 1024) * 1024) if fits >= floor else 0


def layers_that_fit(info: GGUFInfo, vram_mb: int, n_ctx: int,
                    cache_bits: int = 16, headroom_mb: int = 700,
                    integrated: bool = False, system_mb: int = 0) -> int:
    """How many layers can be offloaded to a GPU of this size.

    llama.cpp puts whatever it is told on the GPU and fails if that does not
    allocate; it does not spill the remainder to system RAM by itself. So the
    split has to be decided here, and the safe answer is the number of layers
    whose weights and KV cache fit in what the device actually has, less a
    margin for the compute buffers and whatever the display is already using.

    Returns 0 when nothing sensible fits, which runs entirely on the CPU.
    """
    if not info.n_layer or vram_mb <= 0 or not info.file_size:
        return 0
    per_layer_mb = (info.file_size / (1024 ** 2)) / info.n_layer
    kv_per_layer_mb = 0.0
    if n_ctx and info.n_layer:
        kv_per_layer_mb = (kv_bytes(info, n_ctx, cache_bits) / info.n_layer
                           / (1024 ** 2))
    cost = per_layer_mb + kv_per_layer_mb
    if cost <= 0:
        return 0
    budget = vram_mb - headroom_mb
    if integrated:
        # Offloading to an integrated GPU does not move data anywhere: the
        # memory is the same physical RAM. The real constraint is therefore
        # total system memory, less what the operating system and this
        # application need, rather than a separate pool.
        reserve = max(3072, int(system_mb * 0.18)) if system_mb else headroom_mb
        budget = min(vram_mb, system_mb - reserve) if system_mb else vram_mb
    if budget <= 0:
        return 0
    return max(0, min(info.n_layer, int(budget / cost)))


# Architectures that ship an image encoder, and the metadata keys that prove it.
VISION_KEYS = ("clip.has_vision_encoder", "clip.vision.embedding_length",
               "vision.block_count", "mm.projector_type",
               "gemma3.vision.block_count", "qwen2vl.vision.block_count")
VISION_ARCHES = ("gemma3", "gemma4", "qwen2vl", "qwen2_5_vl", "qwen3vl",
                 "llava", "minicpmv", "internvl", "pixtral", "mllama",
                 "idefics", "smolvlm", "moondream", "paligemma")
PROJECTOR_HINTS = ("mmproj", "projector", "vision", "-vit", "clip")


def find_projector(path: Path) -> str:
    """A sibling mmproj file, which is how llama.cpp is given an image encoder.

    Vision models are usually published as two files: the language weights and
    a projector. Loading only the first gives a model that quietly cannot see,
    which is indistinguishable from a text model unless the pair is noticed.
    """
    try:
        folder = Path(path).parent
        stem = Path(path).stem.lower()
        candidates = []
        for f in folder.glob("*.gguf"):
            name = f.name.lower()
            if f.resolve() == Path(path).resolve():
                continue
            if any(hint in name for hint in PROJECTOR_HINTS):
                candidates.append(f)
        if not candidates:
            return ""
        # The projector has to belong to *this* model. Any mmproj in the folder
        # would otherwise be attached to every model in it, which is worse than
        # finding none: llama.cpp would be handed an encoder for other weights.
        tokens = {t for t in re.split(r"[-_.\s]+", stem)
                  if len(t) > 2 and not t.startswith("q")}
        best, score = "", 0
        for f in candidates:
            other = {t for t in re.split(r"[-_.\s]+", f.name.lower())
                     if len(t) > 2}
            shared = len(tokens & other)
            if shared > score:
                best, score = str(f), shared
        return best if score >= 2 else ""
    except OSError:
        return ""


def detect_vision(info: "GGUFInfo", meta: dict | None = None) -> None:
    """Fill in `vision` and `projector` for a model already read."""
    meta = meta or {}
    if any(key in meta for key in VISION_KEYS):
        info.vision = True
    arch = (info.architecture or "").lower()
    if any(a in arch for a in VISION_ARCHES):
        info.vision = True
    # The file name is weak evidence on its own — "gemma3" in a name may be a
    # text-only variant — so it counts only alongside a projector or metadata.
    name = Path(info.path).name.lower().replace("-", "").replace("_", "") \
        if info.path else ""
    named_vision = any(a in name for a in VISION_ARCHES)
    if info.path:
        info.projector = find_projector(Path(info.path))
        if info.projector:
            info.vision = True
        elif named_vision and not info.vision:
            # A known vision family with no projector beside it: it can see in
            # principle but not as installed, which is worth saying plainly.
            info.projector = ""
            info.vision = True
