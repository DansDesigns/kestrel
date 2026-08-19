"""Configuration model and on-disk persistence."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from .runtime import Runtime, Sampling, Thinking

APP_NAME = "kestrel"
APP_TITLE = "Kestrel"


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / APP_NAME


def default_workspace() -> str:
    return str(Path.home() / "kestrel-workspace")


@dataclass
class Node:
    """One llama.cpp rpc-server worker on the network."""

    host: str = "127.0.0.1"
    port: int = 50052
    label: str = ""
    mem_mb: int = 0
    enabled: bool = True

    @property
    def addr(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def display(self) -> str:
        return self.label or self.addr


@dataclass
class MemoryCfg:
    enabled: bool = True
    auto_capture: bool = True       # extract durable facts at the end of a task
    recall: int = 6                 # memories retrieved per turn
    global_scope: bool = False      # one pool shared across all workspaces
    db_path: str = ""
    max_items: int = 2000


@dataclass
class SpeechCfg:
    """Speech settings. Local engines are the default; network engines stay
    inert until `allow_network` is switched on deliberately."""

    tts_enabled: bool = False
    tts_engine: str = "auto"          # auto prefers the best available local engine
    tts_voice: str = ""
    tts_speed: float = 1.0
    auto_speak: bool = True           # read final answers aloud when TTS is on
    speak_tool_calls: bool = False

    stt_enabled: bool = False
    stt_engine: str = "auto"
    stt_model: str = ""
    stt_language: str = "auto"
    record_seconds: int = 15
    input_device: str = ""

    allow_network: bool = False       # gate for every non-local engine
    api_base: str = ""
    api_key: str = ""
    tts_api_model: str = "tts-1"
    stt_api_model: str = "whisper-1"
    elevenlabs_key: str = ""

    voice_dirs: list[str] = field(default_factory=list)
    model_dirs: list[str] = field(default_factory=list)


@dataclass
class Config:
    # ---- inference endpoint -------------------------------------------------
    server_url: str = "http://127.0.0.1:8080"
    api_key: str = ""
    model: str = ""

    # ---- model + backend ----------------------------------------------------
    manage_server: bool = False
    llama_server_bin: str = ""
    model_path: str = ""
    model_dirs: list[str] = field(default_factory=list)
    hf_token: str = ""
    download_dir: str = ""

    runtime: Runtime = field(default_factory=Runtime)
    sampling: Sampling = field(default_factory=Sampling)
    thinking: Thinking = field(default_factory=Thinking)
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    speech: SpeechCfg = field(default_factory=SpeechCfg)

    # ---- cluster ------------------------------------------------------------
    nodes: list[Node] = field(default_factory=list)
    rpc_bin: str = ""
    discovery_port: int = 50051

    # ---- agent --------------------------------------------------------------
    tool_dialect: str = "auto"
    max_steps: int = 24
    approval: str = "safe"
    workspace: str = field(default_factory=default_workspace)
    workspace_root: str = ""          # projects live in folders beneath this
    skills_dirs: list[str] = field(default_factory=list)
    persona: str = ""                 # inline fallback, kept for older configs
    persona_file: str = ""            # a .md persona / SOUL.md to compile
    persona_dirs: list[str] = field(default_factory=list)
    persona_level: int = -1           # -1 = follow the context profile
    todo_enabled: bool = True
    auto_plan: bool = True            # ask for a checklist before multi-step work
    plan_driven: bool = True          # keep working until the checklist is closed
    bell_on_finish: bool = True
    bell_sound: str = ""              # blank uses the bundled chime
    llama_backend: str = "auto"       # cpu | cuda | vulkan | hip | metal | sycl
    llama_with_rpc: bool = True       # build with the RPC backend for clustering
    auto_install_llama: bool = False
    auto_start_server: bool = True    # start the backend if nothing is serving
    theme: str = "dark"               # dark | light
    ui_tint: str = "slate"            # surface hue
    ui_accent: str = "amber"          # buttons and highlights
    show_tool_detail: bool = False    # show arguments and raw output in the transcript
    favourite_models: list[str] = field(default_factory=list)
    canvas_enabled: bool = True       # give the model the shared code canvas
    team_enabled: bool = True         # several agents sharing one model
    model_vision: bool = False        # the loaded model accepts images
    ui_font: str = ""                 # blank follows the platform default
    mono_font: str = ""
    font_size: int = 13
    watch_skills: bool = True         # rescan when the skills folders change
    profile_override: str = ""

    path: str = ""

    # -- serialisation --------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else config_dir() / "config.json"
        cfg = cls()
        cfg.path = str(p)
        if p.exists():
            try:
                raw: dict[str, Any] = json.loads(p.read_text("utf-8"))
            except (OSError, ValueError):
                raw = {}
            _apply(cfg, raw)
            cfg.path = str(p)
        if not cfg.skills_dirs:
            cfg.skills_dirs = default_skill_dirs()
        if not cfg.model_dirs:
            from .models import default_model_dirs
            cfg.model_dirs = default_model_dirs()
        if not cfg.download_dir:
            cfg.download_dir = str(Path.home() / "models")
        if not cfg.workspace_root:
            cfg.workspace_root = str(Path(cfg.workspace).expanduser().parent
                                     if Path(cfg.workspace).name else cfg.workspace)
        if not cfg.persona_dirs:
            from .persona import default_persona_dirs
            cfg.persona_dirs = default_persona_dirs(config_dir(), cfg.workspace)
        if not cfg.speech.voice_dirs:
            cfg.speech.voice_dirs = [str(config_dir() / "voices"),
                                     str(Path.home() / ".local" / "share" / "piper"),
                                     str(Path.home() / "piper-voices")]
        if not cfg.speech.model_dirs:
            cfg.speech.model_dirs = [str(config_dir() / "speech-models"),
                                     str(Path.home() / ".cache" / "whisper"),
                                     str(Path.home() / "whisper.cpp" / "models")]
        if not cfg.memory.db_path:
            cfg.memory.db_path = str(config_dir() / "memory.db")
        return cfg

    def save(self) -> None:
        p = Path(self.path) if self.path else config_dir() / "config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("path", None)
        p.write_text(json.dumps(data, indent=2), "utf-8")
        self.path = str(p)

    # -- helpers --------------------------------------------------------------
    def active_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.enabled]

    def rpc_arg(self) -> str:
        return ",".join(n.addr for n in self.active_nodes())

    def workspace_path(self) -> Path:
        p = Path(self.workspace).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def memory_scope(self) -> str:
        return "" if self.memory.global_scope else str(Path(self.workspace).expanduser())

    # -- legacy accessors, so older call sites keep working -------------------
    @property
    def ctx_size(self) -> int:
        return self.runtime.ctx_size

    @property
    def tensor_split(self) -> str:
        return self.runtime.tensor_split

    @property
    def temperature(self) -> float:
        return self.sampling.temperature

    @property
    def top_p(self) -> float:
        return self.sampling.top_p


def _apply(obj, raw: dict) -> None:
    """Populate a dataclass from a dict, recursing into nested dataclasses and
    ignoring keys from older versions."""
    known = {f.name for f in fields(obj)}
    for key, value in raw.items():
        if key not in known:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        elif key == "nodes" and isinstance(value, list):
            obj.nodes = [Node(**{k: v for k, v in n.items() if k in Node.__annotations__})
                         for n in value if isinstance(n, dict)]
        else:
            try:
                setattr(obj, key, value)
            except AttributeError:
                pass    # read-only legacy property



def default_skill_dirs() -> list[str]:
    """Places we look for agentskills.io-format skills, including other harnesses'."""
    home = Path.home()
    candidates = [
        config_dir() / "skills",
        home / ".kestrel" / "skills",
        home / ".hermes" / "skills",
        home / ".config" / "hermes" / "skills",
        home / ".claude" / "skills",
        home / ".config" / "agent-skills",
        Path.cwd() / "skills",
        Path.cwd() / ".claude" / "skills",
    ]
    return [str(c) for c in candidates]
