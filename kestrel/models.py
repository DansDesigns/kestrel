"""Finding and fetching models.

Two halves: a scanner that indexes GGUF files already on disk (including the
folders LM Studio and huggingface-cli use, so an existing library just appears),
and a downloader that searches the Hugging Face API and streams a file down with
progress.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import requests

from .gguf import GGUFInfo, PART_RE, read

HF_API = "https://huggingface.co/api"
HF_HOST = "https://huggingface.co"


def default_model_dirs() -> list[str]:
    home = Path.home()
    candidates = [
        home / "models",
        home / ".cache" / "lm-studio" / "models",
        home / ".lmstudio" / "models",
        home / ".cache" / "huggingface" / "hub",
        home / "llama.cpp" / "models",
        home / ".local" / "share" / "models",
        Path.cwd() / "models",
    ]
    if os.name == "nt":
        candidates.append(Path(os.environ.get("USERPROFILE", str(home))) / ".cache" / "lm-studio" / "models")
    return [str(c) for c in candidates]


@dataclass
class ModelEntry:
    info: GGUFInfo
    repo: str = ""                  # inferred from the folder if it came from HF

    @property
    def path(self) -> Path:
        return self.info.path

    @property
    def name(self) -> str:
        return self.info.path.name

    def row(self) -> tuple[str, str, str, str]:
        i = self.info
        ctx = f"{i.n_ctx_train // 1024}k" if i.n_ctx_train >= 1024 else str(i.n_ctx_train or "?")
        return (i.name or i.path.stem, i.quant or "?", f"{i.size_gb:.1f} GB", ctx)


class Catalog:
    """Indexes local GGUF files. Metadata is cached by path, size and mtime, so
    rescanning a big library is cheap."""

    def __init__(self, dirs: Iterable[str] | None = None, cache_path: Path | None = None):
        self.dirs = [str(d) for d in (dirs or default_model_dirs())]
        self.entries: list[ModelEntry] = []
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text("utf-8"))
            except (OSError, ValueError):
                self._cache = {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), "utf-8")
        except OSError:
            pass

    def scan(self, progress: Callable[[str], None] | None = None) -> list[ModelEntry]:
        found: dict[str, ModelEntry] = {}
        for d in self.dirs:
            root = Path(d).expanduser()
            if not root.is_dir():
                continue
            if progress:
                progress(f"scanning {root}")
            try:
                files = sorted(root.rglob("*.gguf"))
            except OSError:
                continue
            for f in files:
                m = PART_RE.search(f.name)
                if m and m.group(1) != "00001":
                    continue        # only the first shard represents the model
                key = str(f)
                if key in found:
                    continue
                info = self._info_for(f)
                entry = ModelEntry(info=info, repo=_infer_repo(f, root))
                found[key] = entry
        self._save_cache()
        with self._lock:
            self.entries = sorted(found.values(),
                                  key=lambda e: (e.info.name or e.name).lower())
        return self.entries

    def _info_for(self, f: Path) -> GGUFInfo:
        try:
            st = f.stat()
        except OSError:
            return read(f)
        key = str(f)
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
        if PART_RE.search(f.name):
            # Shards arrive one at a time; the stamp has to change as they do.
            try:
                siblings = sorted(p.name for p in f.parent.glob("*.gguf"))
                stamp += ":" + str(len(siblings))
            except OSError:
                pass
        hit = self._cache.get(key)
        if hit and hit.get("stamp") == stamp:
            info = GGUFInfo(path=f)
            for k, v in hit["info"].items():
                if hasattr(info, k):
                    setattr(info, k, v)
            info.path = f
            return info
        info = read(f)
        self._cache[key] = {
            "stamp": stamp,
            "info": {
                "name": info.name, "architecture": info.architecture,
                "quant": info.quant, "size_label": info.size_label,
                "n_ctx_train": info.n_ctx_train, "n_layer": info.n_layer,
                "n_embd": info.n_embd, "n_params": info.n_params,
                "file_size": info.file_size, "parts": info.parts,
                "chat_template": info.chat_template[:4000], "error": info.error,
            },
        }
        return info

    def find(self, query: str) -> list[ModelEntry]:
        q = query.lower().strip()
        if not q:
            return self.entries
        return [e for e in self.entries
                if q in (e.info.name or "").lower() or q in e.name.lower()
                or q in (e.info.quant or "").lower()]


def _infer_repo(f: Path, root: Path) -> str:
    """LM Studio and huggingface-cli both encode the repo in the folder path."""
    try:
        rel = f.relative_to(root).parts
    except ValueError:
        return ""
    if rel and rel[0].startswith("models--"):
        return rel[0][len("models--"):].replace("--", "/", 1)
    if len(rel) >= 3:
        return f"{rel[0]}/{rel[1]}"
    return ""


# ------------------------------------------------------------- hugging face --
@dataclass
class RepoFile:
    name: str
    size: int = 0

    @property
    def size_gb(self) -> float:
        return self.size / (1024 ** 3)


@dataclass
class RepoResult:
    id: str
    downloads: int = 0
    likes: int = 0
    updated: str = ""
    files: list[RepoFile] = field(default_factory=list)


def search_repos(query: str, limit: int = 20, timeout: float = 20.0) -> list[RepoResult]:
    """Search Hugging Face for GGUF repositories."""
    params = {"search": query, "filter": "gguf", "limit": limit,
              "sort": "downloads", "direction": -1}
    r = requests.get(f"{HF_API}/models", params=params, timeout=timeout)
    r.raise_for_status()
    out = []
    for item in r.json():
        out.append(RepoResult(
            id=str(item.get("modelId") or item.get("id") or ""),
            downloads=int(item.get("downloads") or 0),
            likes=int(item.get("likes") or 0),
            updated=str(item.get("lastModified") or "")[:10],
        ))
    return [x for x in out if x.id]


def list_repo_files(repo: str, timeout: float = 20.0) -> list[RepoFile]:
    r = requests.get(f"{HF_API}/models/{repo}/tree/main", timeout=timeout,
                     params={"recursive": "true"})
    r.raise_for_status()
    files = []
    for item in r.json():
        path = str(item.get("path") or "")
        if not path.lower().endswith(".gguf"):
            continue
        size = item.get("size") or (item.get("lfs") or {}).get("size") or 0
        files.append(RepoFile(name=path, size=int(size)))
    return sorted(files, key=lambda f: f.name)


def download_url(repo: str, filename: str) -> str:
    return f"{HF_HOST}/{repo}/resolve/main/{filename}?download=true"


def download(repo: str, filename: str, dest_dir: str | Path,
             on_progress: Callable[[int, int], None] | None = None,
             cancel: Callable[[], bool] | None = None,
             token: str = "") -> Path:
    """Stream a file down, resuming if a partial is already there."""
    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(filename).name
    part = target.with_suffix(target.suffix + ".part")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    done = part.stat().st_size if part.exists() else 0
    if done:
        headers["Range"] = f"bytes={done}-"

    with requests.get(download_url(repo, filename), headers=headers,
                      stream=True, timeout=60) as r:
        if done and r.status_code == 416:
            part.rename(target)
            return target
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0) + done
        mode = "ab" if done and r.status_code == 206 else "wb"
        if mode == "wb":
            done = 0
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if cancel and cancel():
                    raise InterruptedError("download cancelled")
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
    part.replace(target)
    return target


def delete_model(entry: "ModelEntry") -> tuple[list[str], list[str]]:
    """Remove a model from disk. Returns (removed, failures).

    Sharded models are several files; deleting only the one that is listed
    leaves the rest as orphans that no longer load and are not obvious to find.
    """
    path = entry.path
    targets = [path]
    match = PART_RE.search(path.name)
    if match:
        stem = path.name[:match.start()]
        targets = sorted(path.parent.glob(f"{stem}-*-of-*.gguf"))
    # The JSON sidecar some quantisers write alongside the weights.
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        targets.append(sidecar)

    removed: list[str] = []
    failures: list[str] = []
    for target in targets:
        try:
            target.unlink()
            removed.append(str(target))
        except OSError as e:
            failures.append(f"{target.name}: {e}")
    return removed, failures


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PB"
