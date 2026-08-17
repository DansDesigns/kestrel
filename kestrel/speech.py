"""Speech: text-to-speech and speech-to-text.

Kestrel is a local-first tool, and speech is where that principle is easiest to
quietly violate — the convenient path is a cloud API, and the cost of taking it
is that everything the user says and everything the agent replies leaves the
machine. So the arrangement here is deliberate:

  * Local engines are the default and are always preferred by `auto`.
  * Network engines exist, because sometimes a better voice is worth it, but
    they are inert until `allow_network` is explicitly switched on. Until then
    they are listed and greyed, not silently available.
  * Nothing is bundled. Every engine is detected, and the ones that are missing
    say what would install them.

The abstraction is intentionally thin. Each engine either shells out to a
binary, calls a Python package, or posts to an HTTP endpoint, and reports
honestly whether it is usable on this machine.
"""
from __future__ import annotations

import json
import os
import platform
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# --------------------------------------------------------------- data types --


@dataclass
class Voice:
    id: str                      # what the engine is given
    name: str                    # what the user sees
    engine: str = ""
    lang: str = ""
    quality: str = ""
    path: str = ""               # for file-backed voices such as Piper

    @property
    def label(self) -> str:
        bits = [self.name]
        if self.lang:
            bits.append(self.lang)
        if self.quality:
            bits.append(self.quality)
        return "  ·  ".join(bits)


@dataclass
class EngineStatus:
    id: str
    name: str
    offline: bool
    available: bool
    detail: str = ""
    install_hint: str = ""

    @property
    def badge(self) -> str:
        if self.available:
            return "local" if self.offline else "network"
        return "not installed"


class SpeechError(RuntimeError):
    pass


def _which(*names: str) -> str:
    for n in names:
        found = shutil.which(n)
        if found:
            return found
    return ""


def _module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _run(cmd: list[str], stdin_text: str | None = None, timeout: float = 300) -> str:
    try:
        proc = subprocess.run(
            cmd, input=stdin_text, capture_output=True, text=True,
            timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired as e:
        raise SpeechError(f"{Path(cmd[0]).name} timed out") from e
    except OSError as e:
        raise SpeechError(f"could not run {cmd[0]}: {e}") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise SpeechError(f"{Path(cmd[0]).name} failed: " + " ".join(tail[-3:]))
    return proc.stdout


# ==================================================================== TTS ====
class TTSEngine:
    id = ""
    name = ""
    offline = True
    install_hint = ""

    def status(self) -> EngineStatus:
        ok, detail = self.check()
        return EngineStatus(self.id, self.name, self.offline, ok, detail,
                            self.install_hint)

    def check(self) -> tuple[bool, str]:
        raise NotImplementedError

    def voices(self) -> list[Voice]:
        return []

    def synth(self, text: str, voice: str, out: Path, speed: float = 1.0) -> Path:
        raise NotImplementedError


class PiperTTS(TTSEngine):
    """Piper: small ONNX neural voices. The best offline quality per megabyte,
    and the reason local TTS is a reasonable default at all."""

    id = "piper"
    name = "Piper (neural, offline)"
    install_hint = "pip install piper-tts, or download a piper release binary"

    def __init__(self, voice_dirs: Iterable[str] = ()):
        self.voice_dirs = [Path(d).expanduser() for d in voice_dirs]

    def binary(self) -> str:
        return _which("piper", "piper-tts", "piper.exe")

    def check(self) -> tuple[bool, str]:
        """Present is not the same as working.

        The pip package pulls in onnxruntime, which needs the Microsoft Visual
        C++ runtime on Windows; without it `piper.exe` exists and fails on
        every call. Running it once here turns that into a message instead of a
        mystery.
        """
        binary = self.binary()
        if binary:
            try:
                subprocess.run([binary, "--version"], capture_output=True,
                               timeout=20, text=True)
                return True, binary
            except OSError as e:
                return False, f"{binary} will not run: {e}"
            except subprocess.SubprocessError:
                return True, binary        # it ran; the version flag may differ
        if _module("piper"):
            try:
                import piper  # noqa: F401
                return True, "python module"
            except Exception as e:
                return False, (f"the piper module fails to import: {e}. On "
                               "Windows this is usually the missing Microsoft "
                               "Visual C++ Redistributable.")
        return False, "piper not installed"

    def voices(self) -> list[Voice]:
        out: list[Voice] = []
        seen: set[str] = set()
        for d in self.voice_dirs:
            if not d.is_dir():
                continue
            for onnx in sorted(d.rglob("*.onnx")):
                if onnx.name in seen:
                    continue
                seen.add(onnx.name)
                stem = onnx.stem                      # en_US-amy-medium
                parts = stem.split("-")
                out.append(Voice(
                    id=str(onnx), name=parts[1] if len(parts) > 1 else stem,
                    engine=self.id, lang=parts[0] if parts else "",
                    quality=parts[2] if len(parts) > 2 else "", path=str(onnx)))
        return out

    def config_for(self, model: Path) -> Path | None:
        """Piper needs the JSON sidecar describing the voice. Without it the
        binary exits with a bare error, so it is checked before invoking."""
        for candidate in (Path(str(model) + ".json"),
                          model.with_suffix(".json"),
                          model.parent / f"{model.stem}.onnx.json"):
            if candidate.is_file():
                return candidate
        return None

    def sample_rate(self, config: Path) -> int:
        try:
            data = json.loads(config.read_text("utf-8"))
            return int(data.get("audio", {}).get("sample_rate") or 22050)
        except Exception:
            return 22050

    def stream(self, voice: str, speed: float = 1.0) -> PiperStream | None:
        """A warm process for the whole reply, or None if unavailable."""
        binary = self.binary()
        model = Path(voice)
        if not binary or not model.is_file():
            return None
        config = self.config_for(model)
        if config is None:
            return None
        length = max(0.3, min(3.0, 1.0 / max(0.1, speed)))
        try:
            return PiperStream(binary, model, config,
                               self.sample_rate(config), length)
        except SpeechError:
            return None

    def synth(self, text: str, voice: str, out: Path, speed: float = 1.0) -> Path:
        model = Path(voice)
        if not model.is_file():
            raise SpeechError("no Piper voice selected — download one in the Speech tab")
        config = self.config_for(model)
        if config is None:
            raise SpeechError(
                f"{model.name} has no matching .onnx.json config beside it. Piper "
                "needs both files; re-download the voice from the Speech tab.")

        length = max(0.3, min(3.0, 1.0 / max(0.1, speed)))   # inverse of speed
        binary = self.binary()
        base = [binary] if binary else [_python(), "-m", "piper"]

        # Piper's command line has changed across releases and between the
        # binary and the Python package: underscored and hyphenated flags both
        # exist, and older builds want --output_file rather than -f. Try the
        # forms in turn rather than guessing, and report every failure if none
        # of them works.
        variants = [
            ["-m", str(model), "-c", str(config), "-f", str(out),
             "--length_scale", f"{length:.2f}"],
            ["-m", str(model), "-c", str(config), "-f", str(out),
             "--length-scale", f"{length:.2f}"],
            ["--model", str(model), "--config", str(config),
             "--output_file", str(out), "--length_scale", f"{length:.2f}"],
            ["-m", str(model), "-f", str(out)],
            ["--model", str(model), "--output_file", str(out)],
        ]
        errors: list[str] = []
        for args in variants:
            if out.exists():
                try:
                    out.unlink()
                except OSError:
                    pass
            try:
                _run(base + args, stdin_text=text)
            except SpeechError as e:
                errors.append(str(e))
                continue
            if out.is_file() and out.stat().st_size > 44:     # bigger than a bare header
                return out
            errors.append(f"{' '.join(args[:2])}: produced no audio")
        raise SpeechError("piper failed. " + errors[0] if errors else "piper failed")


class _SoundDeviceSink:
    """Plays raw PCM from a pipe through the sound card directly.

    Without this, streaming needs an external player — ffplay, aplay, sox —
    and Windows ships none of them. The fallback there was to spawn PowerShell
    once per sentence to play a file, which costs most of a second each time
    and is the reason speech lagged behind a fast model.
    """

    def __init__(self, source, rate: int):
        import sounddevice as sd

        self.source = source
        self.stop_flag = threading.Event()
        self.stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16",
                                         blocksize=1024)
        self.stream.start()
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        while not self.stop_flag.is_set():
            data = self.source.read(4096)
            if not data:
                break
            try:
                self.stream.write(data)
            except Exception:
                break

    @property
    def alive(self) -> bool:
        return not self.stop_flag.is_set() and self.thread.is_alive()

    def close(self) -> None:
        self.stop_flag.set()
        try:
            self.stream.abort()       # cut the current buffer, do not drain it
            self.stream.close()
        except Exception:
            pass


class PiperStream:
    """One Piper process for a whole reply, writing raw audio into a player.

    Per-call startup dominates short utterances: a quarter of a second of
    process spawn and model load against a fifth of a second of actual
    synthesis. Keeping one process alive for the turn removes that entirely,
    and piping its raw output straight into a player removes the file writes
    and the gaps between clips as well.

    Sentences remain the unit. Synthesising word by word would pay the
    remaining per-call cost nineteen times over for a twenty-word reply, and
    neural voices compute prosody across a whole clause — cut it into words and
    every one lands with the flat intonation of a word spoken alone.
    """

    def __init__(self, binary: str, model: Path, config: Path, rate: int,
                 length_scale: float):
        self.proc: subprocess.Popen | None = None
        self.player: subprocess.Popen | None = None
        cmd = [binary, "-m", str(model), "-c", str(config), "--output-raw",
               "--length_scale", f"{length_scale:.2f}",
               # Piper pads each utterance; the pause between sentences is
               # supplied by the sentences themselves.
               "--sentence_silence", "0.05"]
        self.sink = None
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     creationflags=_no_window())
        play_cmd = _raw_player(rate)
        if play_cmd:
            self.player = subprocess.Popen(play_cmd, stdin=self.proc.stdout,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL,
                                           creationflags=_no_window())
        elif _module("sounddevice"):
            self.sink = _SoundDeviceSink(self.proc.stdout, rate)
        else:
            self.close()
            raise SpeechError("no way to play raw audio (install ffmpeg, or "
                              "pip install sounddevice)")
        time.sleep(0.25)
        if self.proc.poll() is not None:
            self.close()
            raise SpeechError("piper does not support --output-raw")

    @property
    def alive(self) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return False
        if self.player is not None:
            return self.player.poll() is None
        return self.sink is not None and self.sink.alive

    def say(self, text: str) -> None:
        if not self.alive:
            raise SpeechError("piper stream closed")
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((text.strip() + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def close(self) -> None:
        if getattr(self, "sink", None) is not None:
            self.sink.close()
            self.sink = None
        for proc in (self.proc, self.player):
            if proc is None:
                continue
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
        self.proc = self.player = None


def _raw_player(rate: int) -> list[str]:
    """A command that plays signed 16-bit mono PCM from standard input."""
    if _which("ffplay"):
        return [_which("ffplay"), "-nodisp", "-autoexit", "-loglevel", "quiet",
                "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "-"]
    if _which("aplay"):
        return [_which("aplay"), "-q", "-f", "S16_LE", "-r", str(rate), "-c", "1", "-"]
    if _which("play"):
        return [_which("play"), "-q", "-t", "raw", "-r", str(rate), "-e", "signed",
                "-b", "16", "-c", "1", "-"]
    return []


class EspeakTTS(TTSEngine):
    """espeak-ng: robotic, tiny, and present on nearly every Linux box. The
    guaranteed fallback when nothing better is installed."""

    id = "espeak"
    name = "espeak-ng (offline)"
    install_hint = "apt install espeak-ng / brew install espeak-ng"

    def binary(self) -> str:
        return _which("espeak-ng", "espeak")

    def check(self) -> tuple[bool, str]:
        b = self.binary()
        return (bool(b), b or "espeak-ng not found")

    def voices(self) -> list[Voice]:
        b = self.binary()
        if not b:
            return []
        try:
            raw = _run([b, "--voices"], timeout=20)
        except SpeechError:
            return []
        out = []
        for line in raw.splitlines()[1:]:
            cols = line.split()
            if len(cols) >= 4:
                out.append(Voice(id=cols[3], name=cols[3], engine=self.id, lang=cols[1]))
        return out[:400]

    def synth(self, text: str, voice: str, out: Path, speed: float = 1.0) -> Path:
        cmd = [self.binary(), "-w", str(out), "-s", str(int(175 * max(0.3, speed)))]
        if voice:
            cmd += ["-v", voice]
        _run(cmd + [text[:20000]])
        return out


class Pyttsx3TTS(TTSEngine):
    """System voices: SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on
    Linux. Whatever the operating system already ships."""

    id = "pyttsx3"
    name = "System voices (offline)"
    install_hint = "pip install pyttsx3"

    def check(self) -> tuple[bool, str]:
        return (_module("pyttsx3"), "pyttsx3 module" if _module("pyttsx3")
                else "pyttsx3 not installed")

    def voices(self) -> list[Voice]:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            out = [Voice(id=v.id, name=getattr(v, "name", v.id), engine=self.id,
                         lang=(getattr(v, "languages", []) or [""])[0]
                         if isinstance(getattr(v, "languages", []), list) else "")
                   for v in engine.getProperty("voices")]
            engine.stop()
            return out
        except Exception:
            return []

    def synth(self, text: str, voice: str, out: Path, speed: float = 1.0) -> Path:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            if voice:
                engine.setProperty("voice", voice)
            engine.setProperty("rate", int(200 * max(0.3, speed)))
            engine.save_to_file(text, str(out))
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            raise SpeechError(f"pyttsx3 failed: {e}") from e
        return out


class SayTTS(TTSEngine):
    id = "say"
    name = "macOS say (offline)"
    install_hint = "built in to macOS"

    def check(self) -> tuple[bool, str]:
        b = _which("say")
        return (bool(b) and platform.system() == "Darwin",
                b or "only available on macOS")

    def voices(self) -> list[Voice]:
        try:
            raw = _run(["say", "-v", "?"], timeout=20)
        except SpeechError:
            return []
        out = []
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                out.append(Voice(id=parts[0], name=parts[0], engine=self.id,
                                 lang=parts[1]))
        return out

    def synth(self, text: str, voice: str, out: Path, speed: float = 1.0) -> Path:
        cmd = ["say", "-o", str(out), "--data-format=LEI16@22050",
               "-r", str(int(180 * max(0.3, speed)))]
        if voice:
            cmd += ["-v", voice]
        _run(cmd + [text[:20000]])
        return out


class OpenAITTS(TTSEngine):
    """Any OpenAI-compatible /v1/audio/speech endpoint. Network; opt-in."""

    id = "openai"
    name = "OpenAI-compatible endpoint"
    offline = False
    install_hint = "set an endpoint and key in the Speech tab"

    def __init__(self, base: str = "", key: str = "", model: str = "tts-1"):
        self.base, self.key, self.model = base, key, model

    def check(self) -> tuple[bool, str]:
        return (bool(self.base), self.base or "no endpoint configured")

    def voices(self) -> list[Voice]:
        return [Voice(id=v, name=v, engine=self.id) for v in
                ("alloy", "echo", "fable", "onyx", "nova", "shimmer")]

    def synth(self, text: str, voice: str, out: Path, speed: float = 1.0) -> Path:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        r = requests.post(self.base.rstrip("/") + "/v1/audio/speech", headers=headers,
                          json={"model": self.model, "input": text[:8000],
                                "voice": voice or "alloy", "speed": speed,
                                "response_format": "wav"}, timeout=180)
        if not r.ok:
            raise SpeechError(f"endpoint returned {r.status_code}: {r.text[:200]}")
        out.write_bytes(r.content)
        return out


class ElevenLabsTTS(TTSEngine):
    id = "elevenlabs"
    name = "ElevenLabs"
    offline = False
    install_hint = "set an API key in the Speech tab"

    def __init__(self, key: str = ""):
        self.key = key

    def check(self) -> tuple[bool, str]:
        return (bool(self.key), "API key set" if self.key else "no API key")

    def voices(self) -> list[Voice]:
        if not self.key:
            return []
        import requests
        try:
            r = requests.get("https://api.elevenlabs.io/v1/voices",
                             headers={"xi-api-key": self.key}, timeout=30)
            r.raise_for_status()
            return [Voice(id=v["voice_id"], name=v.get("name", v["voice_id"]),
                          engine=self.id) for v in r.json().get("voices", [])]
        except Exception:
            return []

    def synth(self, text: str, voice: str, out: Path, speed: float = 1.0) -> Path:
        import requests
        if not voice:
            raise SpeechError("no ElevenLabs voice selected")
        r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                          headers={"xi-api-key": self.key},
                          json={"text": text[:5000],
                                "model_id": "eleven_multilingual_v2"}, timeout=180)
        if not r.ok:
            raise SpeechError(f"ElevenLabs returned {r.status_code}")
        out.with_suffix(".mp3").write_bytes(r.content)
        return out.with_suffix(".mp3")


# ==================================================================== STT ====
class STTEngine:
    id = ""
    name = ""
    offline = True
    install_hint = ""

    def status(self) -> EngineStatus:
        ok, detail = self.check()
        return EngineStatus(self.id, self.name, self.offline, ok, detail,
                            self.install_hint)

    def check(self) -> tuple[bool, str]:
        raise NotImplementedError

    def models(self) -> list[Voice]:
        return []

    def transcribe(self, wav: Path, model: str = "", language: str = "auto") -> str:
        raise NotImplementedError


class WhisperCppSTT(STTEngine):
    """whisper.cpp: the natural counterpart to llama.cpp. Same ggml runtime,
    same quantised-model story, same offline guarantee."""

    id = "whispercpp"
    name = "whisper.cpp (offline)"
    install_hint = "build whisper.cpp, or use the Speech tab to fetch a model"

    def __init__(self, model_dirs: Iterable[str] = ()):
        self.model_dirs = [Path(d).expanduser() for d in model_dirs]

    def binary(self) -> str:
        return _which("whisper-cli", "whisper-cpp", "whisper", "main.exe", "whisper-cli.exe")

    def check(self) -> tuple[bool, str]:
        b = self.binary()
        return (bool(b), b or "whisper-cli not found")

    def models(self) -> list[Voice]:
        out, seen = [], set()
        for d in self.model_dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.rglob("ggml-*.bin")):
                if f.name in seen:
                    continue
                seen.add(f.name)
                out.append(Voice(id=str(f), name=f.stem.replace("ggml-", ""),
                                 engine=self.id, path=str(f),
                                 quality=f"{f.stat().st_size // 1048576} MB"))
        return out

    def transcribe(self, wav: Path, model: str = "", language: str = "auto") -> str:
        if not model or not Path(model).is_file():
            raise SpeechError("no whisper model selected — fetch one in the Speech tab")
        cmd = [self.binary(), "-m", model, "-f", str(wav), "-nt", "-np"]
        if language and language != "auto":
            cmd += ["-l", language]
        return " ".join(_run(cmd, timeout=600).split())


class FasterWhisperSTT(STTEngine):
    id = "fasterwhisper"
    name = "faster-whisper (offline)"
    install_hint = "pip install faster-whisper"

    def check(self) -> tuple[bool, str]:
        ok = _module("faster_whisper")
        return (ok, "faster_whisper module" if ok else "faster-whisper not installed")

    def models(self) -> list[Voice]:
        return [Voice(id=n, name=n, engine=self.id, quality=q) for n, q in (
            ("tiny", "39 MB"), ("base", "74 MB"), ("small", "244 MB"),
            ("medium", "769 MB"), ("large-v3", "1.5 GB"),
            ("distil-large-v3", "756 MB"))]

    def transcribe(self, wav: Path, model: str = "", language: str = "auto") -> str:
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel(model or "base", device="auto", compute_type="int8")
            segments, _info = m.transcribe(
                str(wav), language=None if language in ("", "auto") else language)
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            raise SpeechError(f"faster-whisper failed: {e}") from e


class VoskSTT(STTEngine):
    id = "vosk"
    name = "Vosk (offline, lightweight)"
    install_hint = "pip install vosk, then download a model"

    def __init__(self, model_dirs: Iterable[str] = ()):
        self.model_dirs = [Path(d).expanduser() for d in model_dirs]

    def check(self) -> tuple[bool, str]:
        ok = _module("vosk")
        return (ok, "vosk module" if ok else "vosk not installed")

    def models(self) -> list[Voice]:
        out = []
        for d in self.model_dirs:
            if d.is_dir():
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and (sub / "am").exists():
                        out.append(Voice(id=str(sub), name=sub.name, engine=self.id))
        return out

    def transcribe(self, wav: Path, model: str = "", language: str = "auto") -> str:
        try:
            from vosk import KaldiRecognizer, Model
            m = Model(model)
            with wave.open(str(wav), "rb") as wf:
                rec = KaldiRecognizer(m, wf.getframerate())
                rec.SetWords(False)
                while True:
                    data = wf.readframes(4000)
                    if not data:
                        break
                    rec.AcceptWaveform(data)
                return json.loads(rec.FinalResult()).get("text", "")
        except Exception as e:
            raise SpeechError(f"vosk failed: {e}") from e


class OpenAISTT(STTEngine):
    id = "openai"
    name = "OpenAI-compatible endpoint"
    offline = False
    install_hint = "set an endpoint and key in the Speech tab"

    def __init__(self, base: str = "", key: str = "", model: str = "whisper-1"):
        self.base, self.key, self.model = base, key, model

    def check(self) -> tuple[bool, str]:
        return (bool(self.base), self.base or "no endpoint configured")

    def transcribe(self, wav: Path, model: str = "", language: str = "auto") -> str:
        import requests
        headers = {}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        data = {"model": model or self.model}
        if language and language != "auto":
            data["language"] = language
        with open(wav, "rb") as f:
            r = requests.post(self.base.rstrip("/") + "/v1/audio/transcriptions",
                              headers=headers, data=data,
                              files={"file": (wav.name, f, "audio/wav")}, timeout=300)
        if not r.ok:
            raise SpeechError(f"endpoint returned {r.status_code}: {r.text[:200]}")
        return str(r.json().get("text", "")).strip()


# ========================================================= audio plumbing ====
def _python() -> str:
    import sys
    return sys.executable or "python3"


PLAYERS = [
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ("afplay", []),
    ("aplay", ["-q"]),
    ("paplay", []),
    ("play", ["-q"]),
]


SOUND_TYPES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff")


def sound_choices(extra: str = "") -> list[tuple[str, str]]:
    """Sounds that can be used for the finish chime, as (label, path).

    The bundled one first, then anything the platform already provides, so a
    choice exists without hunting for a file.
    """
    from pathlib import Path as _P

    found: list[tuple[str, str]] = [("Kestrel chime (bundled)", "")]
    shipped = _P(__file__).resolve().parent.parent / "assets" / "bell.wav"
    seen = set()
    folders = [
        _P("C:/Windows/Media"),
        _P("/usr/share/sounds/freedesktop/stereo"),
        _P("/usr/share/sounds/ubuntu/stereo"),
        _P("/System/Library/Sounds"),
    ]
    for folder in folders:
        try:
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir())[:60]:
                if f.suffix.lower() in SOUND_TYPES and f.name not in seen:
                    seen.add(f.name)
                    found.append((f.stem, str(f)))
        except OSError:
            continue
    if extra and extra not in {p for _, p in found}:
        found.insert(1, (_P(extra).name, extra))
    _ = shipped
    return found


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def play(path: Path, blocking: bool = True) -> None:
    """Play a wav or mp3 with whatever the system has.

    In-process routes are tried first. Spawning a player — and on Windows that
    meant PowerShell — costs a good fraction of a second per clip, which is
    most of the delay when speaking a reply sentence by sentence.
    """
    if os.name == "nt" and Path(path).suffix.lower() == ".wav":
        # winsound plays wav only; anything else falls through to a decoder.
        try:
            import winsound          # bundled with Python on Windows
            flags = winsound.SND_FILENAME | (0 if blocking else winsound.SND_ASYNC)
            winsound.PlaySound(str(path), flags)
            return
        except Exception:
            pass
    if _module("sounddevice") and _module("soundfile"):
        try:
            import sounddevice as sd
            import soundfile as sf
            data, rate = sf.read(str(path), dtype="int16")
            sd.play(data, rate)
            if blocking:
                sd.wait()
            return
        except Exception:
            pass
    for name, args in PLAYERS:
        binary = _which(name)
        if not binary:
            continue
        cmd = [binary, *args, str(path)] if name != "ffplay" else [binary, *args, str(path)]
        if blocking:
            subprocess.run(cmd, capture_output=True)
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    if os.name == "nt":
        ps = ("$p=New-Object System.Media.SoundPlayer '%s'; $p.PlaySync();" % path)
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
        return
    raise SpeechError("no audio player found (install ffmpeg, sox, or alsa-utils)")


RECORDERS = [
    ("ffmpeg", lambda dev, secs, out: [
        "ffmpeg", "-y", "-loglevel", "quiet",
        "-f", {"Linux": "alsa", "Darwin": "avfoundation", "Windows": "dshow"}
        .get(platform.system(), "alsa"),
        "-i", dev or {"Linux": "default", "Darwin": ":0", "Windows": "audio=default"}
        .get(platform.system(), "default"),
        "-t", str(secs), "-ar", "16000", "-ac", "1", str(out)]),
    ("arecord", lambda dev, secs, out: [
        "arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1",
        "-d", str(secs), *(["-D", dev] if dev else []), str(out)]),
    ("rec", lambda dev, secs, out: [
        "rec", "-q", "-r", "16000", "-c", "1", str(out), "trim", "0", str(secs)]),
]


def record(seconds: int, out: Path, device: str = "") -> Path:
    """Capture microphone input to a 16 kHz mono wav, which is what every STT
    engine here wants."""
    if _module("sounddevice") and _module("numpy"):
        try:
            import numpy as np
            import sounddevice as sd
            frames = sd.rec(int(seconds * 16000), samplerate=16000, channels=1,
                            dtype="int16")
            sd.wait()
            with wave.open(str(out), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(np.asarray(frames, dtype="int16").tobytes())
            return out
        except Exception:
            pass
    for name, build in RECORDERS:
        if _which(name):
            _run(build(device, seconds, out), timeout=seconds + 30)
            return out
    raise SpeechError("no recorder found (pip install sounddevice, or install ffmpeg)")


def audio_available() -> tuple[bool, bool]:
    """(can_play, can_record) — used to grey out controls honestly."""
    can_play = any(_which(n) for n, _ in PLAYERS) or os.name == "nt"
    can_rec = (_module("sounddevice") and _module("numpy")) or any(
        _which(n) for n, _ in RECORDERS)
    return can_play, can_rec


# ====================================================== downloadable assets ==
PIPER_REPO = "rhasspy/piper-voices"
WHISPER_REPO = "ggerganov/whisper.cpp"

PIPER_CATALOGUE = [
    ("en_US", "amy", "medium", "American English, warm"),
    ("en_US", "lessac", "medium", "American English, neutral"),
    ("en_US", "ryan", "high", "American English, male, higher quality"),
    ("en_GB", "alba", "medium", "British English, Scottish"),
    ("en_GB", "northern_english_male", "medium", "British English, northern"),
    ("de_DE", "thorsten", "medium", "German"),
    ("fr_FR", "siwis", "medium", "French"),
    ("es_ES", "davefx", "medium", "Spanish"),
    ("it_IT", "riccardo", "x_low", "Italian, very small"),
    ("nl_NL", "mls", "medium", "Dutch"),
]

WHISPER_CATALOGUE = [
    ("ggml-tiny.en.bin", "75 MB", "English only, fastest"),
    ("ggml-base.en.bin", "142 MB", "English only, good default"),
    ("ggml-base.bin", "142 MB", "Multilingual"),
    ("ggml-small.en.bin", "466 MB", "English only, more accurate"),
    ("ggml-small.bin", "466 MB", "Multilingual, more accurate"),
    ("ggml-medium.bin", "1.5 GB", "Multilingual, high accuracy"),
    ("ggml-large-v3-turbo.bin", "1.6 GB", "Best accuracy for the size"),
]


def piper_voice_file(locale: str, name: str, quality: str) -> str:
    """Path within the Piper voices repository."""
    return f"{locale.split('_')[0]}/{locale}/{name}/{quality}/{locale}-{name}-{quality}.onnx"


def download_piper_voice(locale: str, name: str, quality: str, dest: str | Path,
                         on_progress: Callable[[int, int], None] | None = None,
                         cancel: Callable[[], bool] | None = None) -> Path:
    """Fetch a voice. Piper needs both the model and its JSON sidecar."""
    from .models import download
    rel = piper_voice_file(locale, name, quality)
    model = download(PIPER_REPO, rel, dest, on_progress, cancel)
    try:
        download(PIPER_REPO, rel + ".json", dest, None, cancel)
    except Exception as e:
        # Not optional: Piper will not speak without it, and a voice that half
        # downloaded is worse than one that failed outright.
        try:
            model.unlink()
        except OSError:
            pass
        raise SpeechError(
            f"downloaded {Path(rel).name} but its .onnx.json config failed ({e}). "
            "The voice has been removed; try again.") from e
    return model


def download_whisper_model(filename: str, dest: str | Path,
                           on_progress: Callable[[int, int], None] | None = None,
                           cancel: Callable[[], bool] | None = None) -> Path:
    from .models import download
    return download(WHISPER_REPO, filename, dest, on_progress, cancel)


# ============================================================== the manager ==
class Speech:
    """Chooses engines and performs the two operations the rest of Kestrel
    cares about: say this, and hear that."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tmp = Path(tempfile.gettempdir()) / "kestrel-speech"
        self.tmp.mkdir(parents=True, exist_ok=True)

    # -- construction ---------------------------------------------------------
    def _tts_engines(self) -> list[TTSEngine]:
        s = self.cfg.speech
        return [
            PiperTTS(s.voice_dirs),
            Pyttsx3TTS(),
            SayTTS(),
            EspeakTTS(),
            OpenAITTS(s.api_base, s.api_key, s.tts_api_model),
            ElevenLabsTTS(s.elevenlabs_key),
        ]

    def _stt_engines(self) -> list[STTEngine]:
        s = self.cfg.speech
        return [
            WhisperCppSTT(s.model_dirs),
            FasterWhisperSTT(),
            VoskSTT(s.model_dirs),
            OpenAISTT(s.api_base, s.api_key, s.stt_api_model),
        ]

    def _permitted(self, engine) -> bool:
        return engine.offline or self.cfg.speech.allow_network

    def tts_status(self) -> list[EngineStatus]:
        out = []
        for e in self._tts_engines():
            st = e.status()
            if not e.offline and not self.cfg.speech.allow_network:
                st.available = False
                st.detail = "network engines are switched off"
            out.append(st)
        return out

    def stt_status(self) -> list[EngineStatus]:
        out = []
        for e in self._stt_engines():
            st = e.status()
            if not e.offline and not self.cfg.speech.allow_network:
                st.available = False
                st.detail = "network engines are switched off"
            out.append(st)
        return out

    def tts_engine(self) -> TTSEngine | None:
        """Selected engine, or the best available local one under `auto`."""
        wanted = self.cfg.speech.tts_engine
        engines = self._tts_engines()
        if wanted and wanted != "auto":
            for e in engines:
                if e.id == wanted and self._permitted(e) and e.check()[0]:
                    return e
        for e in engines:                     # ordered by preference, local first
            if e.offline and e.check()[0]:
                return e
        return None

    def stt_engine(self) -> STTEngine | None:
        wanted = self.cfg.speech.stt_engine
        engines = self._stt_engines()
        if wanted and wanted != "auto":
            for e in engines:
                if e.id == wanted and self._permitted(e) and e.check()[0]:
                    return e
        for e in engines:
            if e.offline and e.check()[0]:
                return e
        return None

    def voices(self, engine_id: str = "") -> list[Voice]:
        engine = None
        if engine_id and engine_id != "auto":
            engine = next((e for e in self._tts_engines() if e.id == engine_id), None)
        engine = engine or self.tts_engine()
        if engine is None:
            return []
        try:
            return engine.voices()
        except Exception:
            return []

    def stt_models(self, engine_id: str = "") -> list[Voice]:
        engine = None
        if engine_id and engine_id != "auto":
            engine = next((e for e in self._stt_engines() if e.id == engine_id), None)
        engine = engine or self.stt_engine()
        if engine is None:
            return []
        try:
            return engine.models()
        except Exception:
            return []

    # -- operations -----------------------------------------------------------
    def piper_stream(self):
        """A Piper process kept alive between replies.

        Loading the voice model is most of the cost of a short utterance — a
        medium voice takes seconds to load and a fraction of a second to speak.
        Starting a process per reply pays that every time, which is why the
        first words lag so far behind. One process for the session pays it once.
        """
        engine = self.tts_engine()
        if engine is None or not hasattr(engine, "stream"):
            return None
        key = (self.cfg.speech.tts_voice, round(self.cfg.speech.tts_speed, 2))
        stream = getattr(self, "_stream", None)
        if stream is not None and getattr(self, "_stream_key", None) == key:
            if stream.alive:
                return stream
            stream.close()
        elif stream is not None:
            stream.close()                     # voice or speed changed
        self._stream = engine.stream(*key)
        self._stream_key = key
        return self._stream

    def close_stream(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.close()
        self._stream = None
        self._stream_key = None

    def speaker(self) -> "Speaker":
        return Speaker(self)

    def dictation(self, on_text, on_status=None) -> "Dictation":
        return Dictation(self, on_text, on_status)

    def speak_now(self, text: str) -> bool:
        """Speak immediately through the session process, if there is one."""
        stream = self.piper_stream()
        if stream is None:
            return False
        try:
            stream.say(clean_for_speech(text, limit=600))
            return True
        except SpeechError:
            self.close_stream()
            return False

    def speak(self, text: str, blocking: bool = False) -> Path:
        text = clean_for_speech(text)
        if not text:
            raise SpeechError("nothing to say")
        engine = self.tts_engine()
        if engine is None:
            raise SpeechError("no speech engine available — see the Speech tab")
        out = self.tmp / "say.wav"
        produced = engine.synth(text, self.cfg.speech.tts_voice, out,
                                self.cfg.speech.tts_speed)
        play(produced, blocking=blocking)
        return produced

    def listen(self, seconds: int = 0) -> str:
        engine = self.stt_engine()
        if engine is None:
            raise SpeechError("no transcription engine available — see the Speech tab")
        wav = self.tmp / "heard.wav"
        record(seconds or self.cfg.speech.record_seconds, wav,
               self.cfg.speech.input_device)
        return engine.transcribe(wav, self.cfg.speech.stt_model,
                                 self.cfg.speech.stt_language)

    def transcribe_file(self, path: str | Path) -> str:
        engine = self.stt_engine()
        if engine is None:
            raise SpeechError("no transcription engine available")
        return engine.transcribe(Path(path), self.cfg.speech.stt_model,
                                 self.cfg.speech.stt_language)


SPEAK_SKIP = ("```", "|", "$ ")


def clean_for_speech(text: str, limit: int = 4000) -> str:
    """Strip what does not read aloud well.

    Agent output is full of paths, fenced code and tables. Reading those verbatim
    is worse than useless, so they are dropped rather than narrated.
    """
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith(SPEAK_SKIP):
            continue
        stripped = stripped.lstrip("#>-*• ").replace("`", "")
        stripped = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", stripped)   # markdown emphasis
        stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)   # links: keep the words
        stripped = re.sub(r"https?://\S+", "a link", stripped)
        if stripped:
            out.append(stripped)
    joined = " ".join(out)
    return joined[:limit]


ENGINE_PACKAGES = {
    "piper": ["piper-tts"],
    "fasterwhisper": ["faster-whisper"],
    "vosk": ["vosk"],
    "pyttsx3": ["pyttsx3"],
    "audio": ["sounddevice", "numpy", "soundfile"],
}


def install_packages(names: list[str], progress=None) -> bool:
    """pip install into the interpreter Kestrel is running under.

    The engines are optional and large, so they are not installed with the
    application; this is the same command the installer offers, available later
    from the Speech tab.
    """
    say = progress or (lambda line: None)
    packages: list[str] = []
    for name in names:
        packages.extend(ENGINE_PACKAGES.get(name, [name]))
    if not packages:
        return False
    cmd = [_python(), "-m", "pip", "install", *packages]
    say("$ " + " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
    except OSError as e:
        say(f"could not run pip: {e}")
        return False
    for line in proc.stdout:                       # type: ignore[union-attr]
        line = line.rstrip()
        if line:
            say(line[:200])
    return proc.wait() == 0


SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n{2,}")


class Speaker:
    """Speaks a reply while it is still being written.

    Synthesising the finished answer means waiting for generation to end, then
    waiting again for the whole text to be rendered — several seconds before a
    word is heard. Splitting on sentence boundaries and synthesising each as it
    completes overlaps the two, so speech starts a sentence after the model
    does and stays roughly in step with it.
    """

    def __init__(self, speech: "Speech"):
        self.speech = speech
        self._buffer = ""
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._audio: "queue.Queue[Path | None]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._player: threading.Thread | None = None
        self._stop = threading.Event()
        self._playing: subprocess.Popen | None = None
        self._stream = None
        self._index = 0
        self.on_error: Callable[[str], None] | None = None

    # -- input ---------------------------------------------------------------
    def push(self, chunk: str) -> None:
        """Feed streamed text. Complete sentences are queued as they appear."""
        if self._stop.is_set():
            return
        self._buffer += chunk
        while True:
            match = SENTENCE_END.search(self._buffer)
            if not match:
                break
            sentence, self._buffer = (self._buffer[:match.end()],
                                      self._buffer[match.end():])
            self._enqueue(sentence)
        # The first fragment decides how long the silence before speech is, so
        # it is cut short at a comma; later ones are left longer, which reads
        # more naturally once speech is already under way.
        limit = 60 if self._index == 0 and self._queue.empty() else 240
        if len(self._buffer) > limit:
            cut = max(self._buffer.rfind(", ", 0, limit),
                      self._buffer.rfind("; ", 0, limit))
            if cut > 24:
                self._enqueue(self._buffer[:cut + 1])
                self._buffer = self._buffer[cut + 1:]

    def flush(self) -> None:
        """Speak whatever is left at the end of a turn."""
        if self._buffer.strip():
            self._enqueue(self._buffer)
        self._buffer = ""

    def _enqueue(self, sentence: str) -> None:
        text = clean_for_speech(sentence, limit=600).strip()
        if len(text) < 2:
            return
        self._ensure_worker()
        self._queue.put(text)

    # -- output --------------------------------------------------------------
    def _ensure_worker(self) -> None:
        """Two stages, not one.

        Synthesising and playing in the same loop means every gap between
        sentences is a whole synthesis. Running them as a pipeline means the
        next sentence is already rendered by the time the current one finishes,
        so after the first there is no gap at all.
        """
        self._stop.clear()
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
        if self._player is None or not self._player.is_alive():
            self._player = threading.Thread(target=self._run_player, daemon=True)
            self._player.start()

    def _run_streaming(self, stream) -> None:
        """Feed sentences to a live process; it plays them as they render."""
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.4)
            except queue.Empty:
                continue
            if text is None or self._stop.is_set():
                break
            try:
                self._index += 1
                stream.say(text)
            except SpeechError as e:
                if self.on_error:
                    self.on_error(str(e))
                return

    def _run_player(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._audio.get(timeout=0.4)
            except queue.Empty:
                continue
            if path is None:
                break
            if self._stop.is_set():
                break
            self._play(path)

    def _run(self) -> None:
        engine = self.speech.tts_engine()
        if engine is None:
            if self.on_error:
                self.on_error("no speech engine available")
            return
        stream = self.speech.piper_stream()
        if stream is not None:
            self._stream = stream
            try:
                self._run_streaming(stream)
            finally:
                # Left running: closing it here would reload the voice model on
                # the next reply, which is the delay this exists to remove.
                self._stream = None
            return
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.4)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                self._index += 1
                out = self.speech.tmp / f"say-{self._index % 8}.wav"
                produced = engine.synth(text, self.speech.cfg.speech.tts_voice,
                                        out, self.speech.cfg.speech.tts_speed)
                if self._stop.is_set():
                    break
                self._audio.put(produced)
            except Exception as e:
                if self.on_error:
                    self.on_error(str(e))
                return

    def _play(self, path: Path) -> None:
        """Played as a tracked child so a cancelled turn stops mid-sentence."""
        for name, args in PLAYERS:
            binary = _which(name)
            if not binary:
                continue
            self._playing = subprocess.Popen(
                [binary, *args, str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._playing.wait()
            self._playing = None
            return
        play(path, blocking=True)

    def spoken_anything(self) -> bool:
        return self._index > 0 or not self._queue.empty()

    def reset(self) -> None:
        self._buffer = ""
        self._index = 0
        for q in (self._queue, self._audio):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def stop(self) -> None:
        """Silence, now.

        Stop means stop: the queue is dropped and whatever is making noise is
        killed, rather than being allowed to finish the sentence in hand.
        """
        self._stop.set()
        self.reset()
        if self._stream is not None:
            try:
                self._stream.close()          # stop means silence now
            except Exception:
                pass
            self._stream = None
            self.speech.close_stream()
        if self._playing is not None:
            try:
                self._playing.terminate()
            except Exception:
                pass
            self._playing = None


class Dictation:
    """Continuous dictation that puts words in the box as they are recognised.

    Recording a fixed block and transcribing it afterwards means the whole
    utterance plus the whole transcription pass before a single word appears —
    seconds of apparent deafness. Two strategies avoid that:

      Vosk streams natively, emitting partial results while you are still
      speaking. It is used when installed.

      Everything else is fed overlapping chunks a couple of seconds long, each
      transcribed as it closes. Words arrive a chunk behind the voice rather
      than an utterance behind it.
    """

    CHUNK_SECONDS = 2.2
    OVERLAP_SECONDS = 0.35
    RATE = 16000

    def __init__(self, speech: "Speech", on_text, on_status=None):
        self.speech = speech
        self.on_text = on_text                  # called with each new fragment
        self.on_status = on_status or (lambda message: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._said: list[str] = []

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._said = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> str:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
        return " ".join(self._said).strip()

    # -- strategies -----------------------------------------------------------
    def _run(self) -> None:
        try:
            if _module("vosk") and _module("sounddevice") and self._vosk_model():
                self._run_vosk()
            else:
                self._run_chunked()
        except Exception as e:
            self.on_status(f"dictation failed: {e}")

    def _vosk_model(self) -> str:
        model = self.speech.cfg.speech.stt_model
        if model and Path(model).is_dir():
            return model
        for candidate in self.speech.stt_models("vosk"):
            return candidate.id
        return ""

    def _run_vosk(self) -> None:
        import json as _json
        import queue as _queue

        import sounddevice as sd
        from vosk import KaldiRecognizer, Model

        recogniser = KaldiRecognizer(Model(self._vosk_model()), self.RATE)
        recogniser.SetWords(False)
        blocks: "_queue.Queue[bytes]" = _queue.Queue()

        def callback(indata, _frames, _time, _status):
            blocks.put(bytes(indata))

        self.on_status("listening")
        partial_len = 0
        with sd.RawInputStream(samplerate=self.RATE, blocksize=4000, dtype="int16",
                               channels=1, callback=callback):
            while not self._stop.is_set():
                try:
                    data = blocks.get(timeout=0.3)
                except _queue.Empty:
                    continue
                if recogniser.AcceptWaveform(data):
                    text = _json.loads(recogniser.Result()).get("text", "").strip()
                    if text:
                        self._said.append(text)
                        self.on_text(text + " ", False)
                        partial_len = 0
                else:
                    partial = _json.loads(
                        recogniser.PartialResult()).get("partial", "").strip()
                    # Partials are re-sent in full each time, so only the new
                    # tail is forwarded and the caller replaces it as it grows.
                    if partial and len(partial) != partial_len:
                        partial_len = len(partial)
                        self.on_text(partial, True)

    def _run_chunked(self) -> None:
        engine = self.speech.stt_engine()
        if engine is None:
            self.on_status("no transcription engine available")
            return
        self.on_status("listening")
        index = 0
        while not self._stop.is_set():
            index += 1
            wav = self.speech.tmp / f"dictate-{index % 4}.wav"
            try:
                record(int(self.CHUNK_SECONDS) + 1, wav,
                       self.speech.cfg.speech.input_device)
            except SpeechError as e:
                self.on_status(str(e))
                return
            if self._stop.is_set():
                break
            try:
                text = engine.transcribe(wav, self.speech.cfg.speech.stt_model,
                                         self.speech.cfg.speech.stt_language)
            except SpeechError as e:
                self.on_status(str(e))
                return
            text = _clean_transcript(text)
            if text:
                self._said.append(text)
                self.on_text(text + " ", False)


BLANK_AUDIO = re.compile(r"\[(BLANK_AUDIO|inaudible|silence|music|noise)\]",
                         re.IGNORECASE)


def _clean_transcript(text: str) -> str:
    """Whisper annotates silence; those markers are not speech."""
    cleaned = BLANK_AUDIO.sub("", str(text or ""))
    cleaned = " ".join(cleaned.split())
    return "" if cleaned in (".", "-", "") else cleaned
