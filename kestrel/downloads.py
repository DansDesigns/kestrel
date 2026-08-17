"""Downloads that survive being ignored.

Model files are large enough that a download is something you start and then go
and do something else. That means it has to run in the background, several at a
time, and be pausable — a 30 GB file on a shared connection is not something to
be forced into finishing in one go.

Resuming is done with an HTTP range request against a `.part` file, which is
also what makes pausing safe: the bytes already on disk are the state, so
nothing is lost by stopping, closing the window, or losing the connection.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

QUEUED, RUNNING, PAUSED, DONE, FAILED, CANCELLED = (
    "queued", "running", "paused", "done", "failed", "cancelled")
ACTIVE = (QUEUED, RUNNING, PAUSED)


@dataclass
class Job:
    id: int
    repo: str
    filename: str
    dest: str
    total: int = 0
    done: int = 0
    state: str = QUEUED
    error: str = ""
    speed: float = 0.0            # bytes per second, smoothed
    started: float = 0.0
    _pause: threading.Event = field(default_factory=threading.Event)
    _cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def name(self) -> str:
        return Path(self.filename).name

    @property
    def part_path(self) -> Path:
        target = Path(self.dest).expanduser() / self.name
        return target.with_suffix(target.suffix + ".part")

    @property
    def final_path(self) -> Path:
        return Path(self.dest).expanduser() / self.name

    @property
    def percent(self) -> float:
        return 100.0 * self.done / self.total if self.total else 0.0

    @property
    def remaining(self) -> str:
        if self.state != RUNNING or self.speed <= 0 or not self.total:
            return ""
        seconds = (self.total - self.done) / self.speed
        if seconds < 90:
            return f"{int(seconds)}s left"
        if seconds < 5400:
            return f"{int(seconds // 60)}m left"
        return f"{seconds / 3600:.1f}h left"

    def summary(self) -> str:
        from .models import human_size
        if self.state == DONE:
            return f"{human_size(self.total or self.done)} · complete"
        if self.state == FAILED:
            return f"failed: {self.error[:60]}"
        if self.state == CANCELLED:
            return "cancelled"
        size = (f"{human_size(self.done)} / {human_size(self.total)}"
                if self.total else human_size(self.done))
        if self.state == PAUSED:
            return f"{size} · paused"
        rate = f" · {human_size(self.speed)}/s" if self.speed > 0 else ""
        left = f" · {self.remaining}" if self.remaining else ""
        return f"{size}{rate}{left}"


class DownloadManager:
    """Runs a handful of downloads at once and lets them be stopped.

    Deliberately independent of the interface: downloads continue while the
    window that started them is closed, and the window rebuilds its view from
    the job list whenever it is opened again.
    """

    def __init__(self, max_concurrent: int = 2, token: str = "",
                 state_path: str | Path = ""):
        self.jobs: list[Job] = []
        self.max_concurrent = max(1, max_concurrent)
        self.token = token
        self.on_change: Callable[[], None] | None = None
        self.state_path = Path(state_path) if state_path else None
        self._next_id = 0
        self._lock = threading.Lock()
        self._restore()

    # -- surviving a restart -------------------------------------------------
    def _restore(self) -> None:
        """Bring back downloads that were still going when Kestrel closed.

        The bytes are already on disk in a `.part` file; without this the only
        record of what they belong to dies with the process, and a half-finished
        30 GB download becomes an orphan nobody can resume.
        """
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        for entry in raw.get("jobs", []):
            try:
                self._next_id += 1
                job = Job(id=self._next_id, repo=entry["repo"],
                          filename=entry["filename"], dest=entry["dest"],
                          total=int(entry.get("total") or 0),
                          state=PAUSED)
            except (KeyError, TypeError, ValueError):
                continue
            if job.final_path.exists():
                continue                      # it finished after all
            if not job.part_path.exists():
                continue                      # nothing to resume from
            job.done = job.part_path.stat().st_size
            # Restored paused, never running: resuming is a decision, and a
            # download that starts itself on launch is a surprise.
            self.jobs.append(job)

    def save_state(self) -> None:
        if self.state_path is None:
            return
        keep = [{"repo": j.repo, "filename": j.filename, "dest": j.dest,
                 "total": j.total, "done": j.done}
                for j in self.jobs if j.state in ACTIVE]
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            if keep:
                self.state_path.write_text(json.dumps({"jobs": keep}, indent=1),
                                           "utf-8")
            else:
                self.state_path.unlink(missing_ok=True)
        except OSError:
            pass

    # -- queue ---------------------------------------------------------------
    def add(self, repo: str, filename: str, dest: str) -> Job:
        with self._lock:
            self._next_id += 1
            job = Job(id=self._next_id, repo=repo, filename=filename, dest=dest)
            existing = job.part_path
            if existing.exists():
                job.done = existing.stat().st_size
            self.jobs.append(job)
        self._changed()
        self._schedule()
        return job

    def get(self, job_id: int) -> Job | None:
        return next((j for j in self.jobs if j.id == job_id), None)

    def running_count(self) -> int:
        return sum(1 for j in self.jobs if j.state == RUNNING)

    def active(self) -> list[Job]:
        return [j for j in self.jobs if j.state in ACTIVE]

    def _schedule(self) -> None:
        """Start whatever the concurrency limit allows."""
        with self._lock:
            free = self.max_concurrent - self.running_count()
            starting = [j for j in self.jobs if j.state == QUEUED][:max(0, free)]
            for job in starting:
                job.state = RUNNING
                job.started = time.time()
        for job in starting:
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
        if starting:
            self._changed()

    # -- controls ------------------------------------------------------------
    def pause(self, job_id: int) -> None:
        job = self.get(job_id)
        if job and job.state in (RUNNING, QUEUED):
            job._pause.set()
            job.state = PAUSED
            self._changed()

    def resume(self, job_id: int) -> None:
        job = self.get(job_id)
        if job and job.state in (PAUSED, FAILED):
            job._pause.clear()
            job._cancel.clear()
            job.error = ""
            job.state = QUEUED
            self._changed()
            self._schedule()

    def cancel(self, job_id: int, remove_part: bool = True) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job._cancel.set()
        job._pause.clear()
        job.state = CANCELLED
        if remove_part:
            # Only on an explicit cancel: pausing must keep the bytes, which is
            # the entire point of being able to pause.
            try:
                job.part_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._changed()
        self._schedule()

    def clear_finished(self) -> int:
        before = len(self.jobs)
        self.jobs = [j for j in self.jobs if j.state in ACTIVE]
        self._changed()
        return before - len(self.jobs)

    def stop_all(self) -> None:
        for job in self.jobs:
            job._cancel.set()

    # -- transfer ------------------------------------------------------------
    def _run(self, job: Job) -> None:
        import requests

        from .models import download_url

        try:
            Path(job.dest).expanduser().mkdir(parents=True, exist_ok=True)
            part = job.part_path
            done = part.stat().st_size if part.exists() else 0
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            if done:
                headers["Range"] = f"bytes={done}-"

            with requests.get(download_url(job.repo, job.filename), headers=headers,
                              stream=True, timeout=60) as response:
                if done and response.status_code == 416:
                    part.replace(job.final_path)         # already complete
                    job.done = job.total = done
                    job.state = DONE
                    self._changed()
                    self._schedule()
                    return
                response.raise_for_status()
                length = int(response.headers.get("Content-Length") or 0)
                partial = response.status_code == 206
                if not partial:
                    done = 0                              # server ignored the range
                job.total = length + (done if partial else 0)
                job.done = done

                last_time, last_bytes = time.time(), done
                with open(part, "ab" if partial and done else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if job._cancel.is_set():
                            return
                        if job._pause.is_set():
                            job.state = PAUSED
                            job.speed = 0.0
                            self._changed()
                            return                       # bytes stay on disk
                        if not chunk:
                            continue
                        handle.write(chunk)
                        job.done += len(chunk)
                        now = time.time()
                        if now - last_time >= 0.5:
                            rate = (job.done - last_bytes) / (now - last_time)
                            job.speed = rate if job.speed <= 0 else (
                                0.7 * job.speed + 0.3 * rate)
                            last_time, last_bytes = now, job.done
                            self._changed()

            if job._cancel.is_set() or job._pause.is_set():
                return
            part.replace(job.final_path)
            job.total = job.total or job.done
            job.state = DONE
            job.speed = 0.0
        except Exception as e:
            if not job._cancel.is_set():
                job.state = FAILED
                job.error = str(e)
        finally:
            self._changed()
            self._schedule()

    def _changed(self) -> None:
        self.save_state()
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                pass
