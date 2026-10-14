"""Experiments run as `python -m hivemind run ...` in a child process.

A separate process keeps a long run isolated from the web server, lets Cancel stop
it cleanly on every platform, and reuses the runner's resume: relaunching the same
config skips tasks already in its outcomes.jsonl. Progress is read from that file.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path


class ExperimentJob:
    def __init__(self, job_id: str, command: list[str], out_dir: Path, total: int,
                 cwd: Path | None = None, meta: dict | None = None):
        self.id = job_id
        self.command = command
        self.out_dir = Path(out_dir)
        self.total = total
        self.cwd = cwd
        self.meta = meta or {}
        self.status = "pending"
        self.returncode: int | None = None
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.process: subprocess.Popen | None = None
        self._log: deque[str] = deque(maxlen=200)
        self._cancelled = False
        self._lock = threading.Lock()

    def start(self) -> "ExperimentJob":
        self.started_at = time.time()
        self.status = "running"
        self.process = subprocess.Popen(
            self.command, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._follow, name=f"job-{self.id}", daemon=True).start()
        return self

    def _follow(self) -> None:
        for line in self.process.stdout:
            with self._lock:
                self._log.append(line.rstrip("\n"))
        self.returncode = self.process.wait()
        self.ended_at = time.time()
        if self._cancelled:
            self.status = "cancelled"
        else:
            self.status = "done" if self.returncode == 0 else "failed"

    def cancel(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self._cancelled = True
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed", "cancelled")

    def _tasks(self) -> list[dict]:
        path = self.out_dir / "outcomes.jsonl"
        if not path.exists():
            return []
        tasks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                outcome = json.loads(line)
            except ValueError:
                continue  # the runner may be halfway through writing this line
            tasks.append({"task_id": outcome["task_id"],
                          "passed": {k: v["passed"] for k, v in outcome.get("answers", {}).items()}})
        return tasks

    def progress(self) -> dict:
        tasks = self._tasks()
        with self._lock:
            log = list(self._log)[-60:]
        end = self.ended_at or time.time()
        return {"kind": "experiment", "id": self.id, "status": self.status,
                "done": len(tasks), "total": max(self.total, len(tasks)), "tasks": tasks,
                "log": log, "returncode": self.returncode, "out_dir": str(self.out_dir),
                "elapsed_s": round(end - self.started_at, 1) if self.started_at else 0.0,
                **self.meta}

    def summary(self) -> dict:
        p = self.progress()
        return {k: p[k] for k in ("kind", "id", "status", "done", "total", "elapsed_s")} | self.meta
