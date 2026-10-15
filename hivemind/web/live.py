"""A live run: one task solved in a background thread, every event kept for replay.

Readers poll `wait_events(after=seq)`, so a page that reconnects (or opens late)
gets the whole history first and then each new event as it happens.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ..agents import Cancelled
from ..coordinator import Coordinator
from ..domains.code import CodeTask
from .events import EventLog


class LiveRun:
    def __init__(self, run_id: str, task: CodeTask, build: Callable[[], Coordinator],
                 out_dir: Path, review: bool = False, meta: dict | None = None):
        self.id = run_id
        self.task = task
        self.build = build
        self.out_dir = Path(out_dir)
        self.review = review
        self.meta = meta or {}
        self.status = "running"
        self.started_at = time.time()
        self.log = EventLog()
        self._cancel = threading.Event()
        self.thread = threading.Thread(target=self._work, name=f"live-{run_id}", daemon=True)

    # ------------------------------------------------------------------ control

    def start(self) -> "LiveRun":
        self.thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def finished(self) -> bool:
        return self.status != "running"

    # ------------------------------------------------------------------ events

    def _emit(self, event: dict) -> None:
        self.log.emit(event)

    def events_after(self, after: int) -> list[dict]:
        return self.log.events_after(after)

    def wait_events(self, after: int, timeout: float) -> list[dict]:
        """Events with seq > `after`, waiting up to `timeout` seconds for the next one."""
        return self.log.wait_events(after, timeout)

    def summary(self) -> dict:
        return {"kind": "live", "id": self.id, "task_id": self.task.task_id, "status": self.status,
                "started_at": self.started_at, "last_seq": self.log.last_seq, **self.meta}

    # ------------------------------------------------------------------ work

    def _work(self) -> None:
        self._emit({"type": "started", "task_id": self.task.task_id,
                    "entry_point": self.task.entry_point, "spec": self.task.spec,
                    "review": self.review, **self.meta})
        try:
            coordinator = self.build()
            outcome, transcript = coordinator.solve(self.task, review=self.review,
                                                    on_event=self._emit, cancel=self._cancel)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"{self.task.task_id.replace('/', '_')}.json"
            path.write_text(json.dumps({"outcome": outcome.to_dict(),
                                        "transcript": asdict(transcript)}, indent=1))
            self._emit({"type": "outcome", "outcome": outcome.to_dict(), "saved_to": str(path)})
            status = "done"
        except Cancelled:
            status = "cancelled"
        except Exception as exc:  # shown to the user; the server keeps running
            self._emit({"type": "error", "message": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc(limit=5)})
            status = "failed"
        self.status = status
        self._emit({"type": "done", "status": status})
        self.log.close()
