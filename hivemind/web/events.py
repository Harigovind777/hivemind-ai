"""An append-only list of events with increasing `seq`, kept for replay.

Readers poll `wait_events(after=seq)`, so a page that reconnects (or opens late)
gets the whole history first and then each new event as it happens.
"""

from __future__ import annotations

import threading
import time


class EventLog:
    def __init__(self, events: list[dict] | None = None):
        self._events: list[dict] = list(events or [])
        self._cond = threading.Condition()
        self._closed = False

    def emit(self, event: dict) -> dict:
        with self._cond:
            stamped = {"seq": len(self._events) + 1, "time": time.time(), **event}
            self._events.append(stamped)
            self._cond.notify_all()
            return stamped

    def close(self) -> None:
        """No more events will come, so waiting readers return at once."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def last_seq(self) -> int:
        with self._cond:
            return self._events[-1]["seq"] if self._events else 0

    def events_after(self, after: int) -> list[dict]:
        with self._cond:
            return list(self._events[max(0, after):])

    def wait_events(self, after: int, timeout: float) -> list[dict]:
        """Events with seq > `after`, waiting up to `timeout` seconds for the next one."""
        with self._cond:
            if len(self._events) <= after and not self._closed:
                self._cond.wait(timeout)
            return list(self._events[max(0, after):])
