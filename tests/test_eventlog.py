"""EventLog: numbered events kept for replay, with waiting readers."""

import threading
import time

from hivemind.web.events import EventLog


def test_events_are_numbered_from_one_and_replayable():
    log = EventLog()
    log.emit({"type": "a"})
    log.emit({"type": "b"})
    events = log.events_after(0)
    assert [(e["seq"], e["type"]) for e in events] == [(1, "a"), (2, "b")]
    assert log.events_after(1) == events[1:] and log.last_seq == 2


def test_a_loaded_history_continues_its_numbering():
    log = EventLog([{"seq": 1, "time": 0.0, "type": "a"}])
    assert log.emit({"type": "b"})["seq"] == 2


def test_a_waiting_reader_wakes_on_the_next_event():
    log = EventLog()
    threading.Timer(0.05, log.emit, args=({"type": "late"},)).start()
    t0 = time.monotonic()
    got = log.wait_events(after=0, timeout=5)
    assert [e["type"] for e in got] == ["late"] and time.monotonic() - t0 < 2


def test_a_closed_log_does_not_make_readers_wait():
    log = EventLog()
    log.close()
    t0 = time.monotonic()
    assert log.wait_events(after=0, timeout=5) == [] and time.monotonic() - t0 < 1
