"""A live run: one task solved in the background, its events kept for replay."""

import json
import threading

from hivemind.agents import Team
from hivemind.coordinator import Coordinator
from hivemind.llm import Ledger, ScriptedLLM
from hivemind.web.live import LiveRun

from test_coordinator import TASK, respond


def factory(respond_fn=respond):
    def build():
        llm = ScriptedLLM(respond_fn)
        return Coordinator(Team(llm, llm, Ledger()), n_candidates=3)
    return build


def finished(run, timeout=20):
    run.thread.join(timeout)
    assert not run.thread.is_alive()
    return run


def test_a_finished_run_ends_with_outcome_then_done(tmp_path):
    run = finished(LiveRun("r1", TASK, factory(), out_dir=tmp_path).start())
    events = run.events_after(0)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["type"] == "started" and events[0]["task_id"] == "Toy/0"
    assert [e["type"] for e in events[-2:]] == ["outcome", "done"]
    assert events[-1]["status"] == "done" and run.status == "done"
    assert events[-2]["outcome"]["answers"]["arbitration"]["passed"] is True


def test_the_finished_run_is_saved_like_the_cli_demo(tmp_path):
    finished(LiveRun("r1", TASK, factory(), out_dir=tmp_path).start())
    saved = json.loads((tmp_path / "Toy_0.json").read_text())
    assert set(saved) == {"outcome", "transcript"}
    assert saved["transcript"]["task_id"] == "Toy/0" and saved["transcript"]["turns"]


def test_a_reconnecting_reader_gets_only_newer_events(tmp_path):
    run = finished(LiveRun("r1", TASK, factory(), out_dir=tmp_path).start())
    later = run.events_after(5)
    assert later[0]["seq"] == 6 and later == run.events_after(0)[5:]


def test_waiting_reader_is_woken_by_new_events(tmp_path):
    gate = threading.Event()

    def slow(system, prompt):
        gate.wait(10)
        return respond(system, prompt)

    run = LiveRun("r1", TASK, factory(slow), out_dir=tmp_path).start()
    first = run.wait_events(after=0, timeout=5)
    assert first and first[0]["type"] == "started"
    gate.set()
    finished(run)
    assert run.wait_events(after=10**6, timeout=0.1) == []


def test_cancel_stops_the_run_and_reports_it(tmp_path):
    gate = threading.Event()

    def slow(system, prompt):
        gate.wait(10)
        return respond(system, prompt)

    run = LiveRun("r1", TASK, factory(slow), out_dir=tmp_path).start()
    run.cancel()
    gate.set()
    finished(run)
    assert run.status == "cancelled"
    assert run.events_after(0)[-1] == {**run.events_after(0)[-1], "type": "done", "status": "cancelled"}
    assert not (tmp_path / "Toy_0.json").exists()


def test_a_crash_is_reported_as_failed(tmp_path):
    def broken():
        raise RuntimeError("model config is broken")

    run = finished(LiveRun("r1", TASK, broken, out_dir=tmp_path).start())
    kinds = [e["type"] for e in run.events_after(0)]
    assert kinds[-2:] == ["error", "done"] and run.status == "failed"
    assert "model config is broken" in run.events_after(0)[-2]["message"]
