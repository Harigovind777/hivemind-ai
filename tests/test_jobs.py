"""Experiments run as a child process; progress comes from the run's outcomes.jsonl."""

import json
import sys
import textwrap
import time

import pytest

from hivemind.web.jobs import ExperimentJob

FAKE_RUNNER = textwrap.dedent('''
    import json, sys, time
    out, n, delay, code = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    for i in range(n):
        time.sleep(delay)
        with open(out + "/outcomes.jsonl", "a") as f:
            passed = i % 2 == 0
            f.write(json.dumps({"task_id": f"HumanEval/{i}", "answers": {
                "vote": {"passed": True}, "arbitration": {"passed": passed}}}) + "\\n")
        print(f"[fake] {i + 1}/{n} HumanEval/{i}", flush=True)
    if code:
        print("Traceback: something broke", flush=True)
    sys.exit(code)
''')


@pytest.fixture
def runner(tmp_path):
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER)
    out = tmp_path / "run"
    out.mkdir()

    def make(n=3, delay=0.0, code=0, total=None):
        command = [sys.executable, str(script), str(out), str(n), str(delay), str(code)]
        return ExperimentJob("j1", command, out_dir=out, total=total or n)
    return make


def wait_finished(job, timeout=20):
    deadline = time.time() + timeout
    while job.progress()["status"] == "running" and time.time() < deadline:
        time.sleep(0.05)
    return job.progress()


def test_progress_follows_the_outcomes_file(runner):
    p = wait_finished(runner(n=3).start())
    assert p["status"] == "done" and p["returncode"] == 0
    assert (p["done"], p["total"]) == (3, 3)
    assert [t["task_id"] for t in p["tasks"]] == ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
    assert [t["passed"]["arbitration"] for t in p["tasks"]] == [True, False, True]
    assert p["log"][-1] == "[fake] 3/3 HumanEval/2"


def test_a_failing_runner_is_marked_failed_with_its_log(runner):
    p = wait_finished(runner(n=1, code=1).start())
    assert p["status"] == "failed" and p["returncode"] == 1
    assert "Traceback: something broke" in p["log"]


def test_cancel_stops_the_child_process(runner):
    job = runner(n=50, delay=0.2).start()
    time.sleep(0.3)
    job.cancel()
    p = wait_finished(job)
    assert p["status"] == "cancelled"
    assert p["done"] < 50
    assert job.process.poll() is not None


def test_tasks_already_in_the_folder_count_towards_progress(runner, tmp_path):
    (tmp_path / "run" / "outcomes.jsonl").write_text(
        json.dumps({"task_id": "HumanEval/99", "answers": {"arbitration": {"passed": True}}}) + "\n")
    p = wait_finished(runner(n=2, total=3).start())
    assert (p["done"], p["total"]) == (3, 3)


def test_a_half_written_line_is_ignored(runner, tmp_path):
    job = runner(n=1)
    (tmp_path / "run" / "outcomes.jsonl").write_text('{"task_id": "HumanEval/7", "answ')
    assert job.progress()["done"] == 0
