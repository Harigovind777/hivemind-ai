"""The web interface: prompt handling, evidence labelling, and the API surface.

The recurring theme is that a prompt typed into a browser has no hidden test suite. These
tests exist to make sure the UI never implies otherwise.
"""

import json

import pytest

from hivemind.coordinator import STRATEGIES
from hivemind.domains.freeform import (detect_entry_point, evidence_grade, looks_like_a_prompt,
                                       make_task, syntax_check)

# Both examples need at least three distinct values, otherwise "second smallest" is
# ambiguous and the two examples demand different implementations.
PROMPT = '''def second_smallest(numbers):
    """Return the second smallest distinct value.

    >>> second_smallest([4, 1, 3, 3, 2])
    2
    >>> second_smallest([9, 9, 8, 5])
    8
    """
'''


# --------------------------------------------------------------- prompt handling

def test_entry_point_is_detected_from_the_prompt():
    assert detect_entry_point(PROMPT) == "second_smallest"
    assert detect_entry_point("def a():\n    pass\n") == "a"
    assert detect_entry_point("no function here") is None


def test_make_task_wraps_a_prompt_without_inventing_ground_truth():
    task = make_task(PROMPT)
    assert task.entry_point == "second_smallest"
    # These stay empty on purpose; anything else would let a fabricated grade leak in.
    assert task.hidden_test == ""
    assert task.canonical_solution == ""


def test_explicit_entry_point_must_exist_in_the_prompt():
    with pytest.raises(ValueError, match="never defines"):
        make_task(PROMPT, entry_point="not_defined_here")


def test_bad_prompts_are_rejected_with_a_reason():
    assert "too short" in looks_like_a_prompt("hi")
    assert "No function definition" in looks_like_a_prompt("please write something nice for me")
    assert "Enter a prompt" in looks_like_a_prompt("   ")
    assert looks_like_a_prompt(PROMPT) is None


def test_prose_with_a_named_entry_point_is_accepted():
    # A prompt in ordinary language plus the name of the function to build is a complete
    # specification, so it must not be turned away for lacking a `def` line.
    prose = ("I need a function called count_vowels that takes a string and returns how "
             "many vowels appear in it, ignoring case. Empty strings return 0.")
    assert "No function definition" in looks_like_a_prompt(prose)
    assert looks_like_a_prompt(prose, "count_vowels") is None

    task = make_task(prose, entry_point="count_vowels")
    assert task.entry_point == "count_vowels"
    # Prose carries no ground truth either, so nothing may be invented to grade against.
    assert task.hidden_test == ""
    assert task.canonical_solution == ""


def test_prose_without_an_entry_point_is_still_rejected():
    prose = ("Write a function that takes a list of numbers and returns the total sum of "
             "every element, treating an empty list as zero.")
    with pytest.raises(ValueError, match="Could not find a function"):
        make_task(prose)


def test_syntax_check_reports_the_line():
    assert syntax_check("def f(:\n    pass") is not None
    assert syntax_check("") == "empty solution"
    assert syntax_check("def f():\n    return 1\n") is None


# --------------------------------------------------------------- evidence grading

def test_evidence_grade_reports_the_team_evidence_not_accuracy():
    from hivemind.domains.freeform import make_task as mk
    task = mk(PROMPT)
    cases = [{"call": "second_smallest([4, 1, 3, 3, 2])", "expected": "2"},
             {"call": "second_smallest([9, 9, 8, 5])", "expected": "8"}]
    verified = {0: ("sig", "2"), 1: ("sig", "8")}

    # Second smallest *distinct* value: [9, 9, 8, 5] -> 8, not the largest.
    good = "def second_smallest(numbers):\n    return sorted(set(numbers))[1]\n"
    bad = "def second_smallest(numbers):\n    return sorted(numbers)[0]\n"

    ok = evidence_grade(good, task, cases, verified)
    assert (ok["verified_satisfied"], ok["verified_total"]) == (2, 2)
    assert ok["error"] is None

    # Cases the Coordinator never ruled on are excluded from the denominator rather than
    # counted as failures, so a formatting problem cannot look like a wrong answer.
    partial = evidence_grade(bad, task, cases, {0: ("sig", "2")})
    assert (partial["verified_satisfied"], partial["verified_total"]) == (0, 1)
    assert partial["unverified"] == 1


def test_evidence_grade_on_empty_code_is_an_error_not_a_zero():
    from hivemind.domains.freeform import make_task as mk
    out = evidence_grade("", mk(PROMPT), [{"call": "f()"}], {0: ("pass",)})
    assert out["error"] == "empty solution"
    assert out["verified_satisfied"] == 0


def test_task_outcome_marks_itself_evidence_only():
    """A graded benchmark task and a browser prompt must be distinguishable downstream."""
    from hivemind.agents import Team
    from hivemind.coordinator import Coordinator
    from hivemind.llm import Ledger, ScriptedLLM

    code = "```python\ndef second_smallest(numbers):\n    return sorted(set(numbers))[1]\n```"

    def respond(system, prompt):
        if system.startswith("You are the Testing"):
            return "```python\nassert second_smallest([4, 1, 3]) == 2\n```"
        if system.startswith("You are the Coordinator"):
            return "WINNER: 0" if "WINNER" in prompt else ""
        return code

    llm = ScriptedLLM(respond)
    outcome, _ = Coordinator(Team(llm, llm, Ledger()), n_candidates=2).solve(make_task(PROMPT))
    assert outcome.evidence_only is True
    assert outcome.diagnostics["has_reference"] is False
    # Every answer carries evidence detail rather than a bare pass/fail.
    for answer in outcome.answers.values():
        assert "evidence" in answer
        assert "verified_total" in answer["evidence"]


def test_benchmark_task_is_not_marked_evidence_only():
    from hivemind.coordinator import TaskOutcome
    out = TaskOutcome(task_id="HumanEval/0", candidates=[], candidate_passed=[])
    assert out.evidence_only is False


# --------------------------------------------------------------- API

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from hivemind.webapp import create_app
    return TestClient(create_app(default_config={
        "agents": {"provider": "scripted", "model": "s"},
        "coordinator": {"provider": "scripted", "model": "s"},
        "candidates": 2, "repair": True,
    }, jobs={}))


def test_index_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "HiveMind" in r.text
    # The page must not present browser results as benchmark scores.
    assert "not benchmark scores" in r.text


def test_web_interface_does_not_need_toml():
    """The web path reads no config file, so it must not import tomllib at all.

    It reaches the price table in report.py for costing. That used to drag tomllib into
    every browser request path, which broke the interface outright on a runtime without
    it. Regression guard: import the app with tomllib poisoned.
    """
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent(f"""
        import sys
        sys.modules["tomllib"] = None          # any import of it now raises
        sys.path.insert(0, {str(__import__("pathlib").Path(__file__).resolve().parents[1])!r})
        from hivemind.webapp import create_app
        from hivemind.report import cost_usd
        assert cost_usd("anthropic", "claude-opus-5-5", 1_000_000, 0) == 4.0
        assert cost_usd("ollama", "qwen", 1_000_000, 1_000_000) == 0.0
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "ok" in out.stdout, out.stderr[-800:]


def test_config_endpoint_states_the_limitation(client):
    data = client.get("/api/config").json()
    assert "evidence" in data["note"]
    assert "ollama" in data["providers"]


def test_short_prompt_is_rejected_with_a_reason(client):
    r = client.post("/api/solve", data={"prompt": "hi"})
    assert r.status_code == 400
    assert "too short" in r.json()["detail"]


def test_prompt_without_a_function_is_rejected(client):
    r = client.post("/api/solve", data={"prompt": "please write a poem about the sea"})
    assert r.status_code == 400
    assert "function definition" in r.json()["detail"]


def test_unknown_job_is_404(client):
    assert client.get("/api/result/deadbeef").status_code == 404
    assert client.get("/api/events/deadbeef").status_code == 404


def test_solve_returns_a_job_and_flags_evidence_only(client):
    r = client.post("/api/solve", data={"prompt": PROMPT})
    assert r.status_code == 200
    body = r.json()
    assert body["evidence_only"] is True
    assert body["entry_point"] == "second_smallest"


def test_an_omitted_candidate_count_falls_back_to_the_config(client):
    assert client.post("/api/solve", data={"prompt": PROMPT}).json()["candidates"] == 2
    assert client.post("/api/solve",
                       data={"prompt": PROMPT, "candidates": "5"}).json()["candidates"] == 5


def test_candidate_count_is_clamped(client):
    assert client.post("/api/solve", data={"prompt": PROMPT, "candidates": "99"}
                       ).json()["candidates"] == 7
    assert client.post("/api/solve", data={"prompt": PROMPT, "candidates": "0"}
                       ).json()["candidates"] == 1
    assert client.post("/api/solve", data={"prompt": PROMPT, "candidates": "-4"}
                       ).json()["candidates"] == 1


def test_a_dead_provider_surfaces_as_an_error_not_a_zero_result(client, monkeypatch):
    """The failure mode from the validity work, through the HTTP surface."""
    import hivemind.webapp as web

    class Dead:
        provider = "scripted"
        model = "s"

        def complete(self, system, prompt, *, seed=0):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(web, "make_llm", lambda *a, **k: Dead())
    r = client.post("/api/solve", data={"prompt": PROMPT})
    job_id = r.json()["job_id"]
    result = client.get(f"/api/result/{job_id}").json()
    # Either still running or a clean error; never a fabricated pass/fail result.
    assert result["status"] in ("running", "error")
    if result["status"] == "error":
        assert "no usable output" in result["error"]


def test_event_stream_emits_sse_frames(client, monkeypatch):
    import hivemind.webapp as web

    code = "```python\ndef second_smallest(numbers):\n    return sorted(set(numbers))[1]\n```"

    def respond(system, prompt):
        if system.startswith("You are the Testing"):
            return "```python\nassert second_smallest([4, 1, 3]) == 2\n```"
        if system.startswith("You are the Coordinator"):
            return "WINNER: 0" if "WINNER" in prompt else ""
        return code

    from hivemind.llm import ScriptedLLM
    monkeypatch.setattr(web, "make_llm", lambda *a, **k: ScriptedLLM(respond))

    job_id = client.post("/api/solve", data={"prompt": PROMPT}).json()["job_id"]
    with client.stream("GET", f"/api/events/{job_id}") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert "data: " in body
    assert '"kind": "close"' in body

    result = client.get(f"/api/result/{job_id}").json()
    assert result["status"] == "ok"
    assert result["evidence_only"] is True
    assert set(result["answers"]) == set(STRATEGIES)
    assert result["diagnostics"]["has_reference"] is False
