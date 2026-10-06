"""End-to-end pipeline on a toy task with a scripted model.

Setup: two of three candidates share the same bug (x + 2), and the Testing agent
writes one wrong expectation. Majority vote follows the buggy majority; evidence
based arbitration recovers the correct candidate.
"""

import json
import re

from hivemind.agents import Team
from hivemind.coordinator import STRATEGIES, Coordinator
from hivemind.domains.code import CodeTask
from hivemind.llm import Ledger, ScriptedLLM
from hivemind.report import mcnemar_exact, wilson, write_report

TASK = CodeTask(
    task_id="Toy/0",
    prompt='def add_one(x: int) -> int:\n    """Return x plus one.\n    >>> add_one(1)\n    2\n    """\n',
    entry_point="add_one",
    hidden_test="def check(candidate):\n    assert candidate(5) == 6\n    assert candidate(-1) == 0\n",
    canonical_solution="    return x + 1\n",
)
BUGGY = "```python\ndef add_one(x):\n    return x + 2\n```"
CORRECT = "```python\ndef add_one(x):\n    return x + 1\n```"


def respond(system: str, prompt: str) -> str:
    if system.startswith("You are the Research"):
        return "Contract: int -> int. Edge cases: negatives, zero."
    if system.startswith("You are the Testing"):
        # The last expectation is wrong: add_one(0) should be 1.
        return "```python\nassert add_one(5) == 6\nassert add_one(-1) == 0\nassert add_one(0) == 2\n```"
    if system.startswith("You are the Coordinator"):
        if "WINNER" in prompt:
            return "Solution 0 looks fine.\nWINNER: 0"
        verdicts = []
        for n, call, options in re.findall(r"CASE (\d+): (.+)\n((?:  [A-Z]\) .+\n?)+)", prompt):
            truth = repr(eval(call, {"add_one": lambda x: x + 1}))
            letter = next(m[0] for m in re.findall(r"([A-Z])\) (.+)", options) if m[1] == truth)
            verdicts.append(f"CASE {n}: {letter}")
        return "\n".join(verdicts)
    if system.startswith("You are the Coding"):
        if "fails cases whose correct results" in prompt:
            return CORRECT
        if "Your team is implementing" in prompt:  # debate: everyone keeps their own answer
            own = prompt.split("Your current solution")[1].split("```python\n")[1].split("```")[0]
            return f"I keep mine.\n```python\n{own}```"
        return CORRECT if "docstring examples by hand" in prompt else BUGGY
    return "No concerns."


def make_coordinator(ledger=None):
    llm = ScriptedLLM(respond)
    return Coordinator(Team(llm, llm, ledger or Ledger()), n_candidates=3)


def test_arbitration_beats_majority_vote_on_shared_bug():
    outcome, transcript = make_coordinator().solve(TASK)
    passed = {k: v["passed"] for k, v in outcome.answers.items()}
    assert passed == {
        "solo": False, "first": False, "vote": False, "debate": False,
        "tests_only": True, "arbitration_select": True, "arbitration": True,
    }
    assert outcome.candidate_passed == [False, True, False]
    assert outcome.answers["arbitration"]["source"] == "candidate 1"


def test_rulings_overrule_the_wrong_test_and_are_checked_against_reference():
    outcome, _ = make_coordinator().solve(TASK)
    d = outcome.diagnostics
    assert d["n_spec_cases"] == 1 and d["n_tester_cases"] == 3
    assert (d["tester_correct"], d["tester_judged"]) == (2, 3)
    assert d["n_disputes"] == 3
    assert all(r["reference_agrees"] for r in d["rulings"])
    wrong_test = next(r for r in d["rulings"] if r["call"] == "add_one(0)")
    assert wrong_test["choice"] == "1"


def test_every_call_is_recorded_with_its_phase():
    ledger = Ledger()
    _, transcript = make_coordinator(ledger).solve(TASK, review=True)
    phases = [r.phase for r in ledger.records]
    assert phases[:6] == ["solo", "brief", "candidate:0", "candidate:1", "candidate:2", "tests"]
    assert "arbiter" in phases and "judge" in phases
    assert {"review:security", "review:documentation", "review:ethics"} <= set(phases)
    assert len(transcript.turns) == len(ledger.records)


def test_report_is_written(tmp_path):
    ledger = Ledger()
    outcome, _ = make_coordinator(ledger).solve(TASK)
    (tmp_path / "outcomes.jsonl").write_text(json.dumps(outcome.to_dict()) + "\n")
    (tmp_path / "calls.jsonl").write_text(
        "".join(json.dumps(r.__dict__) + "\n" for r in ledger.records))
    (tmp_path / "config.toml").write_text(
        '[run]\nname = "toy"\n[agents]\nprovider = "scripted"\nmodel = "scripted"\n'
        '[coordinator]\nprovider = "scripted"\nmodel = "scripted"\n')
    report = write_report(tmp_path).read_text()
    assert "## Models used" in report and "`scripted`" in report
    for name in STRATEGIES:
        assert f"`{name}`" in report
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["strategies"]["arbitration"]["passed"] == 1
    assert summary["strategies"]["vote"]["passed"] == 0


def test_broken_candidates_cannot_outvote_or_be_offered_as_answers():
    crashing = "```python\ndef add_one(x)\n    return x + 1\n```"  # syntax error: never loads

    def respond_broken(system, prompt):
        if system.startswith("You are the Coding") and "Approach:" in prompt \
                and "docstring examples by hand" not in prompt:
            return crashing  # candidates 0 and 2 fail to load, identically
        return respond(system, prompt)

    llm = ScriptedLLM(respond_broken)
    outcome, transcript = Coordinator(Team(llm, llm, Ledger()), n_candidates=3).solve(TASK)
    assert outcome.answers["vote"]["source"] == "candidate 1"
    assert outcome.answers["vote"]["passed"] is True
    arbiter_prompts = [t["prompt"] for t in transcript.turns if t["phase"] == "arbiter"]
    assert all("load:" not in p for p in arbiter_prompts)


def test_statistics_helpers():
    assert mcnemar_exact(0, 0) == 1.0
    assert round(mcnemar_exact(10, 0), 4) == 0.002
    lo, hi = wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
