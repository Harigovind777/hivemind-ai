"""Live events, cancellation and per-role models, on the scripted toy task."""

import threading

import pytest

from hivemind.agents import Cancelled, Team
from hivemind.coordinator import Coordinator
from hivemind.llm import Ledger, ScriptedLLM
from hivemind.runner import build_coordinator

from test_coordinator import TASK, respond


def scripted_coordinator(ledger=None, role_llms=None):
    llm = ScriptedLLM(respond, model="team-model")
    return Coordinator(Team(llm, ScriptedLLM(respond, model="arbiter-model"), ledger or Ledger(),
                            role_llms=role_llms), n_candidates=3)


def test_role_override_routes_only_that_roles_calls_to_its_model():
    ledger = Ledger()
    tester = ScriptedLLM(respond, model="tester-model")
    scripted_coordinator(ledger, role_llms={"testing": tester}).solve(TASK)
    models = {(r.role, r.model) for r in ledger.records}
    assert ("testing", "tester-model") in models
    assert ("coding", "team-model") in models
    assert ("coordinator", "arbiter-model") in models
    assert ("testing", "team-model") not in models


def test_events_follow_the_pipeline_order():
    events = []
    scripted_coordinator().solve(TASK, on_event=events.append)
    kinds = [(e["type"], e.get("phase")) for e in events]
    assert kinds[:4] == [("turn_started", "solo"), ("turn", "solo"),
                         ("turn_started", "brief"), ("turn", "brief")]
    tests_done = kinds.index(("turn", "tests"))
    evidence = kinds.index(("evidence", None))
    arbiter_started = kinds.index(("turn_started", "arbiter"))
    rulings = kinds.index(("rulings", None))
    assert tests_done < evidence < arbiter_started < kinds.index(("turn", "arbiter")) < rulings


def test_turn_events_carry_the_reply_and_its_cost():
    events = []
    scripted_coordinator().solve(TASK, on_event=events.append)
    brief = next(e for e in events if e["type"] == "turn" and e["phase"] == "brief")
    assert brief["role"] == "research"
    assert brief["model"] == "team-model"
    assert brief["response"] == "Contract: int -> int. Edge cases: negatives, zero."
    assert brief["output_tokens"] > 0 and brief["error"] is None


def test_evidence_and_rulings_describe_the_dispute():
    events = []
    scripted_coordinator().solve(TASK, on_event=events.append)
    evidence = next(e for e in events if e["type"] == "evidence")
    # Candidates 0 and 2 share the x + 2 bug; candidate 1 is correct.
    assert evidence["candidates"] == 3 and evidence["distinct_behaviours"] == 2
    assert evidence["spec_cases"] == 1 and evidence["tester_cases"] == 3
    rulings = next(e for e in events if e["type"] == "rulings")
    assert rulings["disputes"] == 3
    wrong_test = next(r for r in rulings["rulings"] if r["call"] == "add_one(0)")
    assert wrong_test["choice"] == "1"


def test_cancel_stops_before_the_next_model_call():
    ledger = Ledger()
    cancel = threading.Event()

    def stop_after_brief(event):
        if event["type"] == "turn" and event["phase"] == "brief":
            cancel.set()

    with pytest.raises(Cancelled):
        scripted_coordinator(ledger).solve(TASK, on_event=stop_after_brief, cancel=cancel)
    assert [r.phase for r in ledger.records] == ["solo", "brief"]


def test_config_role_sections_override_the_team_model():
    config = {
        "run": {"name": "t"},
        "agents": {"provider": "ollama", "model": "team:1b"},
        "coordinator": {"provider": "ollama", "model": "judge:7b"},
        "roles": {"testing": {"provider": "ollama", "model": "tester:3b", "temperature": 0.1}},
    }
    team = build_coordinator(config, Ledger()).team
    assert team["testing"].llm.model == "tester:3b"
    assert team["testing"].llm.temperature == 0.1
    assert team["coding"].llm.model == "team:1b"
    assert team["coordinator"].llm.model == "judge:7b"
