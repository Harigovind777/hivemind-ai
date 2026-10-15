"""Coordinator.answer: the shared pipeline steps alone, for problems with no hidden tests."""

from hivemind.agents import Team
from hivemind.coordinator import ChatAnswer, Coordinator
from hivemind.domains.code import CodeTask
from hivemind.llm import Ledger, ScriptedLLM

from test_coordinator import BUGGY, TASK, make_coordinator, respond

# The chat's version of the toy task: the same spec, but no hidden tests or reference.
CHAT_TASK = CodeTask("chat/test/1", TASK.prompt, TASK.entry_point, "", "")
SOLVE_PHASES = ["solo", "brief", "candidate:0", "candidate:1", "candidate:2", "tests", "arbiter",
                "debate:1:0", "debate:1:1", "debate:1:2", "judge"]


def coordinator(respond_fn=respond, ledger=None):
    llm = ScriptedLLM(respond_fn)
    return Coordinator(Team(llm, llm, ledger or Ledger()), n_candidates=3)


def test_solve_keeps_its_steps_and_results():
    ledger = Ledger()
    outcome, _ = make_coordinator(ledger).solve(TASK)
    assert [r.phase for r in ledger.records] == SOLVE_PHASES
    assert {k: v["source"] for k, v in outcome.answers.items()} == {
        "solo": "solo coder", "first": "candidate 0", "vote": "candidate 0",
        "debate": "debate winner 0", "tests_only": "candidate 1",
        "arbitration_select": "candidate 1", "arbitration": "candidate 1"}
    assert outcome.diagnostics["verified_scores"] == [0, 4, 0]


def test_answer_skips_the_baselines_and_picks_what_the_evidence_supports():
    ledger = Ledger()
    answer, transcript = coordinator(ledger=ledger).answer(CHAT_TASK)
    assert [r.phase for r in ledger.records] == [
        "brief", "candidate:0", "candidate:1", "candidate:2", "tests", "arbiter"]
    assert isinstance(answer, ChatAnswer)
    assert answer.source == "candidate 1" and "x + 1" in answer.code
    assert (answer.verified_passed, answer.verified_total) == (4, 4)
    assert (answer.spec_cases, answer.ruled_cases, answer.undisputed_cases) == (1, 3, 0)
    assert answer.working and not answer.repaired and answer.failures == []
    assert len(answer.rulings) == 3 and all(r["reference_agrees"] is None for r in answer.rulings)
    assert len(transcript.turns) == 6


def test_a_kept_repair_is_the_answer_and_remaining_failures_are_listed():
    def all_buggy(system, prompt):
        if system.startswith("You are the Coding") and "fails cases whose correct results" not in prompt:
            return BUGGY
        return respond(system, prompt)

    answer, _ = coordinator(all_buggy).answer(CHAT_TASK)
    assert answer.repaired and answer.source == "repair of candidate 0"
    assert (answer.verified_passed, answer.verified_total) == (3, 4)
    # Every buggy candidate agreed with the Testing agent's wrong add_one(0) == 2,
    # so nobody disputed it, and the correct repair now "fails" it. That is reported.
    assert answer.undisputed_cases == 1
    assert answer.failures == [{"call": "add_one(0)", "expected": "2", "got": "1"}]


def test_no_working_candidate_is_reported_not_hidden():
    crashing = "```python\ndef add_one(x)\n    return x + 1\n```"  # never loads

    def broken(system, prompt):
        return crashing if system.startswith("You are the Coding") else respond(system, prompt)

    answer, _ = coordinator(broken).answer(CHAT_TASK)
    assert not answer.working and not answer.repaired
    assert answer.verified_passed == 0 and answer.verified_total >= 1
    assert all(f["got"].startswith("<load:") for f in answer.failures)
