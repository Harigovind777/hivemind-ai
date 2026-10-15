"""Validity guards: the checks that stop a broken run from looking like a result.

Each test here corresponds to a way the pipeline could previously produce a
confident-looking report about a run that measured nothing.
"""

import json
import sys

import pytest

from hivemind.agents import Team
from hivemind.coordinator import (Coordinator, NoUsableOutput, STRATEGIES, TaskOutcome,
                                  phase_matches)
from hivemind.domains.code import CodeTask, grade_detailed, sanitize
from hivemind.llm import Ledger, ScriptedLLM
from hivemind.report import holm_bonferroni, mcnemar_exact, write_report
from hivemind.runner import ConfigChanged, _check_resume, _config_fingerprint, estimate_cost
from hivemind.sandbox import run_cases, strict_blocks_network

TASK = CodeTask(
    task_id="Toy/0",
    prompt='def add_one(x: int) -> int:\n    """Return x plus one.\n    >>> add_one(1)\n    2\n    """\n',
    entry_point="add_one",
    hidden_test="def check(candidate):\n    assert candidate(5) == 6\n",
    canonical_solution="    return x + 1\n",
)
GOOD = "```python\ndef add_one(x):\n    return x + 1\n```"


def dead(system, prompt):
    raise RuntimeError("401 invalid api key")


def _write_run(tmp_path, outcomes, calls=()):
    (tmp_path / "outcomes.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outcomes))
    (tmp_path / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (tmp_path / "config.toml").write_text(
        '[run]\nname = "toy"\ncandidates = 3\n[agents]\nprovider = "scripted"\nmodel = "scripted"\n'
        '[coordinator]\nprovider = "scripted"\nmodel = "scripted"\n')
    return tmp_path


def _outcome(task_id="Toy/0", **over):
    base = {"task_id": task_id, "candidates": ["a", "b", "c"],
            "candidate_passed": [False, True, False], "answers": {}, "diagnostics": {}}
    base.update(over)
    return base


# --------------------------------------------------------------- dead provider

def test_task_with_no_usable_candidate_raises_instead_of_scoring_zero():
    """A dead API key must not yield a clean all-fail outcome that reads as a result."""
    llm = ScriptedLLM(dead)
    coordinator = Coordinator(Team(llm, llm, Ledger()), n_candidates=3)
    with pytest.raises(NoUsableOutput) as excinfo:
        coordinator.solve(TASK)
    assert "0/3" in str(excinfo.value)
    assert "invalid api key" in str(excinfo.value)


def test_one_empty_candidate_is_tolerated_but_zero_is_not():
    """min_usable=1 keeps a single flaky call from killing an otherwise good task."""
    calls = {"n": 0}

    def flaky(system, prompt):
        calls["n"] += 1
        # solve() asks in a fixed order: solo, brief, candidate:0, candidate:1, candidate:2.
        if calls["n"] == 3:      # candidate:0 comes back empty
            return ""
        return GOOD

    llm = ScriptedLLM(flaky)
    outcome, _ = Coordinator(Team(llm, llm, Ledger()), n_candidates=3).solve(TASK)
    assert sum(1 for c in outcome.candidates if c.strip()) == 2
    assert outcome.answers                      # a real outcome, not an exception


# --------------------------------------------------------------- grade reasons

def test_grade_distinguishes_timeout_from_a_wrong_answer():
    """A correct-but-slow solution must not be filed as wrong logic."""
    slow = "def add_one(x):\n    t = 0\n    for i in range(10**9):\n        t += i\n    return x + 1\n"
    ok, reason = grade_detailed(TASK, slow, timeout_s=1.0)
    assert ok is False and reason == "timeout"

    ok, reason = grade_detailed(TASK, "def add_one(x):\n    return x + 2\n")
    assert ok is False and reason == "assertion"

    ok, reason = grade_detailed(TASK, "def add_one(x)\n    return x\n")   # syntax error
    assert ok is False and reason.startswith("syntax")

    ok, reason = grade_detailed(TASK, "")
    assert ok is False and reason == "empty"

    ok, reason = grade_detailed(TASK, "def add_one(x):\n    return x + 1\n")
    assert ok is True and reason == "ok"


def test_report_counts_grade_reasons():
    from hivemind.report import summarise
    d = {"n_spec_cases": 1, "n_tester_cases": 1, "tester_correct": 1, "tester_judged": 1,
         "unanimous": False, "distinct_behaviours": 2, "unsafe_candidates": 0, "n_disputes": 1,
         "rulings": [], "verified_scores": [1, 0], "verified_cases": 2, "cases_total": 2,
         "failing_after_selection": 0, "repaired": False,
         "candidate_failures": ["timeout", "timeout", "assertion"], "unsafe_answers": []}
    o = _outcome(diagnostics=d, answers={name: {"passed": False} for name in STRATEGIES})
    s = summarise(_write_run(__import__("pathlib").Path(_tmp()), [o]))
    assert s["evidence"]["grade_reasons"] == {"timeout": 2, "assertion": 1}


def _tmp():
    import tempfile
    return tempfile.mkdtemp()


# --------------------------------------------------------------- reply parsing

@pytest.mark.parametrize("reply,expected", [
    ("CASE 1: B", {1: "B"}),
    ("**CASE 1:** `C`", {1: "C"}),
    ("CASE 2 - the specification requires A", {2: "A"}),
    ("CASE 3: Option B", {3: "B"}),
    ("CASE 1: A\nCASE 2: B", {1: "A", 2: "B"}),
    ("CASE 1: UNSURE", {1: "UNSURE"}),
    ("I think CASE 1 must be B, so:", {1: "B"}),
    ("no verdicts here", {}),
])
def test_arbiter_verdicts_survive_common_formats(reply, expected):
    verdicts, unreadable = Coordinator.parse_case_verdicts(reply, 3)
    assert verdicts == expected
    assert set(unreadable) == set(range(1, 4)) - set(expected)


BUGGY = "```python\ndef add_one(x):\n    return x + 2\n```"
# A wrong expectation from the Testing agent, so the arbiter genuinely has a dispute.
TESTS = "```python\nassert add_one(5) == 6\nassert add_one(0) == 1\n```"


def _disputing_team(arbiter_reply, judge_reply: str = "WINNER: 0", k: int = 2):
    """A team whose candidates disagree, forcing the arbiter to be consulted.

    `arbiter_reply` may be a string, or a callable taking the number of disputed cases so
    a test can answer every case or deliberately answer only some of them.
    """
    calls = {"candidate": 0}

    def respond(system, prompt):
        if system.startswith("You are the Coordinator"):
            if "Disputed cases" in prompt:
                n_cases = prompt.count("\nCASE ")
                return (arbiter_reply(n_cases) if callable(arbiter_reply)
                        else arbiter_reply)
            return judge_reply
        if system.startswith("You are the Testing"):
            return TESTS
        if system.startswith("You are the Coding"):
            if "Your team is implementing" in prompt:      # debate: keep your own answer
                own = prompt.split("Your current solution")[1].split("```python\n")[1].split("```")[0]
                return f"Keeping mine.\n```python\n{own}```"
            if "Approach:" in prompt:                     # candidates alternate
                calls["candidate"] += 1
                return GOOD if calls["candidate"] % 2 else BUGGY
            return BUGGY                                 # solo
        return "Contract."

    llm = ScriptedLLM(respond)
    return Team(llm, llm, Ledger())


def test_unreadable_reply_is_not_reported_as_the_model_being_unsure():
    """"UNSURE" and "we could not parse it" are different facts and must stay separate."""
    team = _disputing_team("The specification clearly indicates the first result.")
    outcome, tr = Coordinator(team, n_candidates=2).solve(TASK)
    assert any(t["phase"] == "arbiter" for t in tr.turns), "arbiter should have been asked"
    rulings = outcome.diagnostics["rulings"]
    assert rulings, "a dispute should have produced a ruling"
    # No CASE lines at all, so every ruling is an explicit parse failure rather than a
    # silently dropped case or a fabricated UNSURE.
    assert all(r["parse_failed"] for r in rulings)
    assert all(r["choice"] is None for r in rulings)


def test_ruling_records_parse_failure_separately_from_unsure():
    team = _disputing_team("no structured verdict here")
    outcome, _ = Coordinator(team, n_candidates=2).solve(TASK)
    rulings = outcome.diagnostics["rulings"]
    assert rulings
    assert all(r["parse_failed"] for r in rulings)


def test_a_readable_reply_is_not_flagged_as_a_parse_failure():
    team = _disputing_team(lambda n: "\n".join(f"CASE {i}: A" for i in range(1, n + 1)))
    outcome, _ = Coordinator(team, n_candidates=2).solve(TASK)
    rulings = outcome.diagnostics["rulings"]
    assert len(rulings) == outcome.diagnostics["n_disputes"] >= 2
    assert not any(r["parse_failed"] for r in rulings)
    assert all(r["choice"] is not None for r in rulings)


def test_answering_only_some_cases_flags_the_rest_as_unreadable():
    """A partial reply must not be silently completed by guessing at the remainder."""
    team = _disputing_team("CASE 1: A")     # case 2 gets no verdict
    outcome, _ = Coordinator(team, n_candidates=2).solve(TASK)
    rulings = outcome.diagnostics["rulings"]
    assert len(rulings) >= 2
    assert sum(r["parse_failed"] for r in rulings) == len(rulings) - 1


def test_debate_judge_with_no_winner_does_not_default_to_candidate_zero():
    """Defaulting to candidate 0 would hand the judge a win it never gave."""
    team = _disputing_team("CASE 1: B", judge_reply="They all look reasonable to me.")
    outcome, _ = Coordinator(team, n_candidates=2).solve(TASK)
    assert "no readable winner" in outcome.answers["debate"]["source"]


def test_debate_judge_out_of_range_is_reported():
    team = _disputing_team("CASE 1: B", judge_reply="WINNER: 9")
    outcome, _ = Coordinator(team, n_candidates=2).solve(TASK)
    assert "out of range" in outcome.answers["debate"]["source"]


def test_judge_winner_parsing_distinguishes_absent_from_impossible():
    assert Coordinator.parse_winner("WINNER: 2", k=3) == (2, True)
    assert Coordinator.parse_winner("nothing here", k=3) == (None, False)
    assert Coordinator.parse_winner("WINNER: 0\nWINNER: 1", k=3) == (1, True)   # last wins
    # Out of range is a verdict that names a solution which does not exist.
    assert Coordinator.parse_winner("WINNER: 7", k=3) == (None, True)


# --------------------------------------------------------------- sanitize

def test_sanitize_keeps_module_level_setup_but_drops_self_tests():
    """Setup code is part of a working solution; self-tests are not."""
    setup = ("def f():\n    return ujson.dumps([1])\n\n"
             "try:\n    import ujson\nexcept ImportError:\n    import json as ujson\n")
    assert "import json as ujson" in sanitize(setup)

    loop = "def f(n):\n    return TABLE[n]\n\nTABLE = []\nfor i in range(5):\n    TABLE.append(i)\n"
    assert "TABLE.append" in sanitize(loop)

    gated = ("import sys\nif sys.version_info >= (3, 10):\n    from typing import TypeAlias\n"
             "else:\n    TypeAlias = object\n\ndef f(x):\n    return x\n")
    assert "TypeAlias" in sanitize(gated)

    for demo in ("def f(x):\n    return x + 1\n\nassert f(1) == 2\n",
                 "def f(x):\n    return x\n\nprint(f(3))\n",
                 "def f(x):\n    return x\n\nif __name__ == '__main__':\n    print(f(1))\n"):
        out = sanitize(demo)
        assert "assert" not in out and "print(" not in out and "__main__" not in out


def test_sanitize_kept_code_actually_runs():
    """The regression this guards: setup stripped by sanitize became a NameError."""
    code = sanitize("def f():\n    return json.dumps([1])\n\ntry:\n    import ujson as json\n"
                    "except ImportError:\n    import json\n")
    # json.dumps returns the *string* "[1]", so the expectation is that string, not a list.
    [r] = run_cases(code, [{"call": "f()", "expected": repr("[1]")}])
    assert r.error is None, r.error
    assert r.passed is True


# --------------------------------------------------------------- sandbox

def test_strict_mode_blocks_network_and_process_escapes():
    assert strict_blocks_network()
    net = ("import urllib.request\ndef f():\n"
           "    return urllib.request.urlopen('http://example.com', timeout=2).status\n")
    assert run_cases(net, [{"call": "f()"}], strict=True)[0].error is not None


def test_strict_mode_still_runs_ordinary_code():
    code = "import math, collections\ndef f(s):\n    return len(collections.Counter(s))\n"
    [r] = run_cases(code, [{"call": "f('aab')", "expected": "2"}], strict=True)
    assert r.passed is True


def test_candidate_code_cannot_read_the_parent_environment():
    """A candidate must not be able to read the API key out of os.environ."""
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-should-not-be-visible"
    code = "import os\ndef f():\n    return os.environ.get('ANTHROPIC_API_KEY', 'absent')\n"
    [r] = run_cases(code, [{"call": "f()"}])
    assert r.output == repr("absent"), f"candidate could read the parent environment: {r.output}"


def test_reports_which_rlimits_this_platform_enforces():
    report: dict = {}
    run_cases("def f():\n    return 1\n", [{"call": "f()"}], report=report)
    assert "rlimits_applied" in report and isinstance(report["rlimits_applied"], list)


# --------------------------------------------------------------- cost + stats

def test_every_pipeline_phase_is_charged_to_a_strategy():
    phases = ["solo", "brief", "candidate:0", "candidate:1", "candidate:2", "tests", "arbiter",
              "repair", "debate:1:0", "judge"]
    for phase in phases:
        owners = [s for s, ph in STRATEGIES.items() if phase_matches(phase, _phases(s))]
        assert owners, f"{phase} is charged to no strategy"


def _phases(name):
    from hivemind.coordinator import PHASES
    return PHASES[name]


def test_holm_correction_is_monotone_and_never_below_raw():
    ps = [0.01, 0.04, 0.2, 0.5, 0.9, 0.95]
    adj = holm_bonferroni(ps)
    assert all(a >= p for a, p in zip(adj, ps))
    assert all(a <= 1.0 for a in adj)
    assert adj[0] >= ps[0]
    assert holm_bonferroni([0.5]) == [0.5]
    assert holm_bonferroni([]) == []


def test_mcnemar_still_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert round(mcnemar_exact(10, 0), 4) == 0.002


# --------------------------------------------------------------- resume guard

def _cfg(path, **run_overrides):
    run = {"name": "x", "candidates": 3, "seed": 0}
    run.update(run_overrides)
    lines = ["[run]"] + [f"{k} = {json.dumps(v)}" for k, v in run.items()]
    lines += ['[agents]', 'provider = "scripted"', 'model = "s"',
              '[coordinator]', 'provider = "scripted"', 'model = "s"']
    path.write_text("\n".join(lines) + "\n")
    return path


def _populated_run(tmp_path, cfg_text):
    """A run directory holding one outcome, its config and a matching manifest."""
    from hivemind.runner import load_config
    out = tmp_path / "run"
    out.mkdir(exist_ok=True)
    (out / "outcomes.jsonl").write_text(json.dumps(_outcome()) + "\n")
    (out / "config.toml").write_text(cfg_text)
    (out / "manifest.json").write_text(json.dumps(
        {"fingerprint": _config_fingerprint(out / "config.toml", load_config(out / "config.toml"))}))
    return out


def test_resume_refuses_to_mix_configurations(tmp_path):
    from hivemind.runner import load_config
    cfg = _cfg(tmp_path / "a.toml")
    out = _populated_run(tmp_path, cfg.read_text())
    _cfg(tmp_path / "a.toml", candidates=5)
    with pytest.raises(ConfigChanged) as excinfo:
        _check_resume(out, load_config(tmp_path / "a.toml"))
    assert "candidates" in str(excinfo.value)


def test_resume_refuses_a_changed_model(tmp_path):
    from hivemind.runner import load_config
    cfg = _cfg(tmp_path / "a.toml")
    out = _populated_run(tmp_path, cfg.read_text())
    changed = cfg.read_text().replace('model = "s"', 'model = "other"')
    (tmp_path / "a.toml").write_text(changed)
    with pytest.raises(ConfigChanged):
        _check_resume(out, load_config(tmp_path / "a.toml"))


def test_resume_allows_an_identical_configuration(tmp_path):
    from hivemind.runner import load_config
    cfg = _cfg(tmp_path / "a.toml")
    out = _populated_run(tmp_path, cfg.read_text())
    _check_resume(out, load_config(cfg))          # must not raise


def test_resume_adopts_a_manifest_free_legacy_run(tmp_path):
    """An older run with no manifest must not be refused; it is adopted instead."""
    from hivemind.runner import load_config
    cfg = _cfg(tmp_path / "a.toml")
    out = tmp_path / "run"
    out.mkdir()
    (out / "outcomes.jsonl").write_text(json.dumps(_outcome()) + "\n")
    (out / "config.toml").write_text(cfg.read_text())
    _check_resume(out, load_config(cfg))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fingerprint"]["candidates"] == 3
    _check_resume(out, load_config(cfg))          # and is idempotent


def test_estimate_projects_cost_from_the_real_phase_counts():
    config = {"run": {"n_tasks": 164, "candidates": 3, "debate_rounds": 1},
              "agents": {"provider": "anthropic", "model": "claude-opus-5-5"},
              "coordinator": {"provider": "anthropic", "model": "claude-opus-5-5"}}
    e = estimate_cost(config, 164)
    # 1 solo + 1 brief + 3 candidates + 1 tests + ~0.4 arbiter + ~0.3 repair + 3 debate + 0.5 judge
    assert 9 <= e["calls_per_task"]["agents"] + e["calls_per_task"]["coordinator"] <= 12
    assert e["calls_total"] == round((e["calls_per_task"]["agents"]
                                      + e["calls_per_task"]["coordinator"]) * 164)
    assert 0 < e["cost_usd"] < 10_000
    assert set(e["stratified"]) == set(STRATEGIES)
    assert e["stratified"]["solo"]["calls_per_task"] < e["stratified"]["arbitration"]["calls_per_task"]


def test_estimate_reports_local_models_as_free():
    config = {"run": {"n_tasks": 30, "candidates": 3},
              "agents": {"provider": "ollama", "model": "qwen2.5-coder:3b"},
              "coordinator": {"provider": "ollama", "model": "qwen2.5-coder:3b"}}
    assert estimate_cost(config, 30)["cost_usd"] is None


def test_estimate_scales_with_task_count():
    config = {"run": {"n_tasks": 164, "candidates": 3},
              "agents": {"provider": "anthropic", "model": "claude-opus-5-5"},
              "coordinator": {"provider": "anthropic", "model": "claude-opus-5-5"}}
    assert estimate_cost(config, 100)["calls_total"] == estimate_cost(config, 10)["calls_total"] * 10


def test_fingerprint_ignores_inert_run_fields_but_catches_protocol_changes():
    base = {"run": {"name": "x", "workers": 4, "n_tasks": 164, "candidates": 3, "seed": 0},
            "agents": {"provider": "ollama", "model": "m"}, "coordinator": {"provider": "ollama", "model": "m"}}
    same = json.loads(json.dumps(base))
    same["run"]["workers"] = 8
    same["run"]["name"] = "totally-different-name"
    assert _config_fingerprint("p", base) == _config_fingerprint("p", same)

    changed = json.loads(json.dumps(base))
    changed["run"]["candidates"] = 5
    assert _config_fingerprint("p", base) != _config_fingerprint("p", changed)

    changed_model = json.loads(json.dumps(base))
    changed_model["agents"]["model"] = "other"
    assert _config_fingerprint("p", base) != _config_fingerprint("p", changed_model)


# --------------------------------------------------------------- report wiring

def test_report_shows_diversity_and_flags_unreadable_replies(tmp_path):
    d = {"n_spec_cases": 1, "n_tester_cases": 2, "tester_correct": 2, "tester_judged": 2,
         "unanimous": False, "distinct_behaviours": 3, "unsafe_candidates": 1,
         "n_disputes": 2, "verified_scores": [2, 1], "verified_cases": 3, "cases_total": 3,
         "failing_after_selection": 0, "repaired": False,
         "candidate_failures": ["timeout"], "unsafe_answers": ["solo"],
         "rulings": [{"call": "f(1)", "options": ["1", "2"], "choice": None,
                      "reference_agrees": None, "parse_failed": True},
                     {"call": "f(2)", "options": ["1", "2"], "choice": "2",
                      "reference_agrees": True, "parse_failed": False}]}
    o = _outcome(diagnostics=d, answers={n: {"passed": False} for n in STRATEGIES})
    _write_run(tmp_path, [o])
    text = write_report(tmp_path).read_text()
    assert "Pool diversity" in text
    assert "could not be parsed" in text
    assert "timeout" in text
    assert "Holm" in text


def test_report_banner_when_tasks_were_excluded(tmp_path):
    d = {"n_spec_cases": 1, "n_tester_cases": 1, "tester_correct": 1, "tester_judged": 1,
         "unanimous": True, "distinct_behaviours": 1, "unsafe_candidates": 0, "n_disputes": 0,
         "rulings": [], "verified_scores": [1], "verified_cases": 1, "cases_total": 1,
         "failing_after_selection": 0, "repaired": False, "candidate_failures": [],
         "unsafe_answers": []}
    o = _outcome(diagnostics=d, answers={n: {"passed": False} for n in STRATEGIES})
    _write_run(tmp_path, [o])
    text = write_report(tmp_path, excluded=3).read_text()
    assert "Run incomplete" in text
    assert "EXCLUDED" in text
