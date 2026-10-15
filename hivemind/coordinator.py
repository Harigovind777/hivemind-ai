"""The Coordinator: decides which agents act, then settles their disagreements.

For every task the coordinator builds ONE shared pool of candidate solutions and
applies every resolution strategy to that same pool. Differences between
strategies therefore come from how disagreements are resolved, not from luck in
generation, which is what the research question needs.

Evidence hierarchy used by arbitration (strongest first):
  1. worked examples in the specification (never overruled)
  2. the Coordinator's rulings on disputed cases, made from the spec plus the
     real execution outputs of each candidate
  3. the Testing agent's expected values, trusted only when every candidate agrees
"""

from __future__ import annotations

import random
import re
import zlib
from dataclasses import asdict, dataclass, field

from .agents import Team, Transcript
from .domains.code import (CodeTask, build_program, doc_example_cases, extract_code, grade,
                           parse_assert_cases, reference_program)
from .sandbox import CaseResult, run_cases
from .security import static_scan

STRATEGIES = {
    "solo": "A single Coding agent working alone: no brief, no tests, no teammates.",
    "first": "The team's first candidate (Coding agent + Research brief) with no resolution step.",
    "vote": "Majority vote: run every candidate on the team's test inputs and pick the largest "
            "group with identical behaviour.",
    "debate": "Free-form debate: candidates critique each other and revise; the Coordinator "
              "picks a winner from the arguments, with no execution.",
    "tests_only": "Pick the candidate passing the most team-written tests, trusting the Testing "
                  "agent's expected values without checking them.",
    "arbitration_select": "Evidence-based arbitration, selection only: the Coordinator rules on each "
                          "disputed case from the spec and real execution outputs, then the "
                          "candidate matching the verified results is chosen.",
    "arbitration": "Evidence-based arbitration followed by one repair round in which the Coding "
                   "agent sees the verified failing cases; the repair is kept only if it scores higher.",
}

# Which pipeline phases each strategy consumes; used to attribute calls and tokens.
PHASES = {
    "solo": ("solo",),
    "first": ("brief", "candidate:0"),
    "vote": ("brief", "candidate", "tests"),
    "debate": ("brief", "candidate", "debate", "judge"),
    "tests_only": ("brief", "candidate", "tests"),
    "arbitration_select": ("brief", "candidate", "tests", "arbiter"),
    "arbitration": ("brief", "candidate", "tests", "arbiter", "repair"),
}

APPROACH_HINTS = [
    "Write the clearest straightforward solution.",
    "Work through the docstring examples by hand first, then implement exactly the rule they imply.",
    "List the edge cases first (empty input, zero, negatives, duplicates, boundaries), then implement.",
    "Consider an alternative algorithm to the obvious one and use whichever is more robust.",
]


def phase_matches(phase: str, wanted: tuple[str, ...]) -> bool:
    return any(phase == w or phase.startswith(w + ":") for w in wanted)


@dataclass
class Case:
    call: str
    expected: str | None
    source: str  # "spec" (docstring example) or "tester"


# A verified expectation: ("pass",) means "matches the stated expected value";
# ("sig", s) means "produces exactly the observed result s".
Rule = tuple


def _satisfies(rule: Rule, result: CaseResult) -> bool:
    if rule[0] == "pass":
        return result.passed is True
    return result.signature == rule[1]


@dataclass
class Ruling:
    call: str
    options: list[str]
    choice: str | None            # the option text chosen, or None for UNSURE
    reference_agrees: bool | None = None  # analysis only: does the benchmark reference agree?


@dataclass
class TaskOutcome:
    task_id: str
    candidates: list[str]
    candidate_passed: list[bool]
    answers: dict[str, dict] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    review: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChatAnswer:
    """The team's answer to a problem with no hidden tests (the Chat tab)."""

    code: str
    source: str
    candidates: list[str]
    unsafe: list[bool]
    working: bool            # the final code produced a result on at least one case
    verified_total: int
    verified_passed: int
    spec_cases: int          # verified cases taken from the spec's own examples
    ruled_cases: int         # verified by a Coordinator ruling
    undisputed_cases: int    # verified because no one disagreed
    failures: list[dict]     # verified cases the final code fails: call, expected, got
    rulings: list[dict]
    distinct_behaviours: int
    repaired: bool


@dataclass
class _Pool:
    """The shared middle of a solve: candidates, their evidence, and arbitration's choice."""

    candidates: list[str]
    cases: list[Case]
    results: list[list[CaseResult]]
    unsafe: list[bool]
    sigs: list[tuple]
    cluster: list[int]
    verified: dict[int, Rule]
    rulings: list[tuple[int, Ruling, Rule | None]]
    disputes: int
    scores: list[int]
    arb_code: str
    arb_source: str
    arb_results: list[CaseResult]  # the final code's results (the repair's, when it was kept)
    failing: list[int]             # verified cases the pick failed before any repair
    repaired: bool


class Coordinator:
    def __init__(self, team: Team, n_candidates: int = 3, debate_rounds: int = 1,
                 repair: bool = True, seed: int = 0, max_disputes: int = 8):
        self.team = team
        self.k = n_candidates
        self.debate_rounds = debate_rounds
        self.repair = repair
        self.seed = seed
        self.max_disputes = max_disputes

    # ------------------------------------------------------------------ pipeline

    def solve(self, task: CodeTask, review: bool = False, on_event=None,
              cancel=None) -> tuple[TaskOutcome, Transcript]:
        """Solve one task. `on_event` receives live progress events (see Transcript.emit);
        setting the `cancel` Event stops the run before its next model call."""
        tr = Transcript(task.task_id, on_event=on_event, cancel=cancel)
        seed = self._seed(task)
        ask = self._asker(tr)

        solo = self._code(ask("coding", self._coder_prompt(task), "solo", seed), task)
        pool = self._pool(task, ask, seed, tr)
        candidates, results, unsafe, cluster = pool.candidates, pool.results, pool.unsafe, pool.cluster
        vote_pick = min(range(self.k), key=lambda i: (unsafe[i], -cluster[i], i))
        stated = [sum(r.passed is True for r in res) for res in results]
        tests_pick = min(range(self.k), key=lambda i: (unsafe[i], -stated[i], -cluster[i], i))
        arb_pick = int(pool.arb_source.rsplit(" ", 1)[1])

        debate_code, debate_source = self._debate(task, candidates, ask, seed)

        grades: dict[str, bool] = {}

        def graded(code: str) -> bool:
            if code not in grades:
                grades[code] = grade(task, code)
            return grades[code]

        candidate_passed = [graded(c) for c in candidates]
        answers = {
            "solo": (solo, "solo coder"),
            "first": (candidates[0], "candidate 0"),
            "vote": (candidates[vote_pick], f"candidate {vote_pick}"),
            "debate": (debate_code, debate_source),
            "tests_only": (candidates[tests_pick], f"candidate {tests_pick}"),
            "arbitration_select": (candidates[arb_pick], f"candidate {arb_pick}"),
            "arbitration": (pool.arb_code, pool.arb_source),
        }
        outcome = TaskOutcome(
            task_id=task.task_id,
            candidates=candidates,
            candidate_passed=candidate_passed,
            answers={name: {"code": code, "source": src, "passed": graded(code)}
                     for name, (code, src) in answers.items()},
            diagnostics=self._diagnostics(task, pool.cases, pool.sigs, unsafe, pool.rulings,
                                          pool.disputes, pool.scores, pool.failing, pool.repaired),
        )
        if review:
            outcome.review = self.review(task, pool.arb_code, ask, seed)
        return outcome, tr

    def answer(self, task: CodeTask, on_event=None, cancel=None) -> tuple[ChatAnswer, Transcript]:
        """The shared steps alone, for a problem with no hidden tests: no solo baseline, no
        debate, no grading. Used by the Chat tab."""
        tr = Transcript(task.task_id, on_event=on_event, cancel=cancel)
        pool = self._pool(task, self._asker(tr), self._seed(task), tr)
        failures = [{"call": pool.cases[j].call, "expected": self._expected(pool.cases[j], rule),
                     "got": pool.arb_results[j].signature}
                    for j, rule in sorted(pool.verified.items())
                    if not _satisfies(rule, pool.arb_results[j])]
        spec_cases = sum(c.source == "spec" for c in pool.cases)
        ruled = sum(rule is not None for _, _, rule in pool.rulings)
        working = bool(pool.arb_code) and (not pool.arb_results
                                           or any(r.is_answer for r in pool.arb_results))
        answer = ChatAnswer(
            code=pool.arb_code, source=pool.arb_source, candidates=pool.candidates,
            unsafe=pool.unsafe, working=working,
            verified_total=len(pool.verified), verified_passed=len(pool.verified) - len(failures),
            spec_cases=spec_cases, ruled_cases=ruled,
            undisputed_cases=len(pool.verified) - spec_cases - ruled,
            failures=failures, rulings=[asdict(r) for _, r, _ in pool.rulings],
            distinct_behaviours=len(set(pool.sigs)), repaired=pool.repaired)
        return answer, tr

    def _pool(self, task: CodeTask, ask, seed: int, tr: Transcript) -> _Pool:
        """Brief, candidates, team tests, sandbox, arbitration and repair: the steps every
        strategy and the Chat tab share."""
        brief = ask("research", f"Analyse this function specification for the team.\n\n{task.spec}",
                    "brief", seed)
        candidates = [
            self._code(ask("coding", self._coder_prompt(task, brief, APPROACH_HINTS[i % len(APPROACH_HINTS)]),
                           f"candidate:{i}", seed + 1 + i), task)
            for i in range(self.k)
        ]
        tests_text = ask("testing", self._tester_prompt(task), "tests", seed)
        cases = self._cases(task, tests_text)
        case_dicts = [{"call": c.call, "expected": c.expected} for c in cases]
        results = [run_cases(build_program(task, c), case_dicts) for c in candidates]
        unsafe = [bool(static_scan(c)) for c in candidates]

        sigs = [tuple(r.signature for r in res) for res in results]
        tr.emit("evidence", candidates=self.k, cases=len(cases),
                spec_cases=sum(c.source == "spec" for c in cases),
                tester_cases=sum(c.source == "tester" for c in cases),
                distinct_behaviours=len(set(sigs)), unsafe_candidates=sum(unsafe))
        runnable = [not res or any(r.is_answer for r in res) for res in results]
        # Programs that never produce a result cannot form a voting bloc.
        cluster = [sum(s == t and ok for t, ok in zip(sigs, runnable)) if runnable[i] else 0
                   for i, s in enumerate(sigs)]

        verified, rulings, disputes = self._arbitrate(task, cases, results, ask, seed)
        tr.emit("rulings", disputes=disputes, rulings=[asdict(r) for _, r, _ in rulings])

        def score(res: list[CaseResult]) -> int:
            return sum(_satisfies(rule, res[j]) for j, rule in verified.items())

        scores = [score(res) for res in results]
        arb_pick = min(range(self.k), key=lambda i: (unsafe[i], -scores[i], -cluster[i], i))
        arb_code, arb_source, arb_results = candidates[arb_pick], f"candidate {arb_pick}", results[arb_pick]
        failing = [j for j, rule in verified.items() if not _satisfies(rule, results[arb_pick][j])]
        repaired = False
        if self.repair and failing:
            fixed = self._code(ask("coding", self._repair_prompt(task, arb_code, cases, verified,
                                                                  results[arb_pick], failing),
                                   "repair", seed + 50), task)
            if fixed and not static_scan(fixed):
                fixed_results = run_cases(build_program(task, fixed), case_dicts)
                if score(fixed_results) > scores[arb_pick]:
                    arb_code, arb_source, arb_results, repaired = (
                        fixed, f"repair of candidate {arb_pick}", fixed_results, True)
        return _Pool(candidates, cases, results, unsafe, sigs, cluster, verified, rulings, disputes,
                     scores, arb_code, arb_source, arb_results, failing, repaired)

    def review(self, task: CodeTask, code: str, ask, seed: int) -> dict:
        """Review mode: the Security, Documentation and Ethics agents read the final answer."""
        shown = f"Task:\n{task.spec}\n\nFinal solution:\n```python\n{code}\n```"
        return {
            "static_security_findings": static_scan(code),
            "security": ask("security", shown, "review:security", seed),
            "documentation": ask("documentation", shown, "review:documentation", seed),
            "ethics": ask("ethics", shown, "review:ethics", seed),
        }

    # ------------------------------------------------------------------ arbitration

    def _arbitrate(self, task, cases, results, ask, seed):
        """Rule on disputed cases. Returns (verified rules by case index, rulings, #disputes)."""
        verified: dict[int, Rule] = {}
        disputed: list[tuple[int, list[tuple[str, Rule]], bool]] = []
        for j, case in enumerate(cases):
            if case.source == "spec":
                verified[j] = ("pass",)
                continue
            column = [res[j] for res in results]
            # Only programs that produced a result make a claim about this case.
            answers = [r for r in column if r.is_answer]
            if answers and all(r.passed is True for r in answers):
                verified[j] = ("pass",)
                continue
            options: list[tuple[str, Rule]] = []
            if any(r.passed is not None for r in column):  # the stated expectation is usable
                options.append((case.expected, ("pass",)))
            for r in answers:
                if r.passed is not True and all(o[0] != r.signature for o in options):
                    options.append((r.signature, ("sig", r.signature)))
            if len(options) == 1:
                verified[j] = options[0][1]  # a single claim with no counter-evidence
            elif len(options) >= 2:
                split = len({r.signature for r in answers}) > 1
                disputed.append((j, options, split))
        # Cases where the candidates disagree with each other come first.
        disputed.sort(key=lambda d: not d[2])
        disputed = disputed[: self.max_disputes]
        rulings: list[tuple[int, Ruling, Rule | None]] = []
        if not disputed:
            return verified, rulings, 0

        rng = random.Random(seed)
        shown = []
        for j, options, _ in disputed:
            options = options[:]
            rng.shuffle(options)  # hide which option the Testing agent proposed
            shown.append((j, options))
        reply = ask("coordinator", self._arbiter_prompt(task, cases, shown), "arbiter", seed)
        picks = {int(n): v.upper() for n, v in
                 re.findall(r"CASE\s*(\d+)\s*[:\-]\s*\**\s*([A-Za-z]+)", reply)}
        for n, (j, options) in enumerate(shown, start=1):
            letter = picks.get(n, "UNSURE")
            idx = ord(letter[0]) - ord("A") if len(letter) == 1 else -1
            if 0 <= idx < len(options):
                text, rule = options[idx]
                verified[j] = rule
                rulings.append((j, Ruling(cases[j].call, [o[0] for o in options], text), rule))
            else:
                rulings.append((j, Ruling(cases[j].call, [o[0] for o in options], None), None))
        return verified, rulings, len(shown)

    # ------------------------------------------------------------------ debate

    def _debate(self, task, candidates, ask, seed):
        codes, arguments = list(candidates), [""] * self.k
        for rnd in range(1, self.debate_rounds + 1):
            revised = []
            for i in range(self.k):
                text = ask("coding", self._debate_prompt(task, codes, i), f"debate:{rnd}:{i}",
                           seed + 100 * rnd + i)
                revised.append(self._code(text, task) or codes[i])
                arguments[i] = text.split("```")[0].strip()[:800]
            codes = revised
        if len(set(codes)) == 1:
            return codes[0], "unanimous after debate"
        reply = ask("coordinator", self._judge_prompt(task, codes, arguments), "judge", seed)
        found = [int(n) for n in re.findall(r"WINNER\s*:\s*(?:solution\s*)?(\d+)", reply, re.I)]
        winner = found[-1] if found and found[-1] < self.k else 0
        return codes[winner], f"debate winner {winner}"

    # ------------------------------------------------------------------ analysis

    def _diagnostics(self, task, cases, sigs, unsafe, rulings, disputes, scores, failing, repaired):
        tester = [c for c in cases if c.source == "tester"]
        tester_idx = [j for j, c in enumerate(cases) if c.source == "tester"]
        ref = run_cases(reference_program(task), [{"call": c.call, "expected": c.expected}
                                                  for c in cases]) if cases else []
        judged = [ref[j].passed for j in tester_idx if ref[j].passed is not None]
        for j, ruling, rule in rulings:
            if rule is not None:
                ruling.reference_agrees = _satisfies(rule, ref[j])
        return {
            "n_spec_cases": len(cases) - len(tester),
            "n_tester_cases": len(tester),
            "tester_correct": sum(judged),
            "tester_judged": len(judged),
            "unanimous": len(set(sigs)) == 1,
            "distinct_behaviours": len(set(sigs)),
            "unsafe_candidates": sum(unsafe),
            "n_disputes": disputes,
            "rulings": [asdict(r) for _, r, _ in rulings],
            "verified_scores": scores,
            "failing_after_selection": len(failing),
            "repaired": repaired,
        }

    # ------------------------------------------------------------------ helpers

    def _asker(self, tr: Transcript):
        def ask(role: str, prompt: str, phase: str, seed: int) -> str:
            return self.team[role].ask(prompt, transcript=tr, phase=phase, seed=seed)
        return ask

    def _seed(self, task: CodeTask) -> int:
        return (self.seed * 7919 + zlib.crc32(task.task_id.encode())) % 2**31

    @staticmethod
    def _expected(case: Case, rule: Rule) -> str:
        """What a verified case requires, as text."""
        return case.expected if rule[0] == "pass" else rule[1]

    @staticmethod
    def _code(text: str, task: CodeTask) -> str:
        return extract_code(text, task.entry_point) if text.strip() else ""

    @staticmethod
    def _cases(task: CodeTask, tests_text: str) -> list[Case]:
        spec = [Case(c["call"], c["expected"], "spec") for c in doc_example_cases(task)]
        seen = {c.call for c in spec}
        tester = [Case(c["call"], c["expected"], "tester")
                  for c in parse_assert_cases(tests_text, task.entry_point) if c["call"] not in seen]
        return spec + tester

    @staticmethod
    def _coder_prompt(task: CodeTask, brief: str | None = None, hint: str | None = None) -> str:
        parts = [f"Implement this Python function.\n\n{task.spec}"]
        if brief:
            parts.append(f"Notes from the Research agent:\n{brief.strip()}")
        if hint:
            parts.append(f"Approach: {hint}")
        parts.append("End with one ```python block containing the complete function.")
        return "\n\n".join(parts)

    @staticmethod
    def _tester_prompt(task: CodeTask) -> str:
        return (
            f"Write 8 test cases for the function below, using only its specification.\n\n"
            f"{task.spec}\n\n"
            f"Reply with one ```python block of assert statements, one per line, each exactly of "
            f"the form:\nassert {task.entry_point}(<arguments>) == <expected value>\n"
            f"Use literal values only. Do not repeat the docstring's own examples; cover typical "
            f"inputs and edge cases."
        )

    @staticmethod
    def _arbiter_prompt(task, cases, shown) -> str:
        lines = [
            "Team members wrote implementations of the function below and test cases for it. "
            "When the implementations were executed, they produced conflicting results on the "
            "cases listed. For each case, decide which result the specification requires.",
            f"Specification:\n{task.spec}",
            "Disputed cases (results are real execution outputs shown as Python values; "
            "<Name> means the call raised that exception):",
        ]
        for n, (j, options) in enumerate(shown, start=1):
            opts = "\n".join(f"  {chr(ord('A') + i)}) {text}" for i, (text, _) in enumerate(options))
            lines.append(f"CASE {n}: {cases[j].call}\n{opts}")
        lines.append(
            "Work through each case using the specification's rules and examples. Then finish with "
            "one line per case, exactly in the form\nCASE <number>: <letter>\n"
            "Use UNSURE instead of a letter only if the specification truly does not determine it."
        )
        return "\n\n".join(lines)

    @staticmethod
    def _repair_prompt(task, code, cases, verified, results, failing) -> str:
        evidence = []
        for j in failing:
            rule = verified[j]
            expected = cases[j].expected if rule[0] == "pass" else rule[1]
            why = "specification example" if cases[j].source == "spec" else "verified by the Coordinator"
            evidence.append(f"- {cases[j].call}\n  expected: {expected} ({why})\n"
                            f"  got: {results[j].signature}")
        return (
            f"Your implementation of the function below fails cases whose correct results have "
            f"been verified.\n\n{task.spec}\n\nYour code:\n```python\n{code}\n```\n\n"
            f"Failing cases:\n" + "\n".join(evidence) +
            "\n\nFix the implementation so it satisfies the specification, including these cases. "
            "End with one ```python block containing the complete corrected function."
        )

    @staticmethod
    def _debate_prompt(task, codes, i) -> str:
        others = "\n\n".join(f"Solution {j}:\n```python\n{c}\n```"
                             for j, c in enumerate(codes) if j != i)
        return (
            f"Your team is implementing this function.\n\n{task.spec}\n\n"
            f"Your current solution (Solution {i}):\n```python\n{codes[i]}\n```\n\n"
            f"Your teammates' solutions:\n\n{others}\n\n"
            "Compare all solutions against the specification and briefly point out any bugs. "
            "Then give your final answer: keep your solution or adopt a better one. "
            "End with one ```python block containing your complete final function."
        )

    @staticmethod
    def _judge_prompt(task, codes, arguments) -> str:
        shown = "\n\n".join(
            f"Solution {i}\nAuthor's argument: {arguments[i] or '(none)'}\n```python\n{c}\n```"
            for i, c in enumerate(codes))
        return (
            f"Your team debated how to implement this function and ended with these solutions. "
            f"Choose the one most likely to be fully correct according to the specification.\n\n"
            f"{task.spec}\n\n{shown}\n\n"
            "Give a short justification, then finish with a line exactly of the form\nWINNER: <number>"
        )
