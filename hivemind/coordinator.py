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
from .domains.code import (CodeTask, build_program, doc_example_cases, extract_code,
                           grade_detailed, parse_assert_cases, reference_program)
from .domains.freeform import evidence_grade
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
    choice: str | None            # the option text chosen, or None for UNSURE/unreadable
    reference_agrees: bool | None = None  # analysis only: does the benchmark reference agree?
    parse_failed: bool = False    # the arbiter's verdict could not be read at all


@dataclass
class TaskOutcome:
    task_id: str
    candidates: list[str]
    candidate_passed: list[bool]
    answers: dict[str, dict] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    review: dict | None = None
    # True when the task has no hidden test suite, so `passed` means "satisfied the team's
    # own verified cases" rather than "passed the benchmark". Never compare the two.
    evidence_only: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class NoUsableOutput(RuntimeError):
    """No candidate produced any code, so nothing about this task can be measured.

    Raised instead of returning an all-zero outcome. Without it a run with a dead API key
    or a wrong model name writes a complete, plausible-looking report in which every
    strategy scores 0% -- indistinguishable from a real negative result.
    """


class Coordinator:
    def __init__(self, team: Team, n_candidates: int = 3, debate_rounds: int = 1,
                 repair: bool = True, seed: int = 0, max_disputes: int = 8,
                 min_usable: int = 1, strict: bool = False):
        self.team = team
        self.k = n_candidates
        self.debate_rounds = debate_rounds
        self.repair = repair
        self.seed = seed
        self.max_disputes = max_disputes
        # Refuse to score a task with fewer than this many non-empty candidates. 1 keeps
        # the existing behaviour for one flaky call while still catching a dead provider.
        self.min_usable = min_usable
        self.strict = strict

    # ------------------------------------------------------------------ pipeline

    def solve(self, task: CodeTask, review: bool = False) -> tuple[TaskOutcome, Transcript]:
        tr = Transcript(task.task_id)
        seed = (self.seed * 7919 + zlib.crc32(task.task_id.encode())) % 2**31
        ask = self._asker(tr)

        solo = self._code(ask("coding", self._coder_prompt(task), "solo", seed), task)
        brief = ask("research", f"Analyse this function specification for the team.\n\n{task.spec}",
                    "brief", seed)
        candidates = [
            self._code(ask("coding", self._coder_prompt(task, brief, APPROACH_HINTS[i % len(APPROACH_HINTS)]),
                           f"candidate:{i}", seed + 1 + i), task)
            for i in range(self.k)
        ]
        usable = [c for c in candidates if c.strip()]
        if len(usable) < self.min_usable:
            errors = [t.get("error") for t in tr.turns if t.get("error")]
            raise NoUsableOutput(
                f"{task.task_id}: {len(usable)}/{self.k} candidates were empty"
                + (f"; first error: {errors[0]}" if errors else " (no error recorded)"))
        tests_text = ask("testing", self._tester_prompt(task), "tests", seed)
        cases = self._cases(task, tests_text)
        case_dicts = [{"call": c.call, "expected": c.expected} for c in cases]
        results = [self._run(build_program(task, c), case_dicts) for c in candidates]
        # Every code string the pipeline produces is scanned, not just the base candidates:
        # `solo` and the debate output used to bypass this gate entirely, so an unsafe
        # answer could win those strategies while the report counted only base candidates.
        unsafe = [bool(static_scan(c)) for c in candidates]

        sigs = [tuple(r.signature for r in res) for res in results]
        runnable = [not res or any(r.is_answer for r in res) for res in results]
        # Programs that never produce a result cannot form a voting bloc.
        cluster = [sum(s == t and ok for t, ok in zip(sigs, runnable)) if runnable[i] else 0
                   for i, s in enumerate(sigs)]
        vote_pick = min(range(self.k), key=lambda i: (unsafe[i], -cluster[i], i))

        stated = [sum(r.passed is True for r in res) for res in results]
        tests_pick = min(range(self.k), key=lambda i: (unsafe[i], -stated[i], -cluster[i], i))

        verified, rulings, disputes = self._arbitrate(task, cases, results, ask, seed)

        def score(res: list[CaseResult]) -> int:
            return sum(_satisfies(rule, res[j]) for j, rule in verified.items())

        scores = [score(res) for res in results]
        arb_pick = min(range(self.k), key=lambda i: (unsafe[i], -scores[i], -cluster[i], i))
        arb_code, arb_source = candidates[arb_pick], f"candidate {arb_pick}"
        failing = [j for j, rule in verified.items() if not _satisfies(rule, results[arb_pick][j])]
        repaired = False
        if self.repair and failing:
            fixed = self._code(ask("coding", self._repair_prompt(task, arb_code, cases, verified,
                                                                  results[arb_pick], failing),
                                   "repair", seed + 50), task)
            if fixed and not static_scan(fixed):
                fixed_score = score(self._run(build_program(task, fixed), case_dicts))
                if fixed_score > scores[arb_pick]:
                    arb_code, arb_source, repaired = fixed, f"repair of candidate {arb_pick}", True

        debate_code, debate_source = self._debate(task, candidates, ask, seed)

        # A task with no hidden test suite (a prompt typed into the web UI) cannot be
        # graded against ground truth. Rather than let `grade()` return False for
        # everything and report that as a 0% pass rate, fall back to the team's own
        # verified cases and label every number as evidence, not accuracy.
        evidence_only = not task.hidden_test.strip()
        grades: dict[str, tuple[bool, str]] = {}
        evidence: dict[str, dict] = {}

        def graded(code: str) -> bool:
            if evidence_only:
                if code not in evidence:
                    evidence[code] = evidence_grade(code, task, case_dicts, verified)
                total = evidence[code]["verified_total"]
                return (total > 0
                        and evidence[code]["verified_satisfied"] == total)
            if code not in grades:
                grades[code] = grade_detailed(task, code)
            return grades[code][0]

        candidate_passed = [graded(c) for c in candidates]
        answers = {
            "solo": (solo, "solo coder"),
            "first": (candidates[0], "candidate 0"),
            "vote": (candidates[vote_pick], f"candidate {vote_pick}"),
            "debate": (debate_code, debate_source),
            "tests_only": (candidates[tests_pick], f"candidate {tests_pick}"),
            "arbitration_select": (candidates[arb_pick], f"candidate {arb_pick}"),
            "arbitration": (arb_code, arb_source),
        }
        # Flag answers that reached a strategy without passing the security gate, so the
        # report can say how often the gate changed the outcome instead of implying it
        # never does.
        flagged = {name: static_scan(code) for name, (code, _) in answers.items()
                   if static_scan(code)}
        def grade_detail(code: str) -> str:
            if evidence_only:
                e = evidence[code]
                if e.get("error"):
                    return e["error"]
                return (f"{e['verified_satisfied']}/{e['verified_total']} verified cases "
                        f"({e['stated_passed']}/{e['stated_total']} stated)")
            return grades[code][1]

        outcome = TaskOutcome(
            task_id=task.task_id,
            candidates=candidates,
            candidate_passed=candidate_passed,
            answers={name: {"code": code, "source": src, "passed": graded(code),
                            "grade_reason": grade_detail(code),
                            "security_findings": flagged.get(name, []),
                            **({"evidence": evidence[code]} if evidence_only else {})}
                     for name, (code, src) in answers.items()},
            diagnostics=self._diagnostics(task, cases, sigs, unsafe, rulings, disputes, scores,
                                          failing, repaired, grades, flagged, solo, debate_code,
                                          len(verified), evidence_only),
        )
        outcome.evidence_only = evidence_only
        if review:
            outcome.review = self.review(task, arb_code, ask, seed)
        return outcome, tr

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
        picks, unreadable = self.parse_case_verdicts(reply, len(shown))
        for n, (j, options) in enumerate(shown, start=1):
            raw = picks.get(n)
            if raw is None:
                # We could not read a verdict. Leave the case unverified rather than
                # guessing, and flag it: a silently dropped case would otherwise shrink
                # the set a candidate has to satisfy and flatter every strategy.
                rulings.append((j, Ruling(cases[j].call, [o[0] for o in options], None,
                                          parse_failed=True), None))
                continue
            if raw == "UNSURE":
                rulings.append((j, Ruling(cases[j].call, [o[0] for o in options], None), None))
                continue
            idx = ord(raw[0]) - ord("A")
            if 0 <= idx < len(options):
                text, rule = options[idx]
                verified[j] = rule
                rulings.append((j, Ruling(cases[j].call, [o[0] for o in options], text), rule))
            else:
                # A letter outside the offered range: the model answered a question we
                # did not ask. Record it as unreadable rather than picking option A.
                rulings.append((j, Ruling(cases[j].call, [o[0] for o in options], None,
                                          parse_failed=True), None))
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
        winner, verdict_present = self.parse_winner(reply, len(codes))
        if winner is None:
            # No usable verdict. Defaulting to candidate 0 would hand the judge a silent
            # win it never gave, and would systematically favour whichever solution the
            # pipeline happened to generate first. Fall back to the majority-behaviour
            # pick instead and say so in the source string.
            tally = max(range(len(codes)), key=lambda i: (sum(codes[j] == codes[i]
                                                             for j in range(len(codes))), -i))
            return codes[tally], ("judge winner out of range" if verdict_present
                                  else "judge gave no readable winner")
        return codes[winner], f"debate winner {winner}"

    # ------------------------------------------------------------------ analysis

    def _diagnostics(self, task, cases, sigs, unsafe, rulings, disputes, scores, failing, repaired,
                     grades, flagged, solo, debate_code, n_verified, evidence_only=False):
        tester = [c for c in cases if c.source == "tester"]
        tester_idx = [j for j, c in enumerate(cases) if c.source == "tester"]
        # The reference is run outside strict mode: it is the benchmark's own solution,
        # not model-written code, and it must stay a trustworthy yardstick. Without one
        # (a user-typed prompt) there is nothing to check the rulings against, and those
        # agreement figures are omitted rather than faked.
        has_reference = bool(task.canonical_solution.strip())
        ref = run_cases(reference_program(task), [{"call": c.call, "expected": c.expected}
                                                  for c in cases]) if (cases and has_reference) else []
        judged = [ref[j].passed for j in tester_idx if ref[j].passed is not None] if ref else []
        if ref:
            for j, ruling, rule in rulings:
                if rule is not None and j < len(ref):
                    ruling.reference_agrees = _satisfies(rule, ref[j])
        return {
            "n_spec_cases": len(cases) - len(tester),
            "n_tester_cases": len(tester),
            "tester_correct": sum(judged),
            "tester_judged": len(judged),
            "has_reference": has_reference,
            "unanimous": len(set(sigs)) == 1,
            "distinct_behaviours": len(set(sigs)),
            # The precondition for every hypothesis: if K candidates rarely disagree there
            # is nothing for arbitration to arbitrate, and a flat result means the pipeline
            # was saturated rather than that the method failed.
            "max_distinct_behaviours": len(sigs),
            "unsafe_candidates": sum(unsafe),
            "n_disputes": disputes,
            "rulings": [asdict(r) for _, r, _ in rulings],
            "rulings_unreadable": sum(r.parse_failed for _, r, _ in rulings),
            "verified_scores": scores,
            "verified_cases": n_verified,
            "cases_total": len(cases),
            "failing_after_selection": len(failing),
            "repaired": repaired,
            # Why candidates failed, so a slow-but-correct solution is not filed as wrong.
            "candidate_failures": [reason for _, reason in grades.values() if reason != "ok"],
            "unsafe_answers": sorted(flagged),
            "solo_unsafe": bool(static_scan(solo)),
            "debate_unsafe": bool(static_scan(debate_code)),
        }

    # ------------------------------------------------------------------ reply parsing

    # Model replies vary in how they mark up a verdict: "CASE 1: B", "**CASE 2:** `C`",
    # "CASE 3 - the spec requires A", "CASE 4: Option B". Anything still unreadable is
    # recorded as an explicit parse failure rather than being folded into the model's own
    # "UNSURE", because the two mean very different things for the report: one is the
    # arbiter declining to decide, the other is us failing to read what it decided.
    _CASE_HEAD = re.compile(r"CASE\s*(\d+)\s*(?:[:\-\u2013\u2014]|\b)", re.I)
    # A verdict is either the bare letter or a spelled-out judgement. Captured separately
    # from the case number so that prose between the two ("CASE 1: the spec requires A")
    # is read rather than mistaken for the verdict itself.
    _VERDICT = re.compile(r"(?:^|[:\-\u2013\u2014\*`\s(])"
                          r"(?:option\s+|answer\s+|choose\s+)?"
                          r"(UNSURE|[A-Za-z])\b(?=\s*[\.\,\;\`\*\)]|\s*$)", re.I)
    _NARRATED = re.compile(r"\b(?:spec(?:ification)?|answer|requires|implies|output is|must be|"
                           r"correct (?:is|output)|therefore|so the answer)\b"
                           r"[^A-Za-z]{0,32}\(?([A-Z])\)?\b", re.I)

    @classmethod
    def parse_case_verdicts(cls, reply: str, n_cases: int) -> tuple[dict[int, str], list[int]]:
        """Map 1-based case numbers to letters. Returns (verdicts, cases_we_failed_to_read)."""
        verdicts: dict[int, str] = {}
        # Each case owns the text from its own heading to the next one, so a verdict can
        # never be attributed to the wrong case.
        spans = [m for m in cls._CASE_HEAD.finditer(reply) if 1 <= int(m.group(1)) <= n_cases]
        for i, match in enumerate(spans):
            n = int(match.group(1))
            end = spans[i + 1].start() if i + 1 < len(spans) else len(reply)
            body = reply[match.end():end]
            verdict = cls._VERDICT.search(body)
            if verdict:
                word = verdict.group(1).upper()
                verdicts.setdefault(n, "UNSURE" if word == "UNSURE" else word)
                continue
            narrated = cls._NARRATED.search(body)
            if narrated:
                verdicts.setdefault(n, narrated.group(1).upper())
        unreadable = [n for n in range(1, n_cases + 1) if n not in verdicts]
        return verdicts, unreadable

    @staticmethod
    def parse_winner(reply: str, k: int) -> tuple[int | None, bool]:
        """Read `WINNER: <n>`.

        Returns (index, verdict_present). `index is None` with `verdict_present` False means
        the judge said nothing usable; with True it means the judge named a solution that
        does not exist. Neither is a win by candidate 0.
        """
        found = [int(n) for n in re.findall(r"WINNER\s*[:\-]\s*(?:solution\s*)?(\d+)",
                                            reply, re.I)]
        if not found:
            return None, False          # nothing to read at all
        winner = found[-1]
        if not 0 <= winner < k:
            return None, True            # a verdict was given, but of a solution that is not there
        return winner, True

    # ------------------------------------------------------------------ helpers

    def _asker(self, tr: Transcript):
        def ask(role: str, prompt: str, phase: str, seed: int) -> str:
            return self.team[role].ask(prompt, transcript=tr, phase=phase, seed=seed)
        return ask

    def _run(self, program: str, cases: list[dict]):
        """Run model-written code under this coordinator's sandbox policy."""
        return run_cases(program, cases, strict=self.strict)

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
