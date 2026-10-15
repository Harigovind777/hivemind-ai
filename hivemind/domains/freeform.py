"""Free-form domain: run the same team on a prompt the user typed in.

HumanEval gives every task a hidden test suite, which makes its pass rate ground truth.
A prompt typed into a web form has no such suite, so this module refuses to pretend
otherwise. There is no `grade()` here and no reference solution; a candidate is only ever
measured against cases the team wrote itself, and every number carries that label.

The consequence is worth stating plainly: a "win" in this domain means *this team believed
its own verified cases were satisfied*. That is the same evidence the Coordinator arbitrates
on, so it is a fair way to compare the seven strategies against each other on the same
task, and it is not comparable to the HumanEval pass rates in `report.md`.
"""

from __future__ import annotations

import ast
import re

from .code import CodeTask, build_program, extract_code

_DEF = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.M)


def _is_prose(prompt: str) -> bool:
    """True when the prompt describes a function in words instead of declaring one."""
    return not _DEF.search(prompt) and "class " not in prompt


def detect_entry_point(prompt: str) -> str | None:
    """The function a prompt is asking for, if it names exactly one."""
    names = _DEF.findall(prompt)
    if not names:
        return None
    # The last definition is the one the prompt is usually about; nested helpers defined
    # above it are setup. A single name is unambiguous.
    return names[-1]


def make_task(prompt: str, entry_point: str | None = None, task_id: str = "user/0") -> CodeTask:
    """Wrap a user prompt as a task the existing pipeline can consume.

    `hidden_test` and `canonical_solution` stay empty on purpose. Nothing in the
    Coordinator grades against them for this domain; see `evidence_grade`.
    """
    text = prompt.strip()
    entry = entry_point or detect_entry_point(text)
    if not entry:
        raise ValueError(
            "Could not find a function to implement. Include a `def name(...)` line in the "
            "prompt, or set the entry point explicitly.")
    # A prompt written entirely in prose has no `def` to match, so the entry point can only
    # come from the caller. Requiring one anyway would make the prose path unreachable. When
    # the prompt does carry definitions, the entry point must still be one of them, so a typo
    # in the field is still an error instead of a silently unimplemented task.
    if not _is_prose(text) and not re.search(rf"def\s+{re.escape(entry)}\s*\(", text):
        raise ValueError(f"The prompt never defines `{entry}`.")
    return CodeTask(task_id=task_id, prompt=text, entry_point=entry,
                    hidden_test="", canonical_solution="")


def looks_like_a_prompt(prompt: str, entry_point: str = "") -> str | None:
    """A short complaint about a prompt the Coordinator cannot act on."""
    if not prompt.strip():
        return "Enter a prompt."
    if len(prompt.strip()) < 5:
        return "That prompt is too short to contain a specification."
    # Prose plus a named entry point is a complete specification: the agents are told what to
    # build by name and what it must do in words. Only complain when neither is available.
    if _is_prose(prompt) and not entry_point.strip():
        return ("No function definition found. Start with `def name(args):` and a docstring, "
                "or describe the interface in prose and set the entry point.")
    if not _is_prose(prompt) and detect_entry_point(prompt) is None and "class " not in prompt:
        return "No function definition found in the prompt."
    return None


def evidence_grade(code: str, task: CodeTask, cases: list[dict],
                   verified: dict[int, tuple]) -> dict:
    """How well `code` satisfies the team's own verified cases.

    This is NOT ground truth and is never called `passed` on its own. Three numbers come
    back, and the caller is expected to show all of them:

      verified_satisfied / verified_total
          cases the Coordinator actually ruled on. Cases it could not rule on are excluded
          from the denominator rather than counted as failures, so a formatting problem
          does not look like a wrong answer.
      stated_passed / stated_total
          agreement with the Testing agent's written expectations, which the design
          explicitly does not trust. Shown so the gap is visible.
      unverified
          cases that were disputed but never ruled on.
    """
    from ..sandbox import run_cases

    if not cases or not code.strip():
        return {"verified_satisfied": 0, "verified_total": 0, "stated_passed": 0,
                "stated_total": len(cases), "unverified": 0, "error": "empty solution"}

    results = run_cases(build_program(task, code),
                        [{"call": c["call"], "expected": c.get("expected")} for c in cases])

    satisfied = sum(1 for j, rule in verified.items()
                    if j < len(results) and _matches(rule, results[j]))
    unverified = sum(1 for j in range(len(cases)) if j not in verified)
    stated = sum(1 for r in results if r.passed is True)
    return {
        "verified_satisfied": satisfied,
        "verified_total": len(verified),
        "stated_passed": stated,
        "stated_total": sum(1 for c in cases if c.get("expected") is not None),
        "unverified": unverified,
        "error": None,
    }


def _matches(rule: tuple, result) -> bool:
    """Same satisfaction test the Coordinator uses, kept here to avoid a circular import."""
    if not rule or not result:
        return False
    if rule[0] == "pass":
        return result.passed is True
    return result.signature == rule[1]


def syntax_check(code: str) -> str | None:
    """A parse error the user can see immediately, rather than as a failed run."""
    if not code.strip():
        return "empty solution"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    return None


def extract(code_text: str, entry_point: str) -> str:
    """Public alias so the web layer does not import the private benchmark module."""
    return extract_code(code_text, entry_point)
