"""Chat with the team: turn a plain-English request into a spec, and talk about the answer.

The team itself solves a chat problem with `Coordinator.answer`. This module holds
the three calls only chat needs (draft a spec, read a follow-up as a question or a
change, answer a question) and the check that a spec is something the pipeline can
run: imports, then one function signature whose body is only its docstring.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .agents import Team, Transcript
from .domains.code import CodeTask, doc_example_cases

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_CANDIDATE = re.compile(r"candidate (\d+)")
_READING = re.compile(r"\b(QUESTION|CHANGE)\b")

_SPEC_FORM = (
    "Reply with exactly one ```python block containing, in this order:\n"
    "- any imports the signature needs (for example: from typing import List)\n"
    "- one function signature with type hints\n"
    "- a docstring, indented by 4 spaces, that states the rules precisely, including edge "
    "cases, and ends with 2 to 4 examples in doctest form:\n"
    "    >>> function_name(arguments)\n"
    "    expected result\n"
    "Do not write the function body: the docstring is the last thing in the block. This is "
    "a specification for the team, not an implementation."
)


class SpecError(ValueError):
    """A spec the pipeline cannot run; the message says what to fix and where."""

    def __init__(self, message: str, line: int | None = None):
        super().__init__(message)
        self.line = line


class ModelCallFailed(RuntimeError):
    """A chat model call failed, so the action stops (and can be retried) instead of treating
    the empty reply as an answer."""


_CALLER = {"research": "Research agent", "coordinator": "Coordinator"}


def _ask(team: Team, role: str, prompt: str, transcript: Transcript, phase: str, seed: int) -> str:
    """`Agent.ask` records a failed call as an empty reply; for chat, failing loudly is right."""
    reply = team[role].ask(prompt, transcript=transcript, phase=phase, seed=seed)
    error = transcript.turns[-1].get("error")
    if error:
        raise ModelCallFailed(f"The {_CALLER[role]}'s model call failed: {error}")
    return reply


@dataclass(frozen=True)
class Spec:
    text: str          # imports, signature and docstring, ending with a newline
    entry_point: str
    examples: int      # >>> examples usable as trusted test cases


@dataclass(frozen=True)
class Draft:
    text: str              # what the spec card shows
    spec: Spec | None      # None when the draft is still not a valid spec
    error: str | None


def spec_text(reply: str) -> str:
    """The last ```python block of a reply, or the whole reply when it has none."""
    blocks = _FENCE.findall(reply)
    return (blocks[-1] if blocks else reply).strip("\n")


def _is_stub(node: ast.stmt) -> bool:
    return isinstance(node, ast.Pass) or (isinstance(node, ast.Expr)
                                          and isinstance(node.value, ast.Constant)
                                          and node.value.value is Ellipsis)


def validate_spec(text: str) -> Spec:
    """Check that `text` is imports plus one function whose body is only its docstring.

    A trailing `pass` or `...` is dropped, since the pipeline adds its own stub.
    """
    # Python numbers lines by "\n" only; str.splitlines() would also break on form feeds,
    # U+2028 and friends inside the docstring and cut the spec short.
    text = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise SpecError(f"Line {exc.lineno}: {exc.msg}", exc.lineno) from None
    except UnicodeError:  # e.g. half of an emoji pasted on its own
        raise SpecError("The spec has characters that are not valid text; retype the odd "
                        "characters") from None
    functions: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if functions:
                raise SpecError(f"Line {node.lineno}: put imports before the function", node.lineno)
            continue
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
            continue
        raise SpecError(f"Line {node.lineno}: only imports and one function signature belong "
                        f"in the spec", node.lineno)
    if not functions:
        raise SpecError("The spec needs one function: a signature with a docstring")
    if len(functions) > 1:
        line = functions[1].lineno
        raise SpecError(f"Line {line}: the spec can have only one function", line)
    fn = functions[0]
    if fn.decorator_list:
        line = fn.decorator_list[0].lineno
        raise SpecError(f"Line {line}: remove the decorator", line)
    if ast.get_docstring(fn) is None:
        raise SpecError(f"Line {fn.lineno}: add a docstring that says what {fn.name} must do",
                        fn.lineno)
    extra = [n for n in fn.body[1:] if not _is_stub(n)]
    if extra:
        raise SpecError(f"Line {extra[0].lineno}: remove the function body; the team writes it",
                        extra[0].lineno)
    doc = fn.body[0]
    if doc.col_offset != 4:
        raise SpecError(f"Line {doc.lineno}: indent the docstring by 4 spaces", doc.lineno)
    clean = "\n".join(text.split("\n")[: doc.end_lineno]) + "\n"
    examples = len(doc_example_cases(CodeTask("spec", clean, fn.name, "", "")))
    return Spec(clean, fn.name, examples)


def custom_task(spec: Spec, task_id: str) -> CodeTask:
    """A chat problem as a pipeline task. It has no hidden tests and no reference solution."""
    return CodeTask(task_id, spec.text, spec.entry_point, hidden_test="", canonical_solution="")


def _draft_prompt(message: str, current_spec: str | None) -> str:
    if current_spec is None:
        return (f"A user described a programming problem in plain English. Turn it into a "
                f"precise Python function specification for the team.\n\nThe user's request:\n"
                f"{message}\n\n{_SPEC_FORM}")
    return (f"The team is working from this specification:\n```python\n{current_spec.rstrip()}\n"
            f"```\n\nThe user asked for this change:\n{message}\n\nWrite the complete updated "
            f"specification. Keep everything the user did not ask to change.\n\n{_SPEC_FORM}")


def draft_spec(team: Team, message: str, transcript: Transcript, current_spec: str | None = None,
               seed: int = 0) -> Draft:
    """The Research agent writes the spec, with one retry when it is not valid."""
    prompt = _draft_prompt(message, current_spec)
    text, error = "", None
    for attempt in range(2):
        if error:
            prompt += (f"\n\nYour previous reply could not be used ({error}). Reply again with "
                       f"one ```python block in exactly that form.")
        text = spec_text(_ask(team, "research", prompt, transcript, "draft", seed + attempt))
        try:
            spec = validate_spec(text)
        except SpecError as exc:
            error = str(exc)
            continue
        return Draft(spec.text, spec, None)
    return Draft(text, None, error)


def classify(team: Team, message: str, context: str, transcript: Transcript, seed: int = 0) -> str:
    """Is a follow-up a QUESTION about the answer or a CHANGE to the problem? Unclear means
    question: answering is cheap and nothing re-runs."""
    reply = _ask(
        team, "coordinator",
        f"The team solved the problem below. Then the user sent a new message. Decide whether "
        f"the user is asking a QUESTION (about the problem, the answer or how the team got there) "
        f"or asking for a CHANGE to the problem (different behaviour, a new requirement, other "
        f"inputs or outputs).\n\n{context}\n\nThe user's new message:\n{message}\n\n"
        f"Reply with one word: QUESTION or CHANGE.",
        transcript, "classify", seed)
    found = _READING.findall(reply)
    return "change" if found and found[-1] == "CHANGE" else "question"


def answer_question(team: Team, message: str, context: str, history: list[tuple[str, str]],
                    transcript: Transcript, seed: int = 0) -> str:
    """The Coordinator answers a question from the round's evidence."""
    earlier = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in history)
    prompt = (f"The user gave the team a problem and the team answered it. Answer the user's "
              f"question using only the evidence below. Be brief and concrete; quote cases and "
              f"outputs where they help.\n\n{context}\n\n"
              + (f"Earlier questions about this answer:\n{earlier}" if history else "")
              + f"The user's question:\n{message}")
    return _ask(team, "coordinator", prompt, transcript, "question", seed)


def display_source(source: str) -> str:
    """Where an answer came from, counting candidates from 1 as the page does
    ("candidate 0" is shown as "candidate 1")."""
    return _CANDIDATE.sub(lambda m: f"candidate {int(m.group(1)) + 1}", source)


def round_context(spec: str, answer: dict) -> str:
    """What the Coordinator knows about a finished round, as prompt text."""
    how = display_source(answer["source"]) + (" (kept because it passes more verified cases)"
                              if answer["repaired"] else "")
    parts = [
        f"Specification:\n```python\n{spec.rstrip()}\n```",
        f"The team's final answer, from {how}:\n```python\n{answer['code'].rstrip()}\n```",
        f"Evidence: {len(answer['candidates'])} candidates showed {answer['distinct_behaviours']} "
        f"different behaviours. The final answer passes {answer['verified_passed']} of "
        f"{answer['verified_total']} verified cases ({answer['spec_cases']} from the "
        f"specification's examples, {answer['ruled_cases']} ruled on by the Coordinator, "
        f"{answer['undisputed_cases']} that no one disputed).",
    ]
    if answer["failures"]:
        parts.append("Verified cases it still fails:\n" + "\n".join(
            f"- {f['call']}: expected {f['expected']}, got {f['got']}" for f in answer["failures"]))
    if answer["rulings"]:
        parts.append("The Coordinator's rulings on disputed cases:\n" + "\n".join(
            f"- {r['call']}: options {', '.join(r['options'])}; chose "
            f"{r['choice'] if r['choice'] is not None else 'UNSURE'}" for r in answer["rulings"]))
    return "\n\n".join(parts)
