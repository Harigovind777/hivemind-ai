"""Software-engineering domain: write a Python function from its signature and docstring.

The benchmark's own unit tests are hidden from every agent and used only for the
final grade. Agents work from the specification; the coordinator gathers its
evidence by executing candidates on test cases the team writes itself.
"""

from __future__ import annotations

import ast
import doctest
import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ..sandbox import run_program

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str              # signature + docstring shown to the agents
    entry_point: str
    hidden_test: str         # benchmark tests, used only for grading
    canonical_solution: str  # reference body, used only for post-hoc analysis

    @property
    def spec(self) -> str:
        return self.prompt.strip()


def load_humaneval(n: int | None = None, seed: int = 0,
                   task_ids: list[str] | None = None) -> list[CodeTask]:
    """Load HumanEval; take `task_ids`, or a seeded random sample of `n`, or all 164."""
    with gzip.open(DATA_DIR / "HumanEval.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    tasks = [CodeTask(r["task_id"], r["prompt"], r["entry_point"], r["test"],
                      r["canonical_solution"]) for r in rows]
    if task_ids:
        wanted = set(task_ids)
        return [t for t in tasks if t.task_id in wanted]
    if n is not None and n < len(tasks):
        tasks = sorted(random.Random(seed).sample(tasks, n),
                       key=lambda t: int(t.task_id.split("/")[1]))
    return tasks


# ---------------------------------------------------------------- program assembly

def build_program(task: CodeTask, code: str) -> str:
    """Prompt first (imports, helpers, stub), then the candidate, which overrides the stub."""
    return f"{task.prompt}\n    pass\n\n{code}\n"


def reference_program(task: CodeTask) -> str:
    return task.prompt + task.canonical_solution


def grade(task: CodeTask, code: str, *, timeout_s: float = 15.0) -> bool:
    """Ground truth: does the candidate pass the benchmark's hidden tests?"""
    return grade_detailed(task, code, timeout_s=timeout_s)[0]


def grade_detailed(task: CodeTask, code: str, *, timeout_s: float = 15.0,
                   strict: bool = False) -> tuple[bool, str]:
    """Grade, and say why. A timeout is not a wrong answer and must be distinguishable.

    Returns (passed, reason). Reasons worth counting separately in a report:
      "empty", "timeout", "syntax", "load", "assertion", "exception", "ok".
    """
    if not code.strip():
        return False, "empty"
    source = f"{build_program(task, code)}\n{task.hidden_test}\ncheck({task.entry_point})\n"
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax: {exc.msg}"
    ok, err = run_program(source, timeout_s=timeout_s, strict=strict)
    if ok:
        return True, "ok"
    if "Timeout" in err:
        return False, "timeout"
    for marker, reason in (("AssertionError", "assertion"), ("NameError", "load"),
                           ("ImportError", "load"), ("IndentationError", "load"),
                           ("TypeError", "exception"), ("ValueError", "exception"),
                           ("RecursionError", "exception"), ("ZeroDivisionError", "exception"),
                           ("MemoryError", "exception")):
        if marker in err:
            return False, reason
    return False, "load" if "Error" in err else "exception"


# ---------------------------------------------------------------- parsing model output

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, entry_point: str) -> str:
    """The code block that defines `entry_point` (falling back to the last block)."""
    blocks = _FENCE.findall(text)
    if not blocks:
        # Unterminated fence or bare code.
        tail = text.split("```python", 1)[-1] if "```python" in text else text
        blocks = [tail.replace("```", "")]
    for block in reversed(blocks):
        if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", block):
            return sanitize(block.strip())
    return sanitize(blocks[-1].strip())


def sanitize(code: str) -> str:
    """Keep imports, definitions and constants; drop top-level self-tests, prints and main blocks.

    Models often append their own asserts or demo calls after the function; a wrong
    self-test would otherwise crash an implementation that is itself correct.

    Module-level *setup* is deliberately kept: a `try/except` import fallback, a
    version-gated `if`, or a table built in a loop is part of a working solution, and
    dropping it turns a correct candidate into a `NameError` that is then scored as
    wrong. Only statements that merely *exercise* the solution are removed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def calls_own_function(node: ast.AST) -> bool:
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in defined
                   for c in ast.walk(node))

    def is_main_guard(node: ast.AST) -> bool:
        # `if __name__ == "__main__":` and the demo body it guards.
        test = node.test if isinstance(node, ast.If) else None
        return isinstance(test, ast.Compare) and any(
            isinstance(c, ast.Name) and c.id == "__name__" for c in ast.walk(test))

    def keep(node: ast.AST) -> bool:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            return not calls_own_function(node)
        if isinstance(node, ast.If):
            return not is_main_guard(node)  # version-gated setup is real logic
        if isinstance(node, (ast.Try, ast.For, ast.While, ast.With)):
            # Keep it if it sets something up; drop it if it only runs the solution.
            return not _is_self_test_body(node, defined)
        return False

    kept = [n for n in tree.body if keep(n)]
    if not kept:
        return code
    return ast.unparse(ast.Module(body=kept, type_ignores=[]))


def _is_self_test_body(node: ast.AST, defined: set[str]) -> bool:
    """True when a module-level control-flow statement exists only to exercise the solution."""
    body = getattr(node, "body", None) or getattr(node, "finalbody", None) or []
    if not body:
        return False
    for stmt in body:
        # An assert, print or bare call to the solution = a self-test. An import or
        # assignment is setup and must be kept.
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
                             ast.AugAssign, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Pass)):
            continue
        if isinstance(stmt, ast.Assert):
            return True
        if isinstance(stmt, ast.Expr):
            # `TABLE.append(x)` is setup; `print(f(1))` and `f(1)` are self-tests.
            return _calls_any(stmt, defined)
    return False


def _calls_any(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in names
               for c in ast.walk(node))


def _is_target_call(node: ast.AST, entry_point: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == entry_point)


def parse_assert_cases(text: str, entry_point: str, limit: int = 12) -> list[dict]:
    """Turn `assert f(x) == y` style lines into {"call", "expected"} cases."""
    cases: list[dict] = []
    source = "\n".join(_FENCE.findall(text)) or text
    for line in source.splitlines():
        line = line.strip()
        if not line.startswith("assert "):
            continue
        try:
            node = ast.parse(line).body[0]
        except SyntaxError:
            continue
        test = node.test
        call = expected = None
        if isinstance(test, ast.Compare) and len(test.ops) == 1 \
                and isinstance(test.ops[0], (ast.Eq, ast.Is)):
            left, right = test.left, test.comparators[0]
            if _is_target_call(left, entry_point):
                call, expected = left, right
            elif _is_target_call(right, entry_point):
                call, expected = right, left
        elif _is_target_call(test, entry_point):
            call, expected = test, ast.Constant(True)
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) \
                and _is_target_call(test.operand, entry_point):
            call, expected = test.operand, ast.Constant(False)
        if call is not None:
            case = {"call": ast.unparse(call), "expected": ast.unparse(expected)}
            if case not in cases:
                cases.append(case)
        if len(cases) >= limit:
            break
    return cases


def doc_example_cases(task: CodeTask) -> list[dict]:
    """Worked examples from the docstring: the specification's own evidence."""
    cases = []
    try:
        examples = doctest.DocTestParser().get_examples(task.prompt)
    except ValueError:
        return cases
    for ex in examples:
        # doctest counts the docstring's closing quotes as part of the last example's output.
        want = ex.want.split('"""')[0].split("'''")[0].strip()
        call = ex.source.strip()
        if not want or task.entry_point not in call:
            continue
        try:
            ast.parse(want, mode="eval")
            node = ast.parse(call, mode="eval").body
        except SyntaxError:
            continue
        if _is_target_call(node, task.entry_point):
            cases.append({"call": call, "expected": want})
    return cases
