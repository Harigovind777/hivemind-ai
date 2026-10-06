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


def grade(task: CodeTask, code: str) -> bool:
    """Ground truth: does the candidate pass the benchmark's hidden tests?"""
    if not code.strip():
        return False
    source = f"{build_program(task, code)}\n{task.hidden_test}\ncheck({task.entry_point})\n"
    ok, _ = run_program(source)
    return ok


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
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def calls_own_function(node: ast.AST) -> bool:
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in defined
                   for c in ast.walk(node))

    kept = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef))
            or (isinstance(n, (ast.Assign, ast.AnnAssign)) and not calls_own_function(n))]
    if not kept:
        return code
    return ast.unparse(ast.Module(body=kept, type_ignores=[]))


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
