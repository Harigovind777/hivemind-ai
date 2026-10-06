"""Run model-written Python in a separate, resource-limited process.

Execution results are the evidence the coordinator arbitrates on, so this module
is the one place where untrusted code runs. Each run gets a fresh interpreter
(isolated mode, no bytecode writes), an empty temporary working directory, CPU
and file-size limits, and per-case wall-clock timeouts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_LIMITS = r"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
"""

_CASE_HARNESS = _LIMITS + r"""
import contextlib, io, json, math, signal, sys

def _norm(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else round(v, 6)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

payload = json.loads(sys.stdin.read())
signal.signal(signal.SIGALRM, _alarm)
ns = {"__name__": "__candidate__"}
sink = io.StringIO()
load_error = None
try:
    with contextlib.redirect_stdout(sink):
        signal.setitimer(signal.ITIMER_REAL, payload["case_timeout"] * 2)
        exec(compile(payload["code"], "<candidate>", "exec"), ns)
except BaseException as e:
    load_error = (type(e).__name__ + ": " + str(e))[:200]
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)

results = []
for case in payload["cases"]:
    out = {"output": None, "error": None, "passed": None}
    has_expected = False
    if case.get("expected") is not None:
        try:
            expected = eval(case["expected"], {})
            has_expected = True
        except BaseException:
            pass  # malformed expectation: the case cannot be judged
    if load_error is not None:
        out["error"] = "load: " + load_error
        out["passed"] = False if has_expected else None
        results.append(out)
        continue
    try:
        with contextlib.redirect_stdout(sink):
            signal.setitimer(signal.ITIMER_REAL, payload["case_timeout"])
            value = eval(case["call"], ns)
            signal.setitimer(signal.ITIMER_REAL, 0)
        out["output"] = repr(_norm(value))[:500]
        if has_expected:
            out["passed"] = _norm(value) == _norm(expected)
    except _Timeout:
        out["error"] = "Timeout"
    except BaseException as e:
        out["error"] = type(e).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if out["error"] is not None and has_expected:
        out["passed"] = False
    results.append(out)

sys.__stdout__.write("\n@@HIVEMIND@@" + json.dumps({"load_error": load_error, "results": results}))
"""

_PROGRAM_HARNESS = _LIMITS + r"""
import sys
exec(compile(sys.stdin.read(), "<program>", "exec"), {"__name__": "__hivemind__"})
"""

_MARKER = "@@HIVEMIND@@"


@dataclass(frozen=True)
class CaseResult:
    output: str | None   # repr of the (normalised) return value
    error: str | None    # exception name, "Timeout", or a load error
    passed: bool | None  # None when the case had no usable expected value

    @property
    def signature(self) -> str:
        """What the call observably did; used to compare candidates."""
        return self.output if self.error is None else f"<{self.error}>"

    @property
    def is_answer(self) -> bool:
        """False when the program never produced a result (failed to load, hung, crashed).

        Raising an exception still counts as an answer: a spec may require it.
        """
        return self.error is None or not (self.error.startswith(("load:", "Crash"))
                                          or self.error == "Timeout")


def _run(harness: str, stdin: str, timeout_s: float) -> subprocess.CompletedProcess | None:
    with tempfile.TemporaryDirectory(prefix="hivemind-") as cwd:
        try:
            return subprocess.run([sys.executable, "-I", "-B", "-c", harness], input=stdin,
                                  capture_output=True, text=True, timeout=timeout_s, cwd=cwd)
        except subprocess.TimeoutExpired:
            return None


def run_cases(code: str, cases: list[dict], case_timeout: float = 3.0) -> list[CaseResult]:
    """Evaluate each case's `call` expression (and optional `expected`) against `code`."""
    if not cases:
        return []
    payload = json.dumps({"code": code, "cases": cases, "case_timeout": case_timeout})
    proc = _run(_CASE_HARNESS, payload, timeout_s=case_timeout * (len(cases) + 2) + 5)
    if proc is None or _MARKER not in proc.stdout:
        reason = "Timeout" if proc is None else f"Crash(rc={proc.returncode})"
        return [CaseResult(None, reason, None) for _ in cases]
    data = json.loads(proc.stdout.rsplit(_MARKER, 1)[1])
    return [CaseResult(r["output"], r["error"], r["passed"]) for r in data["results"]]


def run_program(source: str, timeout_s: float = 15.0) -> tuple[bool, str]:
    """Run a whole program; success means it exited cleanly. Returns (ok, error tail)."""
    proc = _run(_PROGRAM_HARNESS, source, timeout_s)
    if proc is None:
        return False, "Timeout"
    return proc.returncode == 0, proc.stderr[-400:]
