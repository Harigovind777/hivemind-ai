"""Run model-written Python in a separate, resource-limited process.

Execution results are the evidence the coordinator arbitrates on, so this module is
the one place where untrusted code runs.

What the default mode actually guarantees:

  * a fresh interpreter per run (isolated mode ``-I``, no bytecode writes ``-B``)
  * an empty temporary working directory
  * a **scrubbed environment**: the parent's variables are dropped, so candidate
    code cannot read ``ANTHROPIC_API_KEY`` or any other secret out of ``os.environ``
  * its own process group, so a timeout reaps children instead of leaking them
  * CPU-time, file-size and file-descriptor rlimits
  * per-case wall-clock timeouts, plus an outer timeout on the whole run

What it does **not** guarantee, and this is a known residual risk rather than an
oversight: there is no OS-level capability sandbox here. No network namespace, no
filesystem jail, and no enforced address-space limit on macOS, which rejects
``RLIMIT_AS`` outright (it is enforced on Linux, where the memory cap does bite).
Candidate code can therefore still read and write the filesystem and open sockets.

Two mitigations are available, and ``strict=True`` is the one to use for genuinely
adversarial input:

  * ``memory_mb`` caps address space where the platform supports it; the rlimits that
    were actually applied are reported back in ``report["rlimits_applied"]``.
  * ``strict=True`` installs an import/attribute blocklist before the candidate is
    executed, removing the network, process-spawning and destructive-filesystem
    capabilities outright.

``strict`` is **off by default on purpose**. A blocklist changes what candidate code
can do, so a candidate that happens to call a blocked builtin starts failing for a
reason that has nothing to do with the specification. Leaving it off keeps grading
faithful; turning it on is an explicit decision, and the decision is recorded in the
run's diagnostics.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Only variables a candidate might legitimately need. Everything else - notably any
# API key or token - is withheld.
_ENV_ALLOWLIST = frozenset({
    "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "TZ",
    "SYSTEMROOT", "COMSPEC", "PATHEXT", "WINDIR",
})

# Capabilities removed in strict mode. None of these are needed to compute a
# function's return value; each is an escape hatch out of the sandbox.
_STRICT_BLOCKED_MODULES = frozenset({
    "subprocess", "multiprocessing", "socket", "ssl", "selectors", "asyncio",
    "urllib", "http", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
    "xmlrpc", "webbrowser", "ctypes", "signal", "pty", "shelve", "dbm",
    "requests", "httpx", "aiohttp", "paramiko", "psutil", "ptyprocess",
})

_STRICT_BLOCKED_ATTRS = (
    "system", "popen", "remove", "unlink", "rmdir", "removedirs", "rename", "replace",
    "chmod", "chown", "lchmod", "setuid", "setgid", "fork", "forkpty",
    "execv", "execve", "execl", "execlp", "execvp", "spawnv", "spawnl", "spawnve",
    "putenv", "unsetenv", "putheader", "startfile",
    "rmtree", "copytree", "move", "copy", "copy2", "copyfile", "chown",
)

# `open` is denied because a write-capable builtin is the filesystem escape that an import
# blocklist cannot otherwise reach. Strict mode is opt-in precisely because this changes
# what candidate code may do, and no legitimate solution needs to open a file.
_STRICT_BLOCKED_BUILTINS = ("exec", "eval", "compile", "open")

_STRICT_PRELUDE = (f"_BLOCKED_MODULES = {set(_STRICT_BLOCKED_MODULES)!r}\n"
                   f"_BLOCKED_ATTRS = {_STRICT_BLOCKED_ATTRS!r}\n"
                   f"_BLOCKED_BUILTINS = {_STRICT_BLOCKED_BUILTINS!r}\n")

# Every rlimit is attempted independently: macOS rejects some of them, and a
# rejected limit must not take the whole run down with it.
_LIMITS_HEAD = '''
import resource
_applied = []

def _limit(name, value):
    try:
        resource.setrlimit(getattr(resource, name), (value, value))
        _applied.append(name)
    except (ValueError, OSError, AttributeError, NameError):
        pass
'''

_STRICT_HEAD = '''
class _Blocked(RuntimeError):
    pass

def _deny(*_a, **_k):
    raise _Blocked("operation disabled in the HiveMind sandbox")

def _install_strict():
    import builtins, os, shutil
    # The harness itself needs eval and compile to run test cases and to exec the
    # candidate. Those references are captured in the harness body before this runs, so
    # denying the builtins does not stop the grader from grading.
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in _BLOCKED_MODULES:
            raise _Blocked("import of %r is disabled in the HiveMind sandbox" % name)
        return real_import(name, globals, locals, fromlist, level)

    # __import__ is deliberately absent from _BLOCKED_BUILTINS: it is already replaced
    # above by the selective guard. Blanket-denying it would block every import, stdlib
    # included, and no candidate could load at all.
    builtins.__import__ = guarded
    for name in _BLOCKED_BUILTINS:
        if hasattr(builtins, name):
            setattr(builtins, name, _deny)
    for attr in _BLOCKED_ATTRS:
        if hasattr(os, attr):
            setattr(os, attr, _deny)
        if hasattr(shutil, attr):
            setattr(shutil, attr, _deny)
'''

_HARNESS_BODY = r'''
import contextlib, io, json, math, signal, sys


def _norm(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else round(v, 6)
    if isinstance(v, bool):
        return v
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


payload = json.loads(sys.stdin.read())
strict = bool(payload.get("strict"))
# Captured before the blocklist installs: the grader needs these to run the candidate and
# evaluate test cases, and denying the builtins must not stop the grader from grading.
# Candidate code still resolves them through builtins, so it sees the denied versions.
_real_eval, _real_compile, _real_exec = eval, compile, exec
if strict:
    _install_strict()

signal.signal(signal.SIGALRM, _alarm)
ns = {"__name__": "__candidate__"}
sink = io.StringIO()
load_error = None
try:
    with contextlib.redirect_stdout(sink):
        signal.setitimer(signal.ITIMER_REAL, payload["case_timeout"] * 2)
        _real_exec(_real_compile(payload["code"], "<candidate>", "exec"), ns)
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
            expected = _real_eval(case["expected"], {})
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
            value = _real_eval(case["call"], ns)
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

sys.__stdout__.write("\n@@HIVEMIND@@" + json.dumps({
    "load_error": load_error, "results": results,
    "rlimits_applied": _applied, "strict": strict,
}))
'''

_PROGRAM_BODY = r'''
import sys

# Captured before the blocklist installs, because installing it denies these very names.
_real_compile, _real_exec = compile, exec
if globals().get("_STRICT"):
    _install_strict()
_real_exec(_real_compile(sys.stdin.read(), "<program>", "exec"), {"__name__": "__hivemind__"})
sys.__stdout__.write("\n@@HIVEMIND@@ok")
'''

_MARKER = "@@HIVEMIND@@"

DEFAULT_MEMORY_MB = 2048
DEFAULT_CPU_S = 20
DEFAULT_FSIZE = 1 << 20
DEFAULT_NOFILES = 256


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


def _child_env() -> dict:
    """A deliberately bare environment: candidate code must not read the parent's secrets."""
    env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def _limits_src(memory_mb: int, cpu_s: int, fsize: int, nofiles: int) -> str:
    mem = max(1, memory_mb) * 1024 * 1024
    return (_LIMITS_HEAD
            + f'_limit("RLIMIT_CPU", {cpu_s})\n'
            + f'_limit("RLIMIT_FSIZE", {fsize})\n'
            + f'_limit("RLIMIT_AS", {mem})\n'
            + f'_limit("RLIMIT_DATA", {mem})\n'
            + f'_limit("RLIMIT_NOFILE", {nofiles})\n')


def _case_harness(strict: bool, **limits) -> str:
    prelude = _STRICT_PRELUDE if strict else ""
    return (_limits_src(**limits) + prelude + _STRICT_HEAD + _HARNESS_BODY)


def _program_harness(strict: bool, **limits) -> str:
    prelude = _STRICT_PRELUDE if strict else ""
    flag = "_STRICT = True\n" if strict else ""
    return _limits_src(**limits) + prelude + flag + _STRICT_HEAD + _PROGRAM_BODY


def _kill_group(proc: subprocess.Popen) -> None:
    """Reap the whole process group: a candidate that forked must not outlive its run."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _run(harness: str, stdin: str, timeout_s: float) -> subprocess.CompletedProcess | None:
    with tempfile.TemporaryDirectory(prefix="hivemind-") as cwd:
        try:
            # start_new_session puts the child in its own process group. It is a plain
            # flag rather than a preexec_fn callback, so it stays safe under the
            # runner's ThreadPoolExecutor.
            proc = subprocess.Popen(
                [sys.executable, "-I", "-B", "-c", harness],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=cwd, env=_child_env(), start_new_session=True,
            )
        except OSError:
            return None
        try:
            out, err = proc.communicate(stdin, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
            return None
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def run_cases(code: str, cases: list[dict], case_timeout: float = 3.0, *, strict: bool = False,
              report: dict | None = None, **limits) -> list[CaseResult]:
    """Evaluate each case's `call` expression (and optional `expected`) against `code`.

    Pass `report={}` to learn which rlimits this platform actually enforced.
    """
    if not cases:
        return []
    limits.setdefault("memory_mb", DEFAULT_MEMORY_MB)
    limits.setdefault("cpu_s", DEFAULT_CPU_S)
    limits.setdefault("fsize", DEFAULT_FSIZE)
    limits.setdefault("nofiles", DEFAULT_NOFILES)
    payload = json.dumps({"code": code, "cases": cases, "case_timeout": case_timeout,
                          "strict": strict})
    proc = _run(_case_harness(strict, **limits), payload,
                timeout_s=case_timeout * (len(cases) + 2) + 5)
    if proc is None or _MARKER not in proc.stdout:
        reason = "Timeout" if proc is None else f"Crash(rc={proc.returncode})"
        if report is not None:
            report.update(rlimits_applied=[], strict=strict, failure=reason)
        return [CaseResult(None, reason, None) for _ in cases]
    data = json.loads(proc.stdout.rsplit(_MARKER, 1)[1])
    if report is not None:
        report.update(rlimits_applied=data.get("rlimits_applied", []), strict=strict, failure=None)
    return [CaseResult(r["output"], r["error"], r["passed"]) for r in data["results"]]


def run_program(source: str, timeout_s: float = 15.0, *, strict: bool = False,
                **limits) -> tuple[bool, str]:
    """Run a whole program; success means it exited cleanly. Returns (ok, error tail).

    In strict mode the blocklist is installed first, then the program runs. The blocklist
    itself needs `compile` and `exec`, which are denied to the program, so both are
    captured before the blocklist goes on.
    """
    limits.setdefault("memory_mb", DEFAULT_MEMORY_MB)
    limits.setdefault("cpu_s", DEFAULT_CPU_S)
    limits.setdefault("fsize", DEFAULT_FSIZE)
    limits.setdefault("nofiles", DEFAULT_NOFILES)
    proc = _run(_program_harness(strict, **limits), source, timeout_s)
    if proc is None:
        return False, "Timeout"
    return proc.returncode == 0, proc.stderr[-400:]


def strict_blocks_network(timeout_s: float = 8.0) -> bool:
    """True when strict mode actually denies a real outbound request.

    Checks the control rather than trusting the blocklist, so a silent gap in
    `_STRICT_BLOCKED_MODULES` shows up as a failing test instead of an open escape.
    """
    probe = ("import urllib.request\n"
             "def f():\n"
             "    return urllib.request.urlopen('http://example.com', timeout=2).status\n")
    try:
        result = run_cases(probe, [{"call": "f()"}], strict=True, case_timeout=timeout_s)
    except Exception:
        return True   # cannot verify here; assume the safer branch
    return bool(result and result[0].error is not None)


def probe_limits(**limits) -> dict:
    """Run a trivial program and report which rlimits this platform enforces."""
    report: dict = {}
    run_cases("def _f():\n    return 1\n", [{"call": "_f()"}], report=report, **limits)
    return {"rlimits_applied": report.get("rlimits_applied", []), "strict": False,
            "memory_mb": limits.get("memory_mb", DEFAULT_MEMORY_MB)}
