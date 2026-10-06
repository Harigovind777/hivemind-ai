"""The Security agent's static checker: flags code that reaches outside its sandbox.

It runs on every candidate at zero model cost. In review mode the Security agent
also reads the final solution itself (see `Coordinator.review`).
"""

from __future__ import annotations

import ast

RISKY_MODULES = {"os", "subprocess", "socket", "shutil", "ctypes", "multiprocessing",
                 "urllib", "requests", "http", "pickle"}
RISKY_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}


def static_scan(code: str) -> list[str]:
    """Return human-readable findings; an empty list means nothing risky was seen."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # broken code is caught by execution, not flagged as a security issue
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            names = []
        for name in names:
            if name in RISKY_MODULES:
                findings.append(f"imports {name} (line {node.lineno})")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in RISKY_CALLS:
            findings.append(f"calls {node.func.id}() (line {node.lineno})")
    return findings
