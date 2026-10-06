"""Turn a run directory into report.md + summary.json: models, procedure, results, statistics."""

from __future__ import annotations

import json
import math
import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

from .coordinator import PHASES, STRATEGIES, phase_matches

# USD per million tokens (input, output), Anthropic first-party list prices as of 2026-09-25.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cost_usd(provider: str, model: str, tokens_in: int, tokens_out: int) -> float | None:
    if provider != "anthropic":
        return 0.0
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return (tokens_in * p_in + tokens_out * p_out) / 1e6


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarise(run_dir: Path) -> dict:
    outcomes = _read_jsonl(run_dir / "outcomes.jsonl")
    calls = _read_jsonl(run_dir / "calls.jsonl")
    n = len(outcomes)

    per_task_calls = defaultdict(list)
    for c in calls:
        per_task_calls[c["task_id"]].append(c)

    strategies = {}
    for name in STRATEGIES:
        passed = sum(o["answers"][name]["passed"] for o in outcomes)
        used = [c for c in calls if phase_matches(c["phase"], PHASES[name])]
        lo, hi = wilson(passed, n)
        strategies[name] = {
            "passed": passed, "n": n, "pass_rate": passed / n if n else 0.0,
            "ci95": [lo, hi],
            "calls_per_task": len(used) / n if n else 0.0,
            "tokens_per_task": sum(c["input_tokens"] + c["output_tokens"] for c in used) / n if n else 0.0,
        }

    oracle = sum(any(o["candidate_passed"]) for o in outcomes)

    paired = {}
    for name in STRATEGIES:
        if name == "arbitration":
            continue
        b = sum(o["answers"]["arbitration"]["passed"] and not o["answers"][name]["passed"] for o in outcomes)
        c = sum(o["answers"][name]["passed"] and not o["answers"]["arbitration"]["passed"] for o in outcomes)
        paired[name] = {"arbitration_wins": b, "baseline_wins": c, "p_value": mcnemar_exact(b, c)}

    models = defaultdict(lambda: {"roles": set(), "calls": 0, "errors": 0, "input_tokens": 0,
                                  "output_tokens": 0, "latency_s": 0.0})
    for c in calls:
        m = models[(c["provider"], c["model"])]
        m["roles"].add(c["role"])
        m["calls"] += 1
        m["errors"] += c.get("error") is not None
        m["input_tokens"] += c["input_tokens"]
        m["output_tokens"] += c["output_tokens"]
        m["latency_s"] += c["latency_s"]
    model_rows = []
    for (provider, model), m in sorted(models.items()):
        model_rows.append({
            "provider": provider, "model": model, "roles": sorted(m["roles"]),
            "calls": m["calls"], "errors": m["errors"], "input_tokens": m["input_tokens"],
            "output_tokens": m["output_tokens"],
            "avg_latency_s": m["latency_s"] / m["calls"] if m["calls"] else 0.0,
            "cost_usd": cost_usd(provider, model, m["input_tokens"], m["output_tokens"]),
        })

    diag = [o["diagnostics"] for o in outcomes]
    rulings = [r for d in diag for r in d["rulings"]]
    decided = [r for r in rulings if r["choice"] is not None]
    checked = [r for r in decided if r["reference_agrees"] is not None]
    evidence = {
        "tasks_with_disagreement": sum(not d["unanimous"] for d in diag),
        "tasks_with_disputes": sum(d["n_disputes"] > 0 for d in diag),
        "disputed_cases": sum(d["n_disputes"] for d in diag),
        "rulings_decided": len(decided),
        "rulings_unsure": len(rulings) - len(decided),
        "rulings_agreeing_with_reference": sum(r["reference_agrees"] for r in checked),
        "rulings_checked": len(checked),
        "tester_expectations_correct": sum(d["tester_correct"] for d in diag),
        "tester_expectations_judged": sum(d["tester_judged"] for d in diag),
        "spec_example_cases": sum(d["n_spec_cases"] for d in diag),
        "repairs_attempted": sum(d["failing_after_selection"] > 0 for d in diag),
        "repairs_accepted": sum(d["repaired"] for d in diag),
        "unsafe_candidates": sum(d["unsafe_candidates"] for d in diag),
    }
    return {"n_tasks": n, "strategies": strategies, "oracle_passed": oracle, "paired": paired,
            "models": model_rows, "evidence": evidence,
            "per_task": [{"task_id": o["task_id"],
                          **{k: v["passed"] for k, v in o["answers"].items()},
                          "oracle": any(o["candidate_passed"])} for o in outcomes]}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    s = summarise(run_dir)
    with open(run_dir / "config.toml", "rb") as f:
        config = tomllib.load(f)
    run, agents, coord = config["run"], config["agents"], config["coordinator"]
    n = s["n_tasks"]
    k = run.get("candidates", 3)

    lines = [f"# HiveMind run report: `{run['name']}`", "",
             f"Generated {date.today().isoformat()} from `{run_dir.name}/`. "
             f"Benchmark: **HumanEval**, {n} tasks "
             f"(seeded random sample, sample_seed={run.get('sample_seed', 0)}). "
             f"Graded by the benchmark's hidden unit tests, which no agent ever sees.", ""]

    lines += ["## Models used", "",
              "| Provider | Model | Roles | Calls | Failed calls | Input tokens | Output tokens | Avg latency | Cost (USD) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in s["models"]:
        cost = "n/a" if m["cost_usd"] is None else f"${m['cost_usd']:.2f}"
        lines.append(f"| {m['provider']} | `{m['model']}` | {', '.join(m['roles'])} | {m['calls']} | "
                     f"{m['errors']} | {m['input_tokens']:,} | {m['output_tokens']:,} | "
                     f"{m['avg_latency_s']:.1f}s | {cost} |")
    agent_opts = ", ".join(f"{k_}={v}" for k_, v in agents.items() if k_ not in ("provider", "model"))
    coord_opts = ", ".join(f"{k_}={v}" for k_, v in coord.items() if k_ not in ("provider", "model"))
    lines += ["", f"Agent settings: {agent_opts or 'defaults'}. Coordinator settings: {coord_opts or 'defaults'}.",
              "Local (Ollama) models cost nothing per call; Claude costs use list prices.", ""]

    lines += ["## Procedure", "",
              "Every task goes through one shared pipeline, and every strategy below is applied to "
              "the **same** pool of candidates, so differences between strategies come from how "
              "disagreements are resolved:", "",
              "1. **Solo baseline**: one Coding agent writes the function alone.",
              "2. **Research brief**: the Research agent analyses the spec (contract, edge cases, ambiguities).",
              f"3. **Candidates**: {k} Coding agents write solutions with the brief, each given a "
              "different approach hint for diversity.",
              "4. **Team tests**: the Testing agent writes assert-style test cases from the spec alone. "
              "The docstring's own worked examples are added as trusted specification cases.",
              "5. **Execution**: every candidate runs on every case in an isolated, resource-limited "
              "Python process; the Security agent's static checker flags risky code.",
              "6. **Resolution**: each strategy picks (or produces) a final answer:", ""]
    lines += ["| Strategy | What it does | Pipeline phases charged |", "|---|---|---|"]
    for name, text in STRATEGIES.items():
        lines.append(f"| `{name}` | {text} | {', '.join(PHASES[name])} |")
    lines += ["", f"Debate rounds: {run.get('debate_rounds', 1)}. Repair round: "
              f"{'on' if run.get('repair', True) else 'off'}.", ""]

    lines += ["## Results", "",
              "| Strategy | Passed | Pass rate | 95% CI (Wilson) | Model calls / task | Tokens / task |",
              "|---|---:|---:|---|---:|---:|"]
    for name, r in s["strategies"].items():
        lo, hi = r["ci95"]
        lines.append(f"| `{name}` | {r['passed']}/{n} | {_pct(r['pass_rate'])} | "
                     f"{_pct(lo)}–{_pct(hi)} | {r['calls_per_task']:.1f} | {r['tokens_per_task']:,.0f} |")
    lines += ["", f"**Oracle** (at least one of the {k} candidates is correct): "
              f"{s['oracle_passed']}/{n} ({_pct(s['oracle_passed'] / n if n else 0)}). This is the ceiling "
              "for any strategy that only *selects* a candidate; repair and debate can exceed it.", ""]

    lines += ["## Arbitration vs. each baseline (paired, same tasks)", "",
              "| Baseline | Arbitration right, baseline wrong | Baseline right, arbitration wrong | Exact McNemar p |",
              "|---|---:|---:|---:|"]
    for name, p in s["paired"].items():
        lines.append(f"| `{name}` | {p['arbitration_wins']} | {p['baseline_wins']} | {p['p_value']:.3f} |")

    e = s["evidence"]

    def frac(a, b):
        return f"{a}/{b} ({_pct(a / b)})" if b else "0/0"

    lines += ["", "## Evidence diagnostics", "",
              f"- Tasks where candidates behaved differently: {e['tasks_with_disagreement']}/{n}",
              f"- Tasks needing a ruling: {e['tasks_with_disputes']}/{n}; disputed cases: {e['disputed_cases']}",
              f"- Coordinator rulings: {e['rulings_decided']} decided, {e['rulings_unsure']} unsure",
              f"- Rulings that agree with the benchmark's reference solution (analysis only): "
              f"{frac(e['rulings_agreeing_with_reference'], e['rulings_checked'])}",
              f"- Testing agent's expected values that were actually correct: "
              f"{frac(e['tester_expectations_correct'], e['tester_expectations_judged'])}",
              f"- Docstring example cases used as trusted evidence: {e['spec_example_cases']}",
              f"- Repairs attempted: {e['repairs_attempted']}; accepted (verified score improved): "
              f"{e['repairs_accepted']}",
              f"- Candidates flagged by the static security check: {e['unsafe_candidates']}", ""]

    names = list(STRATEGIES)
    lines += ["## Per-task results", "",
              "| Task | " + " | ".join(f"`{x}`" for x in names) + " | oracle |",
              "|---|" + "---|" * (len(names) + 1)]
    for row in s["per_task"]:
        marks = " | ".join("✓" if row[x] else "·" for x in names + ["oracle"])
        lines.append(f"| {row['task_id']} | {marks} |")

    lines += ["", "## How to read this", ""]
    if n < 100:
        lines.append(f"- This is a **pilot** with {n} tasks: the confidence intervals are wide and a "
                     "p-value above 0.05 means the run cannot distinguish the strategies yet. Its job "
                     "is to show the pipeline works end to end and to estimate cost per task.")
    if agents["provider"] != "anthropic":
        lines.append("- Agents ran on a small local model, not Claude. Absolute pass rates are lower "
                     "than Claude's; the headline experiment reruns this config with Claude.")
    lines.append("- Rulings and Testing-agent accuracy are scored against the benchmark's reference "
                 "solution purely for analysis; no agent sees the reference or the hidden tests.")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1))
    return run_dir / "report.md"
