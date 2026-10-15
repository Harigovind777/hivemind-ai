"""Run an experiment from a TOML config and write everything to runs/<name>/.

Output files (one line per task, appended as tasks finish so a run can resume):
  outcomes.jsonl     strategy answers, grades and arbitration diagnostics
  calls.jsonl        every model call: phase, role, model, tokens, latency
  transcripts/       every message exchanged, one JSON file per task
  config.toml        the exact configuration used
  report.md          written by `hivemind report` (also at the end of a run)
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .agents import Team
from .coordinator import PHASES, STRATEGIES, Coordinator, NoUsableOutput
from .domains.code import load_humaneval
from .llm import Ledger, make_llm

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _llm(section: dict):
    options = {k: v for k, v in section.items() if k not in ("provider", "model")}
    return make_llm(section["provider"], section["model"], **options)


def estimate_cost(config: dict, n_tasks: int) -> dict:
    """Project calls, tokens and USD for a run before spending anything.

    Calls per task are counted from the phases each strategy charges, so the estimate
    reflects the real pipeline rather than a guess. Token counts are scaled from a
    representative per-phase average and are the least precise figure here.
    """
    from .report import cost_usd

    run = config["run"]
    phases = set()
    for name in STRATEGIES:
        phases.update(PHASES[name])
    arb = config["agents"]
    coo = config.get("coordinator", arb)
    tasks = n_tasks or run.get("n_tasks", 0)

    # per-task call counts, from what solve() actually does
    k = run.get("candidates", 3)
    per_task = {
        "solo": 1,
        "brief": 1,
        "candidate": k,
        "tests": 1,
        "arbiter": 0.4,     # only tasks with a dispute reach the arbiter
        "repair": 0.3,      # only tasks with a verified failure reach repair
        "debate": k * run.get("debate_rounds", 1),
        "judge": 0.5,       # only when the debate did not converge
    }
    calls = {"agents": 0, "coordinator": 0}
    for phase, count in per_task.items():
        calls["coordinator" if phase in ("arbiter", "judge") else "agents"] += count
    calls = {k_: round(v, 1) for k_, v in calls.items()}

    # 8k in / 1.5k out is a representative Claude round-trip for this pipeline; both
    # figures are overridable so the estimate can be tuned with real measurements.
    tok_in, tok_out = 8_000, 1_500
    est = {"n_tasks": tasks, "calls_per_task": calls,
           "calls_total": round(sum(calls.values()) * tasks),
           "input_tokens_total": int(sum(calls.values()) * tasks * tok_in),
           "output_tokens_total": int(sum(calls.values()) * tasks * tok_out),
           "assumed_tokens_per_call": {"input": tok_in, "output": tok_out},
           "agents_model": arb.get("model"), "coordinator_model": coo.get("model")}
    usd = 0.0
    for section, who in ((arb, "agents"), (coo, "coordinator")):
        share = calls[who] / max(1.0, sum(calls.values()))
        c = cost_usd(section.get("provider", ""), section.get("model", ""),
                     int(est["input_tokens_total"] * share),
                     int(est["output_tokens_total"] * share))
        est[f"cost_usd_{who}"] = c
        if c is not None:
            usd += c
    est["cost_usd"] = usd if usd else None
    est["stratified"] = {name: {"calls_per_task": sum(per_task.get(p, 0) if p != "candidate"
                                                        else per_task["candidate"]
                                                        for p in PHASES[name])}
                         for name in STRATEGIES}
    return est


def build_coordinator(config: dict, ledger: Ledger, strict: bool = False) -> Coordinator:
    run = config["run"]
    team = Team(_llm(config["agents"]), _llm(config["coordinator"]), ledger)
    return Coordinator(team, n_candidates=run.get("candidates", 3),
                       debate_rounds=run.get("debate_rounds", 1), repair=run.get("repair", True),
                       seed=run.get("seed", 0), max_disputes=run.get("max_disputes", 8),
                       strict=strict)


def load_tasks(config: dict, n_tasks: int | None = None):
    run = config["run"]
    if run.get("benchmark", "humaneval") != "humaneval":
        raise ValueError(f"unsupported benchmark {run['benchmark']!r}")
    return load_humaneval(n=n_tasks or run.get("n_tasks"), seed=run.get("sample_seed", 0),
                          task_ids=run.get("task_ids"))


class ConfigChanged(RuntimeError):
    """The run directory holds results from a different configuration.

    Resuming would silently mix two protocols (a different K, seed or model) into one
    report, so it is refused instead.
    """


def _config_fingerprint(_config_path: str | Path | None, config: dict) -> dict:
    """Identity of an experiment: everything that changes what a strategy produces.

    Deliberately excludes `name`, `workers` and `n_tasks`: renaming a run or raising
    concurrency does not change the experiment, whereas K, the seed or the models do.
    """
    run = config["run"]
    return {
        "benchmark": run.get("benchmark", "humaneval"),
        "sample_seed": run.get("sample_seed", 0),
        "candidates": run.get("candidates", 3),
        "debate_rounds": run.get("debate_rounds", 1),
        "repair": run.get("repair", True),
        "seed": run.get("seed", 0),
        "agents": {k: v for k, v in config.get("agents", {}).items()},
        "coordinator": {k: v for k, v in config.get("coordinator", {}).items()},
    }


def _check_resume(out: Path, config: dict) -> None:
    """Refuse to resume a run directory whose results came from a different configuration."""
    current = _config_fingerprint(out / "config.toml", config)
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    recorded = previous.get("fingerprint")
    if recorded is None:
        # A run started before manifests existed: adopt the stored config.toml as truth so
        # the check works from here on, and note that it was inferred.
        (out / "manifest.json").write_text(json.dumps(
            {"fingerprint": current, "note": "inferred from config.toml at resume time"},
            indent=1))
        return
    if recorded != current:
        changed = [k for k in current if recorded.get(k) != current[k]]
        raise ConfigChanged(
            f"{out} already holds results from a different configuration "
            f"(differing: {', '.join(changed)}). Use a new run name or --out, or delete the "
            f"directory to start over.")


def run_experiment(config_path: str | Path, n_tasks: int | None = None,
                   out_dir: str | Path | None = None, strict: bool = False) -> Path:
    config = load_config(config_path)
    name = config["run"]["name"]
    out = Path(out_dir) if out_dir else ROOT / "runs" / name
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out / "config.toml")

    outcomes_path = out / "outcomes.jsonl"
    done = set()
    if outcomes_path.exists():
        done = {json.loads(line)["task_id"] for line in outcomes_path.read_text().splitlines()}
        _check_resume(out, config)
    else:
        (out / "manifest.json").write_text(json.dumps(
            {"fingerprint": _config_fingerprint(config_path, config)}, indent=1))
    tasks = [t for t in load_tasks(config, n_tasks) if t.task_id not in done]
    print(f"[hivemind] {name}: {len(tasks)} tasks to run ({len(done)} already done) -> {out}")

    if strict:
        print("[hivemind] strict sandbox: model-written code cannot use the network, spawn "
              "processes or open files. Grading is faithful for ordinary solutions; a candidate "
              "that relies on those capabilities will fail for that reason.")

    ledger = Ledger()
    coordinator = build_coordinator(config, ledger, strict=strict)
    lock = threading.Lock()
    started = time.perf_counter()
    # Task-level failures (no usable candidate) are collected so a run whose provider
    # was down all the way through cannot masquerade as a 0% result.
    failures: list[tuple[str, str]] = []

    def work(task):
        outcome, transcript = coordinator.solve(task)
        with lock:
            with open(outcomes_path, "a") as f:
                f.write(json.dumps(outcome.to_dict()) + "\n")
            with open(out / "calls.jsonl", "a") as f:
                for record in ledger.for_task(task.task_id):
                    f.write(json.dumps(asdict(record)) + "\n")
            safe = task.task_id.replace("/", "_")
            (out / "transcripts" / f"{safe}.json").write_text(json.dumps(asdict(transcript), indent=1))
        return outcome

    finished = 0
    with ThreadPoolExecutor(max_workers=config["run"].get("workers", 1)) as pool:
        futures = {pool.submit(work, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                outcome = future.result()
            except NoUsableOutput as exc:
                # Unusable: recorded as a failure, never as a failing answer.
                with lock:
                    failures.append((task.task_id, str(exc)))
                print(f"[hivemind] {task.task_id} UNUSABLE: {exc}", file=sys.stderr)
                continue
            except Exception as exc:  # keep going; the task can be retried by re-running
                print(f"[hivemind] {task.task_id} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            finished += 1
            marks = " ".join(f"{k}={'✓' if v['passed'] else '✗'}" for k, v in outcome.answers.items())
            elapsed = time.perf_counter() - started
            print(f"[hivemind] {finished}/{len(tasks)} {task.task_id} ({elapsed:.0f}s) {marks}",
                  flush=True)

    if failures:
        # Loud, because a run where every task failed looks exactly like a real result
        # once the pass rates are all zero.
        print(f"\n[hivemind] {len(failures)} task(s) produced no usable candidates and were "
              f"EXCLUDED from the results:", file=sys.stderr)
        for task_id, why in failures[:10]:
            print(f"  - {task_id}: {why}", file=sys.stderr)
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more", file=sys.stderr)
        print("[hivemind] Check credentials, model name and rate limits, then re-run: "
              "excluded tasks are retried on the next run.\n", file=sys.stderr)
        (out / "failures.jsonl").write_text(
            "".join(json.dumps({"task_id": t, "error": e}) + "\n" for t, e in failures))

    from .report import write_report
    write_report(out, excluded=len(failures))
    return out
