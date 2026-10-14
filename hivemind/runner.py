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
from .coordinator import Coordinator
from .domains.code import load_humaneval
from .llm import Ledger, make_llm

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _llm(section: dict):
    options = {k: v for k, v in section.items() if k not in ("provider", "model")}
    return make_llm(section["provider"], section["model"], **options)


def build_coordinator(config: dict, ledger: Ledger) -> Coordinator:
    run = config["run"]
    role_llms = {role: _llm(section) for role, section in config.get("roles", {}).items()}
    team = Team(_llm(config["agents"]), _llm(config["coordinator"]), ledger, role_llms=role_llms)
    return Coordinator(team, n_candidates=run.get("candidates", 3),
                       debate_rounds=run.get("debate_rounds", 1), repair=run.get("repair", True),
                       seed=run.get("seed", 0))


def load_tasks(config: dict, n_tasks: int | None = None):
    run = config["run"]
    if run.get("benchmark", "humaneval") != "humaneval":
        raise ValueError(f"unsupported benchmark {run['benchmark']!r}")
    return load_humaneval(n=n_tasks or run.get("n_tasks"), seed=run.get("sample_seed", 0),
                          task_ids=run.get("task_ids"))


def run_experiment(config_path: str | Path, n_tasks: int | None = None,
                   out_dir: str | Path | None = None) -> Path:
    config = load_config(config_path)
    name = config["run"]["name"]
    out = Path(out_dir) if out_dir else ROOT / "runs" / name
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out / "config.toml")

    outcomes_path = out / "outcomes.jsonl"
    done = set()
    if outcomes_path.exists():
        done = {json.loads(line)["task_id"] for line in outcomes_path.read_text().splitlines()}
    tasks = [t for t in load_tasks(config, n_tasks) if t.task_id not in done]
    print(f"[hivemind] {name}: {len(tasks)} tasks to run ({len(done)} already done) -> {out}")

    ledger = Ledger()
    coordinator = build_coordinator(config, ledger)
    lock = threading.Lock()
    started = time.perf_counter()

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
            except Exception as exc:  # keep going; the task can be retried by re-running
                print(f"[hivemind] {task.task_id} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            finished += 1
            marks = " ".join(f"{k}={'✓' if v['passed'] else '✗'}" for k, v in outcome.answers.items())
            elapsed = time.perf_counter() - started
            print(f"[hivemind] {finished}/{len(tasks)} {task.task_id} ({elapsed:.0f}s) {marks}",
                  flush=True)

    from .report import write_report
    write_report(out)
    return out
