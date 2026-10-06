"""Command line: python -m hivemind {run,report,demo} ..."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .llm import Ledger
from .report import write_report
from .runner import ROOT, build_coordinator, load_config, run_experiment
from .domains.code import load_humaneval


def main() -> None:
    parser = argparse.ArgumentParser(prog="hivemind", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an experiment config over benchmark tasks")
    p_run.add_argument("config")
    p_run.add_argument("--tasks", type=int, help="override the number of tasks")
    p_run.add_argument("--out", help="output directory (default runs/<name>)")

    p_report = sub.add_parser("report", help="rebuild report.md for a run directory")
    p_report.add_argument("run_dir")

    p_demo = sub.add_parser("demo", help="solve one task with the full six-agent team and print the discussion")
    p_demo.add_argument("config")
    p_demo.add_argument("--task", default="HumanEval/0")

    args = parser.parse_args()
    if args.command == "run":
        out = run_experiment(args.config, n_tasks=args.tasks, out_dir=args.out)
        print(f"[hivemind] report: {out / 'report.md'}")
    elif args.command == "report":
        print(write_report(args.run_dir))
    else:
        demo(args.config, args.task)


def demo(config_path: str, task_id: str) -> None:
    config = load_config(config_path)
    [task] = load_humaneval(task_ids=[task_id])
    coordinator = build_coordinator(config, Ledger())
    outcome, transcript = coordinator.solve(task, review=True)
    for turn in transcript.turns:
        print(f"\n===== [{turn['phase']}] {turn['role'].upper()} =====\n{turn['response'].strip()}")
    print("\n===== RESULTS =====")
    for name, answer in outcome.answers.items():
        print(f"{name:20s} {'PASS' if answer['passed'] else 'fail'}  ({answer['source']})")
    out = ROOT / "runs" / "demos" / f"{task_id.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"outcome": outcome.to_dict(), "transcript": asdict(transcript)}, indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
