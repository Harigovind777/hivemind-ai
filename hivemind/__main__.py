"""Command line: python -m hivemind {estimate,run,report,demo} ..."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .llm import Ledger
from .report import write_report
from .runner import (ROOT, build_coordinator, estimate_cost, load_config, run_experiment)
from .domains.code import load_humaneval


def serve(host: str, port: int, config: str | None) -> None:
    from .webapp import serve as _serve
    _serve(host=host, port=port, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(prog="hivemind", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_est = sub.add_parser("estimate", help="project the cost of a run before spending anything")
    p_est.add_argument("config")
    p_est.add_argument("--tasks", type=int, help="override the number of tasks")
    p_est.add_argument("--max-cost", type=float, dest="max_cost",
                       help="print the estimate and exit non-zero if it exceeds this USD")

    p_run = sub.add_parser("run", help="run an experiment config over benchmark tasks")
    p_run.add_argument("config")
    p_run.add_argument("--tasks", type=int, help="override the number of tasks")
    p_run.add_argument("--out", help="output directory (default runs/<name>)")
    p_run.add_argument("--strict-sandbox", action="store_true",
                       help="deny network, process and filesystem capabilities to model-written "
                            "code (off by default so grading stays faithful)")
    p_run.add_argument("--yes", action="store_true",
                       help="skip the cost confirmation prompt on an interactive terminal")
    p_run.add_argument("--max-cost", type=float, dest="max_cost",
                       help="refuse to start if the projected cost exceeds this USD")

    p_report = sub.add_parser("report", help="rebuild report.md for a run directory")
    p_report.add_argument("run_dir")

    p_demo = sub.add_parser("demo", help="solve one task with the full six-agent team and print the discussion")
    p_demo.add_argument("config")
    p_demo.add_argument("--task", default="HumanEval/0")

    p_web = sub.add_parser("web", help="serve the browser interface for running the team on any prompt")
    p_web.add_argument("config", nargs="?", help="TOML config to read the model settings from")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--reload", action="store_true", help="auto-reload on code changes")

    args = parser.parse_args()
    if args.command == "estimate":
        _finish(estimate_cmd(args.config, args.tasks), args.max_cost)
    elif args.command == "run":
        # Cost is confirmed before any token is spent: the headline config is ~1800 calls.
        if not args.yes and not _confirm_cost(args.config, args.tasks, args.max_cost):
            print("[hivemind] cancelled; nothing was run.")
            return
        out = run_experiment(args.config, n_tasks=args.tasks, out_dir=args.out,
                             strict=args.strict_sandbox)
        print(f"[hivemind] report: {out / 'report.md'}")
    elif args.command == "report":
        print(write_report(args.run_dir))
    elif args.command == "web":
        serve(args.host, args.port, args.config)
    else:
        demo(args.config, args.task)


def _format_estimate(e: dict) -> str:
    cost = "free (local model)" if not e["cost_usd"] else f"~${e['cost_usd']:,.2f}"
    lines = [f"[hivemind] projected cost of {e['n_tasks']} tasks:", "",
             f"  model calls      : {e['calls_total']:,} total "
             f"({e['calls_per_task']['agents']} agents + {e['calls_per_task']['coordinator']} "
             f"coordinator per task)",
             f"  tokens           : ~{e['input_tokens_total']:,} in / "
             f"~{e['output_tokens_total']:,} out",
             f"  estimated cost   : {cost}",
             f"  agents           : {e['agents_model']}",
             f"  coordinator      : {e['coordinator_model']}",
             "",
             "  token figures assume "
             f"{e['assumed_tokens_per_call']['input']} in / "
             f"{e['assumed_tokens_per_call']['output']} out per call and are the least precise "
             "number here; measure with --tasks 10 for a real rate."]
    return "\n".join(lines)


def estimate_cmd(config_path: str, n_tasks: int | None) -> dict:
    e = estimate_cost(load_config(config_path), n_tasks or 0)
    print(_format_estimate(e))
    return e


def _finish(e: dict, max_cost: float | None) -> None:
    if max_cost is not None and e["cost_usd"] and e["cost_usd"] > max_cost:
        print(f"[hivemind] estimate ${e['cost_usd']:,.2f} exceeds --max-cost ${max_cost:,.2f}",
              file=sys.stderr)
        raise SystemExit(2)


def _confirm_cost(config_path: str, n_tasks: int | None, max_cost: float | None) -> bool:
    """Show the projection and ask before spending. Non-interactive runs proceed."""
    if not sys.stdin.isatty():
        return True
    e = estimate_cost(load_config(config_path), n_tasks or 0)
    print(_format_estimate(e))
    if max_cost is not None and e["cost_usd"] and e["cost_usd"] > max_cost:
        print(f"[hivemind] refusing to start: estimate exceeds --max-cost ${max_cost:,.2f}",
              file=sys.stderr)
        return False
    try:
        return input("[hivemind] proceed? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


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
