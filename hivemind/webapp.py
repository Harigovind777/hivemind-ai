"""Web interface: run the HiveMind team on any prompt, from a browser.

A thin layer over `Coordinator.solve`; it adds no new reasoning. Two things it is careful
about:

* **No fake accuracy.** A prompt typed here has no hidden test suite, so nothing here is
  ground truth. Every number is the team's own verified evidence and is labelled as such in
  the UI and in the JSON. Benchmark numbers still come from `python -m hivemind run`.
* **Visible failure.** A provider error, an empty candidate pool or an unreadable arbiter
  reply is reported as an error, never rendered as a 0% result.

Start it with `python -m hivemind web`. The team runs in a worker thread and the transcript
streams to the browser over Server-Sent Events.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .agents import Team
from .coordinator import STRATEGIES, Coordinator, NoUsableOutput
from .domains.freeform import detect_entry_point, looks_like_a_prompt, make_task, syntax_check
from .llm import Ledger, RefusalError, make_llm

ROOT = Path(__file__).resolve().parents[1]

STRATEGY_BLURB = {
    "solo": "One Coding agent alone: no brief, no tests, no teammates.",
    "first": "The team's first candidate, unresolved.",
    "vote": "Majority vote on identical behaviour.",
    "debate": "Free-form debate; the Coordinator judges the arguments.",
    "tests_only": "Most team tests passed, trusting the Testing agent blindly.",
    "arbitration_select": "Coordinator rules on disputed cases from real outputs.",
    "arbitration": "The above, plus one repair round driven by verified failures.",
}


# --------------------------------------------------------------------------- jobs

class Job:
    """One solve, its transcript, and its result. Safe to touch from the event thread."""

    def __init__(self, job_id: str, prompt: str, config: dict):
        self.id = job_id
        self.prompt = prompt
        self.config = config
        self.events: list[dict] = []
        self.done = threading.Event()
        self.error: str | None = None
        self.result: dict | None = None
        self.lock = threading.Lock()
        self.started = time.time()

    def emit(self, kind: str, **data) -> None:
        # `kind` is positional here and callers pass an error's own `kind` keyword, so the
        # dict is built explicitly rather than with **data, which would collide.
        with self.lock:
            self.events.append({"kind": kind, **data})

    def snapshot(self, after: int = 0) -> list[dict]:
        with self.lock:
            return self.events[after:]


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _build_coordinator(config: dict, ledger: Ledger) -> Coordinator:
    agents = config["agents"]
    coo = config.get("coordinator", agents)
    team = Team(make_llm(agents["provider"], agents["model"]),
                make_llm(coo["provider"], coo["model"]), ledger)
    return Coordinator(team, n_candidates=config.get("candidates", 3),
                       debate_rounds=config.get("debate_rounds", 1),
                       repair=config.get("repair", True),
                       seed=config.get("seed", 0),
                       strict=config.get("strict_sandbox", False))


def _run_job(job: Job) -> None:
    """Solve in the caller's thread; the transcript is mirrored into the job as it goes."""
    started = time.time()
    try:
        task = make_task(job.prompt, job.config.get("entry_point"),
                         task_id=f"user/{job.id[:8]}")
        job.emit("task", entry_point=task.entry_point, spec=task.spec)
        ledger = Ledger()
        coordinator = _build_coordinator(job.config, ledger)
        # Tap the asker so each agent call is mirrored into the job as it happens, and the
        # browser sees the real exchange rather than everything at the end.
        coordinator._asker = _Tap(coordinator, job)  # type: ignore[method-assign]

        outcome, transcript = coordinator.solve(task, review=job.config.get("review", True))
        job.emit("transcript", turns=transcript.turns)
        job.result = _shape(outcome, task, ledger, started)
        job.emit("done", ok=True)
    except NoUsableOutput as exc:
        job.error = f"no usable output: {exc}"
        job.emit("error", message=job.error, error_kind="NoUsableOutput")
    except RefusalError as exc:
        job.error = f"the model declined the request: {exc}"
        job.emit("error", message=job.error, error_kind="RefusalError")
    except ValueError as exc:
        job.error = str(exc)
        job.emit("error", message=job.error, error_kind="BadPrompt")
    except RecursionError as exc:
        job.error = ("the team recursed too deeply on this prompt; try a simpler "
                     "specification or fewer candidates")
        job.emit("error", message=job.error, error_kind="RecursionError")
    except Exception as exc:  # surfaced to the UI; never rendered as a result
        job.error = f"{type(exc).__name__}: {exc}"
        job.emit("error", message=job.error, error_kind=type(exc).__name__)
    finally:
        job.done.set()


class _Tap:
    """Wraps a Coordinator so every agent call is mirrored into the job as it happens.

    The original `_asker` is captured at construction. Reading it back off the coordinator
    at call time would recurse forever, since the tap replaces that attribute.
    """

    def __init__(self, coordinator: Coordinator, job: Job):
        self._original = coordinator._asker
        self._job = job

    def __call__(self, transcript):
        asker = self._original(transcript)

        def ask(role: str, prompt: str, phase: str, seed: int) -> str:
            self._job.emit("phase", phase=phase, role=role, prompt=prompt)
            text = asker(role, prompt, phase, seed)
            self._job.emit("reply", phase=phase, role=role, response=text)
            return text

        return ask


def _shape(outcome, task, ledger: Ledger, started: float) -> dict:
    """Everything the UI needs, with the evidence-only labelling made explicit."""
    d = outcome.diagnostics
    return {
        "task_id": outcome.task_id,
        "entry_point": task.entry_point,
        "spec": task.spec,
        "evidence_only": outcome.evidence_only,
        "ground_truth": not outcome.evidence_only,
        "candidates": outcome.candidates,
        "oracle_evidence": any(outcome.candidate_passed),
        "answers": {
            name: {
                "code": value["code"],
                "source": value["source"],
                "passed": value["passed"],
                "grade_reason": value["grade_reason"],
                "security_findings": value.get("security_findings", []),
                "evidence": value.get("evidence"),
                "syntax_error": syntax_check(value["code"]),
                "blurb": STRATEGY_BLURB.get(name, ""),
            }
            for name, value in outcome.answers.items()
        },
        "strategy_order": list(STRATEGIES),
        "diagnostics": {
            "distinct_behaviours": d["distinct_behaviours"],
            "max_distinct_behaviours": d.get("max_distinct_behaviours"),
            "unanimous": d["unanimous"],
            "n_disputes": d["n_disputes"],
            "rulings_unreadable": d.get("rulings_unreadable", 0),
            "rulings": d["rulings"],
            "n_spec_cases": d["n_spec_cases"],
            "n_tester_cases": d["n_tester_cases"],
            "verified_cases": d.get("verified_cases", 0),
            "cases_total": d.get("cases_total", 0),
            "repaired": d["repaired"],
            "failing_after_selection": d["failing_after_selection"],
            "unsafe_candidates": d["unsafe_candidates"],
            "solo_unsafe": d.get("solo_unsafe", False),
            "debate_unsafe": d.get("debate_unsafe", False),
            "has_reference": d.get("has_reference", False),
            "tester_judged": d["tester_judged"],
        },
        "review": outcome.review,
        "calls": [asdict(r) for r in ledger.records],
        "elapsed_s": round(time.time() - started, 1),
        "cost_usd": _cost(ledger),
    }


def _int_or(raw, fallback: int, low: int, high: int) -> int:
    """Parse a form field that may be blank, and clamp it into a sane range."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = fallback
    return max(low, min(high, value))


def _cost(ledger: Ledger) -> float:
    from .report import cost_usd
    total = 0.0
    for record in ledger.records:
        value = cost_usd(record.provider, record.model, record.input_tokens, record.output_tokens)
        total += value or 0.0
    return round(total, 4)


# --------------------------------------------------------------------------- app

def create_app(default_config: dict | None = None, jobs: dict | None = None):
    from fastapi import FastAPI, Form, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="HiveMind AI", docs_url="/api/docs")
    store = JOBS if jobs is None else jobs
    base = dict(default_config or {})

    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (static_dir / "index.html").read_text()
        return HTMLResponse(html)

    @app.get("/api/config")
    def get_config() -> dict:
        """What the server will run with, so the UI can show it and the user can change it."""
        return {
            "agents": base.get("agents", {}),
            "coordinator": base.get("coordinator", base.get("agents", {})),
            "candidates": base.get("candidates", 3),
            "debate_rounds": base.get("debate_rounds", 1),
            "repair": base.get("repair", True),
            "strict_sandbox": base.get("strict_sandbox", False),
            "providers": ["ollama", "anthropic"],
            "note": ("A prompt typed here has no hidden test suite. Numbers are the team's "
                     "own verified evidence, not benchmark accuracy."),
            "configured": bool(base.get("agents", {}).get("provider")),
        }

    @app.post("/api/solve")
    def solve(prompt: str = Form(...), entry_point: str = Form(""),
              candidates: str = Form(""), debate_rounds: str = Form(""),
              repair: bool = Form(True), provider: str = Form(""),
              model: str = Form(""), coordinator_model: str = Form(""),
              strict_sandbox: bool = Form(False), review: bool = Form(True)) -> dict:
        problem = looks_like_a_prompt(prompt, entry_point)
        if problem:
            raise HTTPException(status_code=400, detail=problem)
        try:
            task = make_task(prompt, entry_point or None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        agents = dict(base.get("agents", {}))
        if provider:
            agents["provider"] = provider
        if model:
            agents["model"] = model
        coo = dict(base.get("coordinator", agents))
        if coordinator_model:
            coo["model"] = coordinator_model

        # An empty field means "whatever the server config says"; only an explicit value
        # overrides it, so the UI can prefill defaults without them becoming mandatory.
        k = _int_or(candidates, base.get("candidates", 3), low=1, high=7)
        rounds = _int_or(debate_rounds, base.get("debate_rounds", 1), low=0, high=3)

        job_id = uuid.uuid4().hex
        job = Job(job_id, prompt, {
            "agents": agents, "coordinator": coo, "entry_point": task.entry_point,
            "candidates": k, "debate_rounds": rounds,
            "repair": repair, "seed": 0, "strict_sandbox": strict_sandbox, "review": review,
        })
        with JOBS_LOCK:
            store[job_id] = job
            if len(store) > 64:                       # keep memory bounded
                for old in sorted(store, key=lambda k: store[k].started)[:16]:
                    store.pop(old, None)

        thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
        thread.start()
        return {"job_id": job_id, "entry_point": task.entry_point,
                "evidence_only": True, "candidates": job.config["candidates"]}

    @app.get("/api/events/{job_id}")
    def events(job_id: str) -> StreamingResponse:
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="unknown job")

        def stream():
            cursor = 0
            # Heartbeats keep intermediaries from closing an idle connection during a long
            # local-model call.
            while True:
                batch = job.snapshot(cursor)
                cursor += len(batch)
                for event in batch:
                    yield f"data: {json.dumps(event)}\n\n"
                if job.done.is_set():
                    tail = job.snapshot(cursor)
                    for event in tail:
                        yield f"data: {json.dumps(event)}\n\n"
                    yield f"data: {json.dumps({'kind': 'close', 'error': job.error})}\n\n"
                    return
                if not batch:
                    yield ": keepalive\n\n"
                time.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/result/{job_id}")
    def result(job_id: str) -> dict:
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="unknown job")
        if not job.done.is_set():
            return {"status": "running"}
        if job.error:
            return {"status": "error", "error": job.error}
        return {"status": "ok", **job.result}

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, config: str | Path | None = None) -> None:
    """Entry point for `python -m hivemind web`."""
    import uvicorn

    from .runner import load_config
    settings = load_config(config) if config else {}
    app = create_app(settings)
    print(f"[hivemind] web interface on http://{host}:{port}")
    print("[hivemind] prompts typed here have no hidden tests: results are the team's own "
          "verified evidence, not benchmark accuracy.")
    uvicorn.run(app, host=host, port=port, log_level="info")
