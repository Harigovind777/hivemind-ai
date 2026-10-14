# HiveMind web interface: design

Status: approved in conversation 2026-10-14. Owner: Harigovind.

## Goal

A local web app, served by the `hivemind` package, for managing and running the
agents. It is used first for watching one problem solved live (for demos to the
mentor), and second for launching and tracking full experiments. Each person runs
their own copy on their own machine; Windows must work (the sandbox fix is a
separate spec).

## Decisions

- **Stack:** Python standard library only. `http.server` with threads, live
  updates as Server-Sent Events, and a plain HTML/CSS/JS front end with no build
  step. No new dependencies.
- **Access:** binds to `127.0.0.1` only; no login.
- **Agent management covers:** team and Coordinator models and settings,
  pipeline settings, and an optional model per role. Role prompts stay fixed in
  code so experiments stay comparable.
- **One job at a time:** a live run or an experiment, never both.
- **Python:** 3.11+ (the code already needs `tomllib`); `pyproject.toml` and the
  README are corrected.

## Architecture

```
python -m hivemind serve [--port 8765] [--no-open]
```

Changes to the existing core (backward compatible):

- `agents.py`: `Team(agent_llm, coordinator_llm, ledger, role_llms=None)`;
  `Transcript(task_id, on_event=None)`; `Agent.ask` emits `turn_started` before
  a call and `turn` after it, and raises `Cancelled` before a call when the
  transcript's cancel flag is set.
- `coordinator.py`: emits `evidence` after the sandbox runs and `rulings` after
  arbitration, through the transcript. `solve(task, review=False, on_event=None,
  cancel=None)`.
- `runner.py`: configs may add `[roles.<role>]` sections overriding the team
  model for that role.

New modules:

| Module | Job |
|---|---|
| `hivemind/config.py` | List, load, validate and save configs; write TOML |
| `hivemind/web/server.py` | HTTP server: routing, JSON, SSE, static files |
| `hivemind/web/live.py` | One live run in a thread; event history for replay |
| `hivemind/web/jobs.py` | Experiments as a `python -m hivemind run` subprocess; progress from `outcomes.jsonl`; cancel |
| `hivemind/web/status.py` | Ollama reachability and pulled models; API key presence |
| `hivemind/web/static/` | `index.html`, `app.js`, `app.css` |

## Screens

1. **Run:** pick problem (searchable), config, "Include reviews"; Run. Pipeline
   checklist (waiting / thinking / done / failed) beside a feed of agent cards
   (agent, model, tokens, seconds, reply; prompt folded). Evidence card after the
   sandbox, rulings card after arbitration, pass/fail for all seven strategies at
   the end. Saved to `runs/demos/`.
2. **Agents:** config list; form for team model, Coordinator model, per-role
   overrides ("same as team" or custom), and pipeline settings. Pulled Ollama
   models offered in a list; warnings for unpulled models and missing API key.
   Save / Save as.
3. **Experiments:** config + number of problems; Launch. Progress, per-task
   marks, log tail, elapsed time, Cancel. Relaunch resumes.
4. **Results:** every `runs/` folder with date, model, tasks, arbitration pass
   rate; a run's strategy table, evidence numbers, `report.md`, and transcript
   replay in the Run tab's card view.

A status strip shows Ollama, pulled models, API key and the current job.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | Ollama, models, key, current job |
| `GET /api/tasks` | HumanEval problems: id, function, first docstring line |
| `GET /api/configs`, `GET/PUT /api/configs/<name>` | List, read, save (validated) |
| `POST /api/live` → `{id}` | Start a live run |
| `GET /api/live/<id>/events` | SSE stream; replays history on connect |
| `POST /api/live/<id>/cancel` | Stop at the next model-call boundary |
| `POST /api/experiments` → `{id}`; `GET /api/experiments/<id>`; `POST …/cancel` | Launch, poll, cancel |
| `GET /api/runs`, `GET /api/runs/<name>`, `GET /api/runs/<name>/transcripts/<task>` | Results |

Event types: `turn_started`, `turn`, `evidence`, `rulings`, `outcome`, `error`,
`done`. Each carries a `seq`.

## Errors

- Ollama down, model not pulled, or Claude without a key: Run disabled with the
  exact fix shown.
- Failed model call: shown on its card; the pipeline continues (current
  behaviour).
- Invalid config: per-field messages; nothing written.
- Crashed experiment: marked failed with log tail; relaunch resumes.
- Names of configs and runs are limited to `[A-Za-z0-9_.-]` without `..`, so no
  request reaches files outside `configs/` and `runs/`.
- A second job while one is running: HTTP 409 with the running job's name.

## Testing

Test-first. Config round-trip and validation; per-role models; event order and
cancel on a scripted solve; a real server on a spare port driven with `urllib`,
including the SSE stream; the experiment manager against a fake runner script;
static files served. The existing suite keeps passing.
