# HiveMind AI

**Specialist AI agents that work as a team, and settle their disagreements with evidence instead of votes.**

Most AI agents work alone. HiveMind gives one task to a team of specialists
(Research, Coding, Testing, Security, Documentation and Ethics agents) run by a
**Coordinator**. The Coordinator decides who acts, merges their work and, most
importantly, **resolves their disagreements**.

The research question:

> When agents disagree, does settling it with *checkable evidence* beat
> majority vote and free-form debate?

When candidate solutions disagree, the Coordinator does not count votes or
reward the most persuasive argument. It runs every candidate on the team's test
cases in a sandbox, shows itself the **real execution outputs** on each disputed
case, rules which output the specification requires, and picks or repairs the
solution that matches the verified results. The full design is in
[`docs/design.md`](docs/design.md).

## Initial-stage results

See [`runs/pilot-local-qwen2.5-coder-3b/report.md`](runs/pilot-local-qwen2.5-coder-3b/report.md)
for the full pilot report: models used, procedure, results, statistics and
per-task outcomes.

<!-- RESULTS -->

## Get the code (any device)

The repository is private, so you need to accept the collaborator invite first.
You'll find it in your GitHub notifications/email, or at
https://github.com/Harigovind777/hivemind-ai/invitations.

- **Download without git:** open https://github.com/Harigovind777/hivemind-ai,
  then click **Code → Download ZIP**.
- **Clone with git:**
  ```bash
  git clone https://github.com/Harigovind777/hivemind-ai.git
  cd hivemind-ai
  ```
  On a new device, sign in first with `gh auth login` or a personal access
  token, because private repos need authentication.

## Run it

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                  # 19 offline tests, ~2 s
```

**Free local pilot** (no API key). Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5-coder:3b
python -m hivemind run configs/pilot-local.toml --tasks 5       # ~7 min on an M-series Mac
python -m hivemind demo configs/pilot-local.toml --task HumanEval/2   # watch all six agents talk
```

**Claude** (the main experiments):

```bash
export ANTHROPIC_API_KEY=...                            # or: ant auth login
python -m hivemind run configs/claude-main.toml --tasks 10      # measure real cost first
python -m hivemind run configs/claude-main.toml                 # all 164 tasks (resumes if interrupted)
```

Every run writes `runs/<name>/` with these files:

- `report.md`: human-readable results
- `summary.json`: the same numbers, machine-readable
- `outcomes.jsonl`: the answer every strategy chose for each task
- `calls.jsonl`: every model call, with its role, model, tokens and latency
- `transcripts/`: every message the agents exchanged

## The strategies being compared

All strategies are applied to the **same** candidates for each task, so any
difference comes from how disagreement is resolved.

| Strategy | How the final answer is chosen |
|---|---|
| `solo` | One Coding agent alone (the "isolated AI" baseline) |
| `first` | The team's first candidate, written with the Research brief, unresolved |
| `vote` | Largest group of candidates with identical behaviour |
| `debate` | Candidates critique and revise; the Coordinator judges the arguments |
| `tests_only` | Most team tests passed, trusting the Testing agent blindly |
| `arbitration_select` | Coordinator rules on disputed cases from real outputs; best match wins |
| `arbitration` | The above, plus one repair round driven by the verified failing cases |

## Layout

```
hivemind/
  agents.py        the six specialist roles + coordinator prompts, call logging
  coordinator.py   shared pipeline, all seven strategies, evidence-based arbitration
  sandbox.py       isolated, resource-limited execution of model-written code
  security.py      Security agent's static checker
  domains/code.py  HumanEval loading, code extraction, test parsing, hidden-test grading
  llm.py           Claude / Ollama / scripted providers behind one interface
  runner.py        experiment runner (parallel, resumable)
  report.py        report.md: models, procedure, results, McNemar tests, diagnostics
configs/           pilot-local.toml (free), claude-main.toml (headline run)
data/              HumanEval (MIT licence, © OpenAI)
docs/design.md     research design, threats to validity, 1-month roadmap
tests/             offline test suite using a scripted model
```
