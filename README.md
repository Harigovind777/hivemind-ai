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

## Results

No run has been committed yet, so there is nothing to quote here. Produce a report with
either command below and it appears under `runs/<name>/report.md`.

```bash
python -m hivemind run configs/pilot-local.toml --tasks 5   # free local pilot
python -m hivemind run configs/claude-main.toml --tasks 10  # headline config, small sample
```

Every report states its own validity up front: an incomplete-run banner if tasks were
excluded, a **pool-diversity** figure (a low one means arbitration had nothing to
arbitrate, so a flat result is not evidence against the method), a count of arbiter replies
that could not be parsed, which candidates failed for mechanical reasons such as a timeout
rather than wrong logic, and Holm-corrected p-values across the baseline comparisons.

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

Requires Python 3.11+ (3.11 is the floor because the runner reads TOML configs with
the standard library's `tomllib`).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                  # 57 offline tests, ~5 s
```

**Free local pilot** (no API key). Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5-coder:3b
python -m hivemind run configs/pilot-local.toml --tasks 5       # ~7 min on an M-series Mac
python -m hivemind demo configs/pilot-local.toml --task HumanEval/2   # watch all six agents talk
```

**Web interface** — run the team on any prompt from a browser:

```bash
pip install -e ".[web]"
python -m hivemind web configs/pilot-local.toml     # http://127.0.0.1:8000
```

Paste a function signature and docstring, and the six agents work on it while you watch the
transcript stream in. The result shows all seven strategies side by side on the same
candidates, plus why the Coordinator ruled the way it did.

A prompt typed into the browser has **no hidden test suite**, so the interface never claims
a pass rate. Every number is the team's own verified evidence, labelled as such on the page
and in the JSON (`evidence_only: true`). Benchmark numbers still come from
`python -m hivemind run`; the two are not comparable.

**Claude** (the main experiments):

```bash
export ANTHROPIC_API_KEY=...                            # or: ant auth login
python -m hivemind estimate configs/claude-main.toml    # projected calls/tokens/USD first
python -m hivemind run configs/claude-main.toml --tasks 10      # measure the real rate
python -m hivemind run configs/claude-main.toml                 # all 164 tasks (resumes if interrupted)
```

`run` prints that same projection and asks before spending anything; `--yes` skips the
prompt, `--max-cost 25` refuses to start above a budget, and `estimate` only prints it.
For the headline config the projection is about 1,670 calls and ~$100, so measure with
`--tasks 10` first.

Model-written code runs in a separate, resource-limited process with a scrubbed
environment (it cannot read your `ANTHROPIC_API_KEY` out of `os.environ`) and its own
process group. Add `--strict-sandbox` to also deny it the network, process spawning and
file access; that is off by default because blocking capabilities changes what candidate
code can do, and the decision is recorded in the run's diagnostics.

A run that produces no usable candidates **fails loudly** instead of reporting 0%:
unusable tasks are excluded from every number, counted in `failures.jsonl`, and flagged at
the top of `report.md`. Resuming refuses to mix configurations, so a run directory always
holds one protocol.

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
  domains/freeform.py  user-typed prompts: evidence grading, never a fake pass rate
  llm.py           Claude / Ollama / scripted providers behind one interface
  runner.py        experiment runner (parallel, resumable), cost estimation
  report.py        report.md: models, procedure, results, McNemar tests, diagnostics
  webapp.py        FastAPI app: streaming transcript, all seven strategies
  static/index.html  the interface itself (single page, no build step)
configs/           pilot-local.toml (free), claude-main.toml (headline run)
data/              HumanEval (MIT licence, © OpenAI)
docs/design.md     research design, threats to validity, 1-month roadmap
tests/             offline test suite using a scripted model
```

## Guardrails

A research run is only worth what its failure modes are worth, so the harness refuses to
produce confident nonsense:

- **No usable output is an error, not a result.** A task whose candidates all come back
  empty raises instead of scoring 0%; `run` collects those tasks separately, excludes them
  from every number and banners the report.
- **Unreadable replies are not "UNSURE".** An arbiter verdict that cannot be parsed is
  recorded as a parse failure, never silently dropped or guessed, because a dropped case
  would shrink the set a candidate must satisfy and flatter every strategy.
- **Timeouts are not wrong answers.** Grading reports why a candidate failed, and the
  report breaks the reasons out so a slow-but-correct solution is visible.
- **Correction code is kept.** `sanitize` strips self-tests but preserves module-level
  setup (`try/except` import fallbacks, version-gated imports, tables built in a loop),
  because removing those turned correct solutions into `NameError`s.
- **Candidate diversity is reported.** `distinct_behaviours` and pool diversity were
  recorded but never surfaced, which hid whether arbitration had anything to arbitrate.
- **Statistics are corrected.** Six McNemar comparisons get Holm–Bonferroni adjustment,
  reported next to the raw values.
- **Resume is checked.** A run directory records a config fingerprint and refuses to mix
  two protocols into one report.
