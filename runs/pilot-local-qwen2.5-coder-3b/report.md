# HiveMind run report: `pilot-local-qwen2.5-coder-3b`

Generated 2026-10-15 from `pilot-local-qwen2.5-coder-3b/`. Benchmark: **HumanEval**, 7 tasks (seeded random sample, sample_seed=0). Graded by the benchmark's hidden unit tests, which no agent ever sees.

## Models used

| Provider | Model | Roles | Calls | Failed calls | Input tokens | Output tokens | Avg latency | Cost (USD) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ollama | `qwen2.5-coder:3b` | coding, coordinator, research, testing | 78 | 0 | 37,217 | 20,360 | 7.8s | $0.00 |

Agent settings: temperature=0.7, max_tokens=1024. Coordinator settings: temperature=0.2, max_tokens=1024.
Local (Ollama) models cost nothing per call; Claude costs use list prices.

## Procedure

Every task goes through one shared pipeline, and every strategy below is applied to the **same** pool of candidates, so differences between strategies come from how disagreements are resolved:

1. **Solo baseline**: one Coding agent writes the function alone.
2. **Research brief**: the Research agent analyses the spec (contract, edge cases, ambiguities).
3. **Candidates**: 3 Coding agents write solutions with the brief, each given a different approach hint for diversity.
4. **Team tests**: the Testing agent writes assert-style test cases from the spec alone. The docstring's own worked examples are added as trusted specification cases.
5. **Execution**: every candidate runs on every case in an isolated, resource-limited Python process; the Security agent's static checker flags risky code.
6. **Resolution**: each strategy picks (or produces) a final answer:

| Strategy | What it does | Pipeline phases charged |
|---|---|---|
| `solo` | A single Coding agent working alone: no brief, no tests, no teammates. | solo |
| `first` | The team's first candidate (Coding agent + Research brief) with no resolution step. | brief, candidate:0 |
| `vote` | Majority vote: run every candidate on the team's test inputs and pick the largest group with identical behaviour. | brief, candidate, tests |
| `debate` | Free-form debate: candidates critique each other and revise; the Coordinator picks a winner from the arguments, with no execution. | brief, candidate, debate, judge |
| `tests_only` | Pick the candidate passing the most team-written tests, trusting the Testing agent's expected values without checking them. | brief, candidate, tests |
| `arbitration_select` | Evidence-based arbitration, selection only: the Coordinator rules on each disputed case from the spec and real execution outputs, then the candidate matching the verified results is chosen. | brief, candidate, tests, arbiter |
| `arbitration` | Evidence-based arbitration followed by one repair round in which the Coding agent sees the verified failing cases; the repair is kept only if it scores higher. | brief, candidate, tests, arbiter, repair |

Debate rounds: 1. Repair round: on.

## Results

| Strategy | Passed | Pass rate | 95% CI (Wilson) | Model calls / task | Tokens / task |
|---|---:|---:|---|---:|---:|
| `solo` | 7/7 | 100.0% | 64.6%–100.0% | 1.0 | 373 |
| `first` | 6/7 | 85.7% | 48.7%–97.4% | 2.0 | 1,083 |
| `vote` | 7/7 | 100.0% | 64.6%–100.0% | 5.0 | 2,796 |
| `debate` | 7/7 | 100.0% | 64.6%–100.0% | 8.0 | 6,736 |
| `tests_only` | 7/7 | 100.0% | 64.6%–100.0% | 5.0 | 2,796 |
| `arbitration_select` | 7/7 | 100.0% | 64.6%–100.0% | 5.7 | 3,094 |
| `arbitration` | 7/7 | 100.0% | 64.6%–100.0% | 6.1 | 3,539 |

**Oracle** (at least one of the 3 candidates is correct): 7/7 (100.0%). This is the ceiling for any strategy that only *selects* a candidate; repair and debate can exceed it.

## Arbitration vs. each baseline (paired, same tasks)

| Baseline | Arbitration right, baseline wrong | Baseline right, arbitration wrong | Exact McNemar p | Holm-corrected p |
|---|---:|---:|---:|---:|
| `solo` | 0 | 0 | 1.000 | 1.000 |
| `first` | 1 | 0 | 1.000 | 1.000 |
| `vote` | 0 | 0 | 1.000 | 1.000 |
| `debate` | 0 | 0 | 1.000 | 1.000 |
| `tests_only` | 0 | 0 | 1.000 | 1.000 |
| `arbitration_select` | 0 | 0 | 1.000 | 1.000 |

The Holm column corrects for the 6 comparisons in this table. Read that one: at these sample sizes an uncorrected p-value of 0.05 across six baselines is roughly what you expect by chance.


## Evidence diagnostics

### Did the disagreement the method needs actually happen?

- Tasks where candidates behaved differently: 1/7
- Tasks needing a ruling: 5/7; disputed cases: 16
- Distinct behaviours among the 3 candidates: 6 tasks with 1 distinct behaviours, 1 task with 2 distinct behaviours
- Pool diversity: 38.1% of the maximum 3. **A low value here means arbitration had little to arbitrate**, so a flat result reflects a saturated candidate pool rather than a method that does not work.
- Cases verified out of cases gathered: 16/16

### Arbitration behaviour

- Coordinator rulings: 16 decided, 0 unsure (the spec did not determine the answer)
- Replies we could not parse: 0
- Rulings that agree with the benchmark's reference solution (analysis only): 11/16 (68.8%)
- Testing agent's expected values that were actually correct: 30/42 (71.4%)
- Docstring example cases used as trusted evidence: 12
- Repairs attempted: 3; accepted (verified score improved): 1

### Failure and security audit

- Candidates flagged by the static security check: 0
- Final answers flagged, by strategy: 0 in total (solo: 0 tasks, debate: 0 tasks)

## Per-task results

| Task | `solo` | `first` | `vote` | `debate` | `tests_only` | `arbitration_select` | `arbitration` | oracle |
|---|---|---|---|---|---|---|---|---|
| HumanEval/10 | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/18 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/24 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/25 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/35 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/98 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/107 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## How to read this

- This is a **pilot** with 7 tasks: the confidence intervals are wide and a p-value above 0.05 means the run cannot distinguish the strategies yet. Its job is to show the pipeline works end to end and to estimate cost per task.
- Agents ran on a small local model, not Claude. Absolute pass rates are lower than Claude's; the headline experiment reruns this config with Claude.
- Rulings and Testing-agent accuracy are scored against the benchmark's reference solution purely for analysis; no agent sees the reference or the hidden tests.
