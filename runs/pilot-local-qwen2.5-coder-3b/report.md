# HiveMind run report: `pilot-local-qwen2.5-coder-3b`

Generated 2026-10-06 from `pilot-local-qwen2.5-coder-3b/`. Benchmark: **HumanEval**, 5 tasks (seeded random sample, sample_seed=0). Graded by the benchmark's hidden unit tests, which no agent ever sees.

## Models used

| Provider | Model | Roles | Calls | Failed calls | Input tokens | Output tokens | Avg latency | Cost (USD) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ollama | `qwen2.5-coder:3b` | coding, coordinator, research, testing | 55 | 0 | 26,189 | 15,883 | 8.9s | $0.00 |

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
| `solo` | 5/5 | 100.0% | 56.6%–100.0% | 1.0 | 364 |
| `first` | 4/5 | 80.0% | 37.6%–96.4% | 2.0 | 1,111 |
| `vote` | 5/5 | 100.0% | 56.6%–100.0% | 5.0 | 2,845 |
| `debate` | 5/5 | 100.0% | 56.6%–100.0% | 8.0 | 7,058 |
| `tests_only` | 5/5 | 100.0% | 56.6%–100.0% | 5.0 | 2,845 |
| `arbitration_select` | 5/5 | 100.0% | 56.6%–100.0% | 5.6 | 3,083 |
| `arbitration` | 5/5 | 100.0% | 56.6%–100.0% | 6.0 | 3,465 |

**Oracle** (at least one of the 3 candidates is correct): 5/5 (100.0%). This is the ceiling for any strategy that only *selects* a candidate; repair and debate can exceed it.

## Arbitration vs. each baseline (paired, same tasks)

| Baseline | Arbitration right, baseline wrong | Baseline right, arbitration wrong | Exact McNemar p |
|---|---:|---:|---:|
| `solo` | 0 | 0 | 1.000 |
| `first` | 1 | 0 | 1.000 |
| `vote` | 0 | 0 | 1.000 |
| `debate` | 0 | 0 | 1.000 |
| `tests_only` | 0 | 0 | 1.000 |
| `arbitration_select` | 0 | 0 | 1.000 |

## Evidence diagnostics

- Tasks where candidates behaved differently: 1/5
- Tasks needing a ruling: 3/5; disputed cases: 9
- Coordinator rulings: 9 decided, 0 unsure
- Rulings that agree with the benchmark's reference solution (analysis only): 7/9 (77.8%)
- Testing agent's expected values that were actually correct: 21/26 (80.8%)
- Docstring example cases used as trusted evidence: 12
- Repairs attempted: 2; accepted (verified score improved): 1
- Candidates flagged by the static security check: 0

## Per-task results

| Task | `solo` | `first` | `vote` | `debate` | `tests_only` | `arbitration_select` | `arbitration` | oracle |
|---|---|---|---|---|---|---|---|---|
| HumanEval/10 | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/18 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/24 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/25 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/35 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## How to read this

- This is a **pilot** with 5 tasks: the confidence intervals are wide and a p-value above 0.05 means the run cannot distinguish the strategies yet. Its job is to show the pipeline works end to end and to estimate cost per task.
- Agents ran on a small local model, not Claude. Absolute pass rates are lower than Claude's; the headline experiment reruns this config with Claude.
- Rulings and Testing-agent accuracy are scored against the benchmark's reference solution purely for analysis; no agent sees the reference or the hidden tests.
