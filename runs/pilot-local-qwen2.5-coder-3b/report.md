# HiveMind run report: `pilot-local-qwen2.5-coder-3b`

Generated 2026-10-15 from `pilot-local-qwen2.5-coder-3b/`. Benchmark: **HumanEval**, 30 tasks (seeded random sample, sample_seed=0). Graded by the benchmark's hidden unit tests, which no agent ever sees.

## Models used

| Provider | Model | Roles | Calls | Failed calls | Input tokens | Output tokens | Avg latency | Cost (USD) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ollama | `qwen2.5-coder:3b` | coding, coordinator, research, testing | 341 | 0 | 187,055 | 98,798 | 12.3s | $0.00 |

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
| `solo` | 25/30 | 83.3% | 66.4%–92.7% | 1.0 | 443 |
| `first` | 20/30 | 66.7% | 48.8%–80.8% | 2.0 | 1,173 |
| `vote` | 24/30 | 80.0% | 62.7%–90.5% | 5.0 | 3,103 |
| `debate` | 24/30 | 80.0% | 62.7%–90.5% | 8.0 | 7,676 |
| `tests_only` | 27/30 | 90.0% | 74.4%–96.5% | 5.0 | 3,103 |
| `arbitration_select` | 24/30 | 80.0% | 62.7%–90.5% | 5.8 | 3,539 |
| `arbitration` | 25/30 | 83.3% | 66.4%–92.7% | 6.4 | 4,088 |

**Oracle** (at least one of the 3 candidates is correct): 27/30 (90.0%). This is the ceiling for any strategy that only *selects* a candidate; repair and debate can exceed it.

## Arbitration vs. each baseline (paired, same tasks)

| Baseline | Arbitration right, baseline wrong | Baseline right, arbitration wrong | Exact McNemar p |
|---|---:|---:|---:|
| `solo` | 3 | 3 | 1.000 |
| `first` | 7 | 2 | 0.180 |
| `vote` | 3 | 2 | 1.000 |
| `debate` | 2 | 1 | 1.000 |
| `tests_only` | 0 | 2 | 0.500 |
| `arbitration_select` | 1 | 0 | 1.000 |

## Evidence diagnostics

- Tasks where candidates behaved differently: 15/30
- Tasks needing a ruling: 24/30; disputed cases: 81
- Coordinator rulings: 76 decided, 5 unsure
- Rulings that agree with the benchmark's reference solution (analysis only): 36/76 (47.4%)
- Testing agent's expected values that were actually correct: 171/217 (78.8%)
- Docstring example cases used as trusted evidence: 23
- Repairs attempted: 17; accepted (verified score improved): 2
- Candidates flagged by the static security check: 0

## Per-task results

| Task | `solo` | `first` | `vote` | `debate` | `tests_only` | `arbitration_select` | `arbitration` | oracle |
|---|---|---|---|---|---|---|---|---|
| HumanEval/10 | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/18 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/24 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/25 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/35 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/37 | ✓ | · | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/52 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/55 | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ |
| HumanEval/64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/66 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/72 | ✓ | · | ✓ | · | ✓ | ✓ | ✓ | ✓ |
| HumanEval/77 | · | · | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/79 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/80 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/84 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/90 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/91 | · | · | · | · | · | · | · | · |
| HumanEval/98 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/103 | ✓ | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/107 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/111 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/120 | ✓ | · | · | · | · | · | · | · |
| HumanEval/122 | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | ✓ |
| HumanEval/124 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/129 | · | · | · | ✓ | ✓ | · | ✓ | ✓ |
| HumanEval/130 | · | · | · | · | · | · | · | · |
| HumanEval/136 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/142 | ✓ | ✓ | ✓ | · | ✓ | · | · | ✓ |
| HumanEval/149 | · | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HumanEval/150 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## How to read this

- This is a **pilot** with 30 tasks: the confidence intervals are wide and a p-value above 0.05 means the run cannot distinguish the strategies yet. Its job is to show the pipeline works end to end and to estimate cost per task.
- Agents ran on a small local model, not Claude. Absolute pass rates are lower than Claude's; the headline experiment reruns this config with Claude.
- Rulings and Testing-agent accuracy are scored against the benchmark's reference solution purely for analysis; no agent sees the reference or the hidden tests.
