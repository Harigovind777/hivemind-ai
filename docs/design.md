# HiveMind AI: research design

Status: initial stage implemented (2026-10-06). Owner: Harigovind.

## Problem

Most AI agents work alone. Multi-agent systems such as MetaGPT, ChatDev, AutoGen,
AgentVerse and multi-agent debate (Du et al., 2023) show that teams of LLM agents
can beat a single agent. However, they usually settle disagreements with a fixed
pipeline, a majority vote, or free-form argument, and all three can be wrong
together: a confident majority can share the same bug, and a persuasive argument
is not proof.

## Research question and claim

> When specialist agents disagree, does settling the disagreement with
> **checkable evidence** beat majority vote and free-form debate?

Hypotheses, tested on the same candidate pools:

- **H1** `arbitration` has a higher pass rate than `vote` and `debate`.
- **H2** `arbitration_select` beats `tests_only`. In other words, it helps to
  check the Testing agent's claims instead of trusting them.
- **H3** These gains come at a comparable or lower cost (model calls and tokens
  per task) than debate.

## System

```
                         ┌──────────────────────────────┐
   task spec  ──────────▶│          COORDINATOR         │──▶ final answer
                         │ routes work, rules on disputes│
                         └──────────────┬───────────────┘
        ┌──────────────┬────────────────┼───────────────┬──────────────┐
        ▼              ▼                ▼               ▼              ▼
   Research        Coding ×K         Testing        Security     Documentation / Ethics
   (spec brief)   (candidates)   (tests from spec)  (static scan     (review mode)
                                                     + review)
                                        │
                                        ▼
                             Sandbox: run every candidate on every case
                             → execution evidence for the Coordinator
```

| Role | Benchmark runs | Review mode (`hivemind demo`) |
|---|---|---|
| Research | writes the brief every candidate sees | same |
| Coding | solo baseline, K candidates, debate revisions, repair | same |
| Testing | assert-style cases from the spec only | same |
| Security | static AST check (zero cost), gates selection | plus an LLM review of the final answer |
| Documentation | not used (does not affect correctness) | describes the final answer |
| Ethics | not used on coding tasks | reviews task and answer |
| Coordinator | arbiter for disputed cases, judge in debate | same |

### Evidence hierarchy used by arbitration

1. **Specification examples** (the docstring's `>>>` examples) are trusted and
   never overruled.
2. **Coordinator rulings.** A case is *disputed* when the candidates produce
   different results, or all agree but contradict the Testing agent. The
   Coordinator sees the spec and the distinct **real execution outputs**, with
   the options shuffled and no indication of how many agents back each one. It
   then rules which output the spec requires, or answers UNSURE.
3. **Testing-agent expectations** are accepted only when every program that ran
   agrees with them.

Programs that fail to load, hang or crash make no claim about any case, so they
are never offered as options and cannot form a voting bloc. The candidate that
satisfies the most verified cases wins (security flags rank last). If it still
fails a verified case, one **repair** round shows the Coding agent the failing
evidence. The fix is kept only if it satisfies strictly more verified cases.

### Fair comparison

Per task, the solo answer, the research brief, K candidates and the team tests
are generated **once**. Every strategy then resolves that same pool, so
differences between strategies come only from how disagreement is resolved.
Every model call is tagged with its pipeline phase, and each strategy is charged
only for the phases it uses (see `PHASES` in `hivemind/coordinator.py`).

### Guarding against leakage

- The benchmark's hidden tests are used only by `grade()`.
- The reference solution is used only after the fact, to measure how often the
  Coordinator's rulings and the Testing agent's expectations are right. No agent
  ever sees either.
- Candidates are sanitised to imports and definitions, the same convention as
  EvalPlus. This stops a model's own appended asserts from crashing a correct
  function.

## Metrics

- Pass rate (hidden tests) with Wilson 95% intervals.
- Paired exact McNemar test, arbitration vs. each baseline, on the same tasks.
- Oracle@K: the share of tasks where some candidate is correct. This is the
  ceiling for any strategy that only selects.
- Cost: model calls and tokens per task for each strategy, plus USD for Claude.
- Mechanism diagnostics: dispute rate, how often rulings agree with the
  reference, how accurate the Testing agent's expectations are, and repair
  acceptance.

## Threats to validity

- **Headroom.** Frontier Claude models score near the ceiling on HumanEval. If
  solo already passes about 95%, no resolution method can show a gain. The
  mitigations are to report Oracle@K and use harder benchmarks (HumanEval+'s
  extra tests, LiveCodeBench, BigCodeBench), or to run the same pipeline on a
  smaller model where there is room to improve.
- **Contamination.** HumanEval is old and probably in training data. Repeat the
  key result on a newer, contamination-resistant set.
- **Seeds.** Claude Opus 5.5 does not accept sampling parameters, so variation
  across runs comes from the model itself. Repeat runs (3 seeds) and report the
  spread.
- **Small pilots.** A 30-task pilot cannot separate the strategies
  statistically. Its job is to validate the pipeline and estimate cost.

## Roadmap (1 month)

| Week | Goal |
|---|---|
| 1 | ✅ Core framework, coding domain, sandbox, 7 strategies, tests, local pilot. Next: add Claude credentials and run a 10-task cost probe. |
| 2 | Full HumanEval runs on Claude (3 seeds). Add HumanEval+ extra tests to the grader for headroom. |
| 3 | Ablations: K ∈ {3, 5}; arbiter shown majority counts vs. hidden; no repair; arbiter model smaller than agents. Optional: reasoning/Q&A domain (GSM8K-style) to test domain-generality. |
| 4 | Error analysis from transcripts, figures, paper draft. Open-ended reports domain becomes future work. |

## Extending to other domains

A domain supplies tasks, a way to gather evidence, and a grader. For
reasoning/Q&A, the evidence is recomputation: run the candidate's own arithmetic
as code, or check its answer against constraints stated in the problem. For
open-ended reports, the evidence is citations checked against sources. The
coordinator's dispute → evidence → ruling loop stays the same.
