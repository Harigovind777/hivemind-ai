"""The specialist agents. Each one is a role prompt bound to a model.

Roles mirror a human engineering team. The coordinator decides which roles act
on a task; in the software-engineering benchmark the research, coding, testing
and security roles plus the coordinator take part, while documentation and
ethics run in review mode (`hivemind demo`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLM, CallRecord, Ledger

ROLE_PROMPTS = {
    "research": (
        "You are the Research agent on an expert software team. You analyse a task "
        "specification before anyone writes code. State the exact input/output "
        "contract, the edge cases, and any ambiguity together with the most likely "
        "intended reading. Be brief (under 150 words). Do not write code."
    ),
    "coding": (
        "You are the Coding agent on an expert software team. Write a correct, "
        "self-contained Python implementation of the requested function. Include any "
        "imports and helper functions it needs. End your reply with exactly one "
        "```python code block containing the complete function."
    ),
    "testing": (
        "You are the Testing agent on an expert software team. You write tests from "
        "the specification alone; you never see the implementation. Derive every "
        "expected value carefully from the specification and its examples."
    ),
    "security": (
        "You are the Security agent on an expert software team. Review code for "
        "unsafe behaviour: shell or process execution, file or network access, "
        "dynamic code evaluation, unbounded resource use. Reply 'NO ISSUES' or list "
        "each issue on its own line."
    ),
    "documentation": (
        "You are the Documentation agent on an expert software team. Write a concise "
        "description of the final solution: what it does, its complexity, and one "
        "usage example. Under 120 words."
    ),
    "ethics": (
        "You are the Ethics agent on an expert software team. Review the task and "
        "solution for ethical risks: unfair bias, privacy, safety, or misuse. For "
        "ordinary algorithmic code, answer 'No concerns' with one sentence of reasoning."
    ),
    "coordinator": (
        "You are the Coordinator of an expert AI team. You settle disagreements "
        "between team members. Decide only on evidence: the task specification, its "
        "worked examples, and the concrete execution results you are shown. Never "
        "side with an answer because more members gave it."
    ),
}


@dataclass
class Transcript:
    """Every message exchanged while solving one task, for inspection and replay."""

    task_id: str
    turns: list[dict] = field(default_factory=list)

    def add(self, **turn) -> None:
        self.turns.append(turn)


class Agent:
    def __init__(self, role: str, llm: LLM, ledger: Ledger):
        if role not in ROLE_PROMPTS:
            raise ValueError(f"unknown role {role!r}")
        self.role = role
        self.llm = llm
        self.ledger = ledger

    @property
    def system(self) -> str:
        return ROLE_PROMPTS[self.role]

    def ask(self, prompt: str, *, transcript: Transcript, phase: str, seed: int = 0) -> str:
        record = dict(task_id=transcript.task_id, phase=phase, role=self.role,
                      provider=self.llm.provider, model=self.llm.model)
        try:
            result = self.llm.complete(self.system, prompt, seed=seed)
        except Exception as exc:  # a failed call is recorded and treated as an empty answer
            self.ledger.add(CallRecord(**record, input_tokens=0, output_tokens=0,
                                       latency_s=0.0, error=f"{type(exc).__name__}: {exc}"))
            transcript.add(phase=phase, role=self.role, prompt=prompt, response="",
                           error=str(exc))
            return ""
        # Record the model that actually served the call (it differs after a fallback).
        self.ledger.add(CallRecord(**{**record, "model": result.model},
                                   input_tokens=result.input_tokens,
                                   output_tokens=result.output_tokens,
                                   latency_s=result.latency_s))
        transcript.add(phase=phase, role=self.role, prompt=prompt, response=result.text)
        return result.text


class Team:
    """One agent per role. Agents and coordinator may use different models."""

    def __init__(self, agent_llm: LLM, coordinator_llm: LLM, ledger: Ledger):
        self.ledger = ledger
        self.agents = {role: Agent(role, agent_llm, ledger)
                       for role in ROLE_PROMPTS if role != "coordinator"}
        self.agents["coordinator"] = Agent("coordinator", coordinator_llm, ledger)

    def __getitem__(self, role: str) -> Agent:
        return self.agents[role]
