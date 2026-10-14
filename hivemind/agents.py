"""The specialist agents. Each one is a role prompt bound to a model.

Roles mirror a human engineering team. The coordinator decides which roles act
on a task; in the software-engineering benchmark the research, coding, testing
and security roles plus the coordinator take part, while documentation and
ethics run in review mode (`hivemind demo`).
"""

from __future__ import annotations

import threading
from dataclasses import InitVar, dataclass, field
from typing import Callable

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


class Cancelled(Exception):
    """The run was cancelled; raised before the next model call."""


@dataclass
class Transcript:
    """Every message exchanged while solving one task, for inspection and replay."""

    task_id: str
    turns: list[dict] = field(default_factory=list)
    # Not fields, so asdict() and saved transcripts are unchanged.
    on_event: InitVar[Callable[[dict], None] | None] = None
    cancel: InitVar[threading.Event | None] = None

    def __post_init__(self, on_event, cancel) -> None:
        self._on_event = on_event
        self._cancel = cancel

    def add(self, **turn) -> None:
        self.turns.append(turn)

    def emit(self, type: str, **data) -> None:
        """Tell a live listener (the web UI) what just happened; a no-op for CLI runs."""
        if self._on_event is not None:
            self._on_event({"type": type, "task_id": self.task_id, **data})

    def check_cancelled(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise Cancelled(self.task_id)


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
        transcript.check_cancelled()
        record = dict(task_id=transcript.task_id, phase=phase, role=self.role,
                      provider=self.llm.provider, model=self.llm.model)
        transcript.emit("turn_started", phase=phase, role=self.role,
                        provider=self.llm.provider, model=self.llm.model)
        try:
            result = self.llm.complete(self.system, prompt, seed=seed)
        except Exception as exc:  # a failed call is recorded and treated as an empty answer
            error = f"{type(exc).__name__}: {exc}"
            self.ledger.add(CallRecord(**record, input_tokens=0, output_tokens=0,
                                       latency_s=0.0, error=error))
            transcript.add(phase=phase, role=self.role, prompt=prompt, response="",
                           error=str(exc))
            transcript.emit("turn", phase=phase, role=self.role, provider=self.llm.provider,
                            model=self.llm.model, prompt=prompt, response="", input_tokens=0,
                            output_tokens=0, latency_s=0.0, error=error)
            return ""
        # Record the model that actually served the call (it differs after a fallback).
        self.ledger.add(CallRecord(**{**record, "model": result.model},
                                   input_tokens=result.input_tokens,
                                   output_tokens=result.output_tokens,
                                   latency_s=result.latency_s))
        transcript.add(phase=phase, role=self.role, prompt=prompt, response=result.text)
        transcript.emit("turn", phase=phase, role=self.role, provider=self.llm.provider,
                        model=result.model, prompt=prompt, response=result.text,
                        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                        latency_s=result.latency_s, error=None)
        return result.text


class Team:
    """One agent per role. Agents and coordinator may use different models, and
    `role_llms` can give any agent role a model of its own."""

    def __init__(self, agent_llm: LLM, coordinator_llm: LLM, ledger: Ledger,
                 role_llms: dict[str, LLM] | None = None):
        role_llms = role_llms or {}
        unknown = set(role_llms) - set(ROLE_PROMPTS) | ({"coordinator"} & set(role_llms))
        if unknown:
            raise ValueError(f"cannot override roles {sorted(unknown)}")
        self.ledger = ledger
        self.agents = {role: Agent(role, role_llms.get(role, agent_llm), ledger)
                       for role in ROLE_PROMPTS if role != "coordinator"}
        self.agents["coordinator"] = Agent("coordinator", coordinator_llm, ledger)

    def __getitem__(self, role: str) -> Agent:
        return self.agents[role]
