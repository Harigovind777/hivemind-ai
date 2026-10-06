"""Model providers behind one small interface, plus a ledger of every call.

Providers:
  anthropic  Claude via the official SDK (main experiments; needs credentials)
  ollama     local open models over Ollama's HTTP API (free pilot runs)
  scripted   deterministic fake used by the test suite
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLM(Protocol):
    provider: str
    model: str

    def complete(self, system: str, prompt: str, *, seed: int = 0) -> Completion: ...


class RefusalError(RuntimeError):
    """The model declined the request (stop_reason == "refusal") even after fallback."""


class AnthropicLLM:
    """Claude through the Messages API.

    Opus 5.5 rejects sampling parameters and cannot disable thinking, so depth is
    controlled with `effort` and diversity between agents comes from their prompts.
    """

    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = "medium",
                 max_tokens: int = 16000):
        import anthropic  # imported lazily so local/scripted runs need no credentials

        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(max_retries=4)

    def complete(self, system: str, prompt: str, *, seed: int = 0) -> Completion:
        kwargs = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        t0 = time.perf_counter()
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            # Route classifier false positives to Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise RefusalError(f"{self.model} refused: {response.stop_details}")
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=time.perf_counter() - t0,
        )


class OllamaLLM:
    """A local model served by Ollama (https://ollama.com)."""

    provider = "ollama"

    def __init__(self, model: str = "qwen2.5-coder:3b", temperature: float = 0.7,
                 max_tokens: int = 1024, host: str = "http://localhost:11434",
                 timeout_s: float = 300):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def complete(self, system: str, prompt: str, *, seed: int = 0) -> Completion:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens,
                        "seed": seed},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read())
        return Completion(
            text=data["message"]["content"],
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_s=time.perf_counter() - t0,
        )


class ScriptedLLM:
    """Answers with a user-supplied function of (system, prompt). For tests only."""

    provider = "scripted"

    def __init__(self, respond: Callable[[str, str], str], model: str = "scripted"):
        self.model = model
        self._respond = respond

    def complete(self, system: str, prompt: str, *, seed: int = 0) -> Completion:
        text = self._respond(system, prompt)
        return Completion(text=text, model=self.model, input_tokens=len(prompt) // 4,
                          output_tokens=len(text) // 4, latency_s=0.0)


def make_llm(provider: str, model: str, **options) -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model=model, **options)
    if provider == "ollama":
        return OllamaLLM(model=model, **options)
    raise ValueError(f"unknown provider {provider!r} (expected 'anthropic' or 'ollama')")


@dataclass
class CallRecord:
    task_id: str
    phase: str          # pipeline step, e.g. "candidate:1", "tests", "arbiter"
    role: str           # agent role that made the call
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    error: str | None = None


@dataclass
class Ledger:
    """Thread-safe log of every model call, used for cost and procedure reporting."""

    records: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, record: CallRecord) -> None:
        with self._lock:
            self.records.append(record)

    def for_task(self, task_id: str) -> list[CallRecord]:
        with self._lock:
            return [r for r in self.records if r.task_id == task_id]

    def dump(self, path) -> None:
        with self._lock, open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r)) + "\n")
