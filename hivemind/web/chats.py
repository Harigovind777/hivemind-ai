"""A chat with the team: your messages, spec drafts, team rounds and answers, as one thread.

The thread is its event list. Every event is kept, and the chat is saved to
runs/chats/<id>.json after each one, so the page rebuilds the whole conversation by
replaying it, the same way a live run replays. One action (reading a message or
running a round) happens at a time, in a background thread.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .. import chat as talk
from ..agents import Cancelled, Transcript
from ..config import is_safe_name
from ..coordinator import Coordinator
from .events import EventLog

Build = Callable[[], Coordinator]


class ChatError(ValueError):
    """A request the chat cannot carry out; the message says why."""


class ChatBusy(ChatError):
    """The team is still working on this chat's previous action."""


class Chat:
    def __init__(self, chat_id: str, path: Path, config: str, created: float | None = None,
                 events: list[dict] | None = None):
        self.id = chat_id
        self.path = Path(path)
        self.config = config
        self.created = created or time.time()
        self.log = EventLog(events)
        self.busy: str | None = None
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ files

    @classmethod
    def create(cls, chats_dir: Path, config: str) -> "Chat":
        chat_id = uuid.uuid4().hex[:12]
        chat = cls(chat_id, Path(chats_dir) / f"{chat_id}.json", config)
        chat.save()
        return chat

    @classmethod
    def load(cls, path: Path) -> "Chat":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chat = cls(data["id"], path, data["config"], data["created"], data["events"])
        if data.get("busy"):  # the server stopped while the team was working
            chat.log.emit({"type": "done", "action": data["busy"], "status": "interrupted"})
            chat.save()
        return chat

    def save(self) -> None:
        """Write the whole thread. Each write has its own temp file, so two writers (a reloaded
        copy, a second server) never trip over each other; the last one wins."""
        with self._save_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"id": self.id, "config": self.config, "created": self.created,
                    "busy": self.busy, "events": self.log.events_after(0)}
            tmp = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self.path)

    # ------------------------------------------------------------------ reading

    @property
    def title(self) -> str:
        first = next((e["text"] for e in self.log.events_after(0) if e["type"] == "message"), "")
        first = " ".join(first.split())
        return first if len(first) <= 60 else first[:59].rstrip() + "…"

    @property
    def updated(self) -> float:
        events = self.log.events_after(0)
        return events[-1]["time"] if events else self.created

    @property
    def finished(self) -> bool:
        """A chat's event stream never ends on its own; the page closes it."""
        return False

    def events_after(self, after: int) -> list[dict]:
        return self.log.events_after(after)

    def wait_events(self, after: int, timeout: float) -> list[dict]:
        return self.log.wait_events(after, timeout)

    def summary(self) -> dict:
        return {"kind": "chat", "id": self.id, "title": self.title, "config": self.config,
                "updated": self.updated, "busy": self.busy}

    def to_dict(self) -> dict:
        return {**self.summary(), "created": self.created, "events": self.log.events_after(0)}

    def _of(self, kind: str) -> list[dict]:
        return [e for e in self.log.events_after(0) if e["type"] == kind]

    def _find(self, kind: str, **match) -> dict | None:
        found = [e for e in self._of(kind) if all(e.get(k) == v for k, v in match.items())]
        return found[-1] if found else None

    def _latest_round(self) -> dict | None:
        """The newest round that has an answer: its spec, its answer and the questions so far."""
        answer = self._find("answer")
        if answer is None:
            return None
        started = self._find("round_started", round=answer["round"])
        questions = [(self._find("message", id=q["message_id"])["text"], q["text"])
                     for q in self._of("question_answer") if q["round"] == answer["round"]]
        return {"round": answer["round"], "spec": started["spec"], "answer": answer["answer"],
                "questions": questions}

    def _current_spec(self) -> str | None:
        """What a change builds on: the newest draft not set aside, or the newest spec that ran."""
        set_aside = {e["spec_id"] for e in self._of("spec_superseded")}
        specs = [(e["seq"], e["text"]) for e in self._of("spec") if e["id"] not in set_aside]
        specs += [(e["seq"], e["spec"]) for e in self._of("round_started")]
        return max(specs)[1] if specs else None

    # ------------------------------------------------------------------ actions

    def _emit(self, event: dict) -> None:
        self.log.emit(event)
        self.save()

    def _start(self, action: str, params: dict, work: Callable[[threading.Event], None],
               first: list[dict] = ()) -> None:
        """Record `first` and the action, then run `work` in the background, one at a time."""
        with self._lock:
            if self.busy:
                raise ChatBusy("The team is still working on this chat. Wait for it, or cancel it.")
            self.busy = action
            self._cancel = cancel = threading.Event()
            for event in first:
                self._emit(event)
            self._emit({"type": "action", "action": action, **params})

        def run() -> None:
            status = "done"
            try:
                work(cancel)
            except Cancelled:
                status = "cancelled"
            except Exception as exc:  # shown in the thread; the server keeps running
                self._emit({"type": "error", "action": action,
                            "message": f"{type(exc).__name__}: {exc}",
                            "trace": traceback.format_exc(limit=5)})
                status = "failed"
            with self._lock:
                self.busy = None
                self._emit({"type": "done", "action": action, "status": status})

        self.thread = threading.Thread(target=run, name=f"chat-{self.id}", daemon=True)
        self.thread.start()

    def send(self, text: str, build: Build) -> str:
        message_id = f"m{len(self._of('message')) + 1}"
        self._start("message", {"message_id": message_id},
                    lambda cancel: self._handle(message_id, None, build, cancel),
                    first=[{"type": "message", "id": message_id, "text": text}])
        return message_id

    def go(self, spec_text: str, spec_id: str, build: Build) -> int:
        if self._find("spec", id=spec_id) is None:
            raise ChatError(f"No spec {spec_id!r} in this chat")
        if self._find("spec_superseded", spec_id=spec_id):
            raise ChatError("That spec was set aside; use the newest one")
        spec = talk.validate_spec(spec_text)
        round_no = len(self._of("round_started")) + 1
        task_id = f"chat/{self.id}/{round_no}"
        self._start("round", {"round": round_no, "spec_id": spec_id},
                    lambda cancel: self._round(round_no, spec, task_id, build, cancel),
                    first=[{"type": "round_started", "round": round_no, "spec_id": spec_id,
                            "spec": spec.text, "task_id": task_id,
                            "entry_point": spec.entry_point}])
        return round_no

    def reinterpret(self, message_id: str, build: Build) -> None:
        classified = self._find("classified", message_id=message_id)
        if classified is None:
            raise ChatError("Only a message the Coordinator read as a question or a change "
                            "can be switched")
        spec = self._find("spec", message_id=message_id)
        if spec and self._find("round_started", spec_id=spec["id"]):
            raise ChatError("That change already ran, so it cannot be read as a question now")
        reading = "change" if classified["reading"] == "question" else "question"
        first = [{"type": "classified", "message_id": message_id, "reading": reading,
                  "automatic": False}]
        if reading == "question" and spec:
            first.insert(0, {"type": "spec_superseded", "spec_id": spec["id"]})
        self._start("reinterpret", {"message_id": message_id, "reading": reading},
                    lambda cancel: self._handle(message_id, reading, build, cancel), first=first)

    def retryable(self) -> dict | None:
        """The last action, when it failed, was cancelled or was cut off by a restart."""
        done = self._find("done")
        if self.busy or done is None or done["status"] == "done":
            return None
        return self._find("action")

    def retry(self, build: Build) -> None:
        action = self.retryable()
        if action is None:
            raise ChatError("There is nothing to retry")
        if action["action"] == "round":
            started = self._find("round_started", round=action["round"])
            self.go(started["spec"], started["spec_id"], build)
            return
        reading = action.get("reading")
        params = {k: action[k] for k in ("message_id", "reading") if k in action}
        self._start(action["action"], params,
                    lambda cancel: self._handle(action["message_id"], reading, build, cancel))

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------ work

    def _transcript(self, cancel: threading.Event) -> Transcript:
        return Transcript(f"chat/{self.id}", on_event=self._emit, cancel=cancel)

    def _handle(self, message_id: str, reading: str | None, build: Build,
                cancel: threading.Event) -> None:
        """Read one message: a problem or change gets a spec draft, a question an answer."""
        text = self._find("message", id=message_id)["text"]
        coordinator = build()
        team, seed, tr = coordinator.team, coordinator.seed, self._transcript(cancel)
        latest = self._latest_round()
        if latest is None:
            reading = "change"  # nothing answered yet: every message describes the problem
        elif reading is None:
            reading = talk.classify(team, text, talk.round_context(latest["spec"], latest["answer"]),
                                    tr, seed=seed)
            self._emit({"type": "classified", "message_id": message_id, "reading": reading,
                        "automatic": True})
        if reading == "question":
            reply = talk.answer_question(team, text,
                                         talk.round_context(latest["spec"], latest["answer"]),
                                         latest["questions"], tr, seed=seed)
            self._emit({"type": "question_answer", "message_id": message_id,
                        "round": latest["round"], "text": reply})
            return
        draft = talk.draft_spec(team, text, tr, current_spec=self._current_spec(), seed=seed)
        self._emit({"type": "spec", "id": f"s{len(self._of('spec')) + 1}", "message_id": message_id,
                    "text": draft.text,
                    "entry_point": draft.spec.entry_point if draft.spec else None,
                    "examples": draft.spec.examples if draft.spec else 0, "error": draft.error})

    def _round(self, round_no: int, spec: talk.Spec, task_id: str, build: Build,
               cancel: threading.Event) -> None:
        answer, _ = build().answer(talk.custom_task(spec, task_id), on_event=self._emit,
                                   cancel=cancel)
        self._emit({"type": "answer", "round": round_no, "answer": asdict(answer)})


def load_chats(chats_dir: Path) -> list[Chat]:
    """Every saved chat, newest first. Files that cannot be read are skipped."""
    chats = []
    for path in Path(chats_dir).glob("*.json"):
        if not is_safe_name(path.stem):
            continue
        try:
            chats.append(Chat.load(path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(chats, key=lambda c: c.updated, reverse=True)
