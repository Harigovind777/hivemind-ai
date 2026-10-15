"""Chat threads: messages, drafts, rounds, follow-ups, saving and reloading."""

import threading

import pytest

from hivemind.web.chats import Chat, ChatBusy, ChatError, load_chats

from chat_helpers import CHANGED_SPEC, SPEC, chat_coordinator, chat_respond


def settle(chat, timeout=20):
    chat.thread.join(timeout)
    assert not chat.thread.is_alive()
    return chat


def kinds(chat):
    return [e["type"] for e in chat.log.events_after(0)]


def last(chat, kind):
    return [e for e in chat.log.events_after(0) if e["type"] == kind][-1]


def gated(gate, respond_fn=chat_respond):
    """A team whose every model call waits for `gate`."""
    return lambda: chat_coordinator(lambda s, p: (gate.wait(10), respond_fn(s, p))[1])


@pytest.fixture
def chat(tmp_path):
    return Chat.create(tmp_path / "chats", "pilot-local")


def answered(chat):
    chat.send("add one to a number", chat_coordinator)
    settle(chat)
    chat.go(SPEC, last(chat, "spec")["id"], chat_coordinator)
    settle(chat)
    return chat


def test_a_first_message_is_drafted_into_a_spec(chat):
    assert chat.send("add one to a number", chat_coordinator) == "m1"
    settle(chat)
    assert kinds(chat)[:3] == ["message", "action", "turn_started"]
    spec = last(chat, "spec")
    assert spec["text"] == SPEC and spec["entry_point"] == "add_one" and spec["examples"] == 1
    assert spec["error"] is None and last(chat, "done") == {**last(chat, "done"), "action": "message",
                                                             "status": "done"}
    assert "classified" not in kinds(chat)  # nothing answered yet, so nothing to classify


def test_go_runs_a_round_and_records_the_answer(chat):
    chat.send("add one to a number", chat_coordinator)
    settle(chat)
    spec_id = last(chat, "spec")["id"]
    assert chat.go(SPEC, spec_id, chat_coordinator) == 1
    settle(chat)
    started = last(chat, "round_started")
    assert (started["spec_id"], started["spec"], started["round"]) == (spec_id, SPEC, 1)
    answer = last(chat, "answer")["answer"]
    assert answer["source"] == "candidate 1" and answer["verified_passed"] == answer["verified_total"]
    phases = [e["phase"] for e in chat.log.events_after(0) if e["type"] == "turn"]
    assert "solo" not in phases and "judge" not in phases and "arbiter" in phases


def test_a_follow_up_question_is_answered(chat):
    answered(chat).send("why candidate 1?", chat_coordinator)
    settle(chat)
    assert last(chat, "classified") == {**last(chat, "classified"), "reading": "question",
                                         "automatic": True}
    reply = last(chat, "question_answer")
    assert reply["round"] == 1 and reply["text"] == "Candidate 1 matched every verified case."


def test_a_follow_up_change_drafts_an_updated_spec(chat):
    answered(chat).send("please also handle None", chat_coordinator)
    settle(chat)
    assert last(chat, "classified")["reading"] == "change"
    assert last(chat, "spec")["text"] == CHANGED_SPEC


def test_reinterpret_switches_a_question_to_a_change_and_back(chat):
    answered(chat).send("why candidate 1?", chat_coordinator)
    settle(chat)
    message_id = last(chat, "message")["id"]
    chat.reinterpret(message_id, chat_coordinator)
    settle(chat)
    assert last(chat, "classified") == {**last(chat, "classified"), "reading": "change",
                                         "automatic": False}
    spec_id = last(chat, "spec")["id"]
    chat.reinterpret(message_id, chat_coordinator)
    settle(chat)
    assert last(chat, "spec_superseded")["spec_id"] == spec_id
    assert last(chat, "classified")["reading"] == "question"
    with pytest.raises(ChatError):
        chat.go(SPEC, spec_id, chat_coordinator)  # that spec was set aside


def test_a_change_that_already_ran_cannot_become_a_question(chat):
    answered(chat).send("please also handle None", chat_coordinator)
    settle(chat)
    chat.go(CHANGED_SPEC, last(chat, "spec")["id"], chat_coordinator)
    settle(chat)
    with pytest.raises(ChatError):
        chat.reinterpret(last(chat, "message")["id"], chat_coordinator)


def test_one_action_at_a_time(chat):
    gate = threading.Event()
    chat.send("add one", gated(gate))
    with pytest.raises(ChatBusy):
        chat.send("again", gated(gate))
    assert kinds(chat).count("message") == 1
    gate.set()
    settle(chat)


def test_cancel_stops_at_the_next_model_call(chat):
    gate = threading.Event()
    chat.send("add one", gated(gate, lambda s, p: "not a spec"))  # invalid, so a retry call follows
    chat.cancel()
    gate.set()
    settle(chat)
    assert last(chat, "done")["status"] == "cancelled" and chat.busy is None
    assert "spec" not in kinds(chat)


def test_a_crash_is_reported_and_can_be_retried(chat):
    def broken():
        raise RuntimeError("model config is broken")

    chat.send("add one", broken)
    settle(chat)
    assert last(chat, "error")["message"] == "RuntimeError: model config is broken"
    assert last(chat, "done")["status"] == "failed" and chat.retryable()["action"] == "message"
    chat.retry(chat_coordinator)
    settle(chat)
    assert last(chat, "spec")["text"] == SPEC and kinds(chat).count("message") == 1
    with pytest.raises(ChatError):
        chat.retry(chat_coordinator)  # the retry worked, so nothing is left to retry


def test_a_chat_is_saved_and_reloaded_with_its_title(chat, tmp_path):
    chat.send("Add one to a number — and keep it fast ✨ " + "x" * 80, chat_coordinator)
    settle(chat)
    [loaded] = load_chats(tmp_path / "chats")
    assert loaded.id == chat.id and loaded.config == "pilot-local"
    assert loaded.log.events_after(0) == chat.log.events_after(0)
    assert loaded.title.startswith("Add one to a number — and keep it fast ✨")
    assert len(loaded.title) == 60 and loaded.title.endswith("…")


def test_a_chat_left_running_by_a_restart_is_marked_interrupted(chat, tmp_path):
    gate = threading.Event()
    chat.send("add one", gated(gate))
    [loaded] = load_chats(tmp_path / "chats")  # as if the server restarted mid-draft
    gate.set()
    settle(chat)
    done = last(loaded, "done")
    assert (done["status"], done["action"], loaded.busy) == ("interrupted", "message", None)
    loaded.retry(chat_coordinator)
    settle(loaded)
    assert last(loaded, "spec")["text"] == SPEC


def test_unreadable_chat_files_are_skipped(tmp_path):
    (tmp_path / "chats").mkdir()
    (tmp_path / "chats" / "broken.json").write_text("{not json")
    assert load_chats(tmp_path / "chats") == []


def failing_build():
    def respond(system, prompt):
        raise TimeoutError("timed out")
    return chat_coordinator(respond)


def test_a_failed_model_call_fails_the_action_so_it_can_be_retried(chat):
    chat.send("add one", failing_build)
    settle(chat)
    assert "model call failed" in last(chat, "error")["message"]
    assert last(chat, "done")["status"] == "failed" and "spec" not in kinds(chat)
    chat.retry(chat_coordinator)
    settle(chat)
    assert last(chat, "spec")["text"] == SPEC


def test_a_failed_classification_is_not_recorded_as_a_reading(chat):
    answered(chat).send("why candidate 1?", failing_build)
    settle(chat)
    assert "classified" not in kinds(chat) and "question_answer" not in kinds(chat)
    assert last(chat, "done")["status"] == "failed"
