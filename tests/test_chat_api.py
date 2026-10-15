"""The chat endpoints over real HTTP, with a scripted team."""

import json
import threading
import urllib.request

from test_server import call, serve, stream  # noqa: F401  (serve is a fixture)

from chat_helpers import SPEC, chat_coordinator, chat_respond


def until(base, path, stop, timeout=20):
    """Read a chat's event stream until `stop(event)`; chat streams never end on their own."""
    events = []
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().rstrip("\n")
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if stop(events[-1]):
                    return events
    raise AssertionError("the stream ended")


def done(event):
    return event["type"] == "done"


def new_chat(base):
    code, body = call(base, "POST", "/api/chats", {"config": "pilot-local"})
    assert code == 201
    return body["id"]


def drafted(base, chat_id, text="add one to a number"):
    """Send a first message; return its spec event and the seq to read later events after."""
    assert call(base, "POST", f"/api/chats/{chat_id}/messages", {"text": text})[0] == 202
    events = until(base, f"/api/chats/{chat_id}/events", done)
    return next(e for e in events if e["type"] == "spec"), events[-1]["seq"]


def coding_waits_for(gate):
    def respond_fn(system, prompt):
        if system.startswith("You are the Coding"):
            gate.wait(10)
        return chat_respond(system, prompt)
    return lambda config: chat_coordinator(respond_fn)


def test_a_chat_goes_from_message_to_spec_to_answer(serve):
    base, _ = serve(build=lambda config: chat_coordinator())
    chat_id = new_chat(base)
    spec, after = drafted(base, chat_id)
    assert spec["text"] == SPEC

    code, body = call(base, "POST", f"/api/chats/{chat_id}/go",
                      {"spec": spec["text"], "spec_id": spec["id"]})
    assert (code, body) == (202, {"round": 1})
    events = until(base, f"/api/chats/{chat_id}/events?after={after}", done)
    assert [e["type"] for e in events][-1] == "done"
    assert "round_started" in [e["type"] for e in events]
    answer = next(e for e in events if e["type"] == "answer")["answer"]
    assert answer["verified_passed"] == answer["verified_total"] > 0

    code, listing = call(base, "GET", "/api/chats")
    assert code == 200 and listing[0]["id"] == chat_id
    assert listing[0]["title"] == "add one to a number"
    code, whole = call(base, "GET", f"/api/chats/{chat_id}")
    assert code == 200 and [e["type"] for e in whole["events"]][-2:] == ["answer", "done"]


def test_a_second_message_while_the_team_works_is_refused(serve):
    gate = threading.Event()
    base, _ = serve(build=lambda config: chat_coordinator(
        lambda s, p: (gate.wait(10), chat_respond(s, p))[1]))
    chat_id = new_chat(base)
    assert call(base, "POST", f"/api/chats/{chat_id}/messages", {"text": "add one"})[0] == 202
    code, body = call(base, "POST", f"/api/chats/{chat_id}/messages", {"text": "again"})
    assert code == 409 and "still working" in body["error"]
    gate.set()
    events = until(base, f"/api/chats/{chat_id}/events", done)
    assert [e["type"] for e in events].count("message") == 1


def test_go_waits_for_another_job_but_drafting_does_not(serve):
    gate = threading.Event()
    base, _ = serve(build=coding_waits_for(gate))
    live = call(base, "POST", "/api/live", {"config": "pilot-local", "task_id": "Toy/0"})[1]["id"]
    chat_id = new_chat(base)
    spec, _ = drafted(base, chat_id)  # the Research agent does not wait for the live run
    code, body = call(base, "POST", f"/api/chats/{chat_id}/go",
                      {"spec": spec["text"], "spec_id": spec["id"]})
    assert code == 409 and body["current"]["id"] == live
    gate.set()
    stream(base, f"/api/live/{live}/events")


def test_a_running_round_is_the_current_job_and_can_be_cancelled(serve):
    gate = threading.Event()
    base, _ = serve(build=coding_waits_for(gate))
    chat_id = new_chat(base)
    spec, after = drafted(base, chat_id)
    assert call(base, "POST", f"/api/chats/{chat_id}/go",
                {"spec": spec["text"], "spec_id": spec["id"]})[0] == 202
    current = call(base, "GET", "/api/status")[1]["current"]
    assert current["kind"] == "chat" and current["id"] == chat_id
    code, body = call(base, "POST", "/api/live", {"config": "pilot-local", "task_id": "Toy/0"})
    assert code == 409 and "add one to a number" in body["error"]
    assert call(base, "POST", f"/api/chats/{chat_id}/cancel") == (200, {"cancelling": True})
    gate.set()
    events = until(base, f"/api/chats/{chat_id}/events?after={after}", done)
    assert events[-1]["status"] == "cancelled"


def test_an_invalid_spec_is_refused_with_its_line(serve):
    base, _ = serve(build=lambda config: chat_coordinator())
    chat_id = new_chat(base)
    spec, _ = drafted(base, chat_id)
    code, body = call(base, "POST", f"/api/chats/{chat_id}/go",
                      {"spec": "def add_one(x):\n    return x + 1\n", "spec_id": spec["id"]})
    assert code == 400 and body["line"] == 1 and "docstring" in body["error"]


def test_bad_chat_requests_say_what_is_wrong(serve):
    base, _ = serve(build=lambda config: chat_coordinator())
    assert call(base, "POST", "/api/chats", {"config": "missing"})[0] == 404
    assert call(base, "POST", "/api/chats", {})[0] == 400
    chat_id = new_chat(base)
    assert call(base, "POST", f"/api/chats/{chat_id}/messages", {"text": "   "})[0] == 400
    assert call(base, "POST", f"/api/chats/{chat_id}/messages", {"text": "x" * 4001})[0] == 400
    assert call(base, "POST", f"/api/chats/{chat_id}/go", {"spec": SPEC, "spec_id": "s9"})[0] == 400
    assert call(base, "POST", f"/api/chats/{chat_id}/retry")[0] == 400
    assert call(base, "POST", f"/api/chats/{chat_id}/reinterpret", {"message_id": "m1"})[0] == 400
    assert call(base, "GET", "/api/chats/nope")[0] == 404
    assert call(base, "GET", "/api/chats/..%2Fconfigs")[0] in (400, 404)


def test_readiness_problems_are_returned_and_nothing_is_recorded(serve):
    base, _ = serve(build=lambda config: chat_coordinator(),
                    preflight=lambda config: ["Ollama is not running. Start it with: ollama serve"])
    chat_id = new_chat(base)
    code, body = call(base, "POST", f"/api/chats/{chat_id}/messages", {"text": "add one"})
    assert code == 400 and body["problems"] == ["Ollama is not running. Start it with: ollama serve"]
    assert call(base, "GET", f"/api/chats/{chat_id}")[1]["events"] == []


def test_a_chat_whose_config_was_deleted_says_so(serve, tmp_path):
    base, _ = serve(build=lambda config: chat_coordinator())
    chat_id = new_chat(base)
    (tmp_path / "configs" / "pilot-local.toml").unlink()
    code, body = call(base, "POST", f"/api/chats/{chat_id}/messages", {"text": "add one"})
    assert code == 404 and "pilot-local" in body["error"]


def test_chats_survive_a_restart_and_an_interrupted_one_can_be_retried(serve):
    gate = threading.Event()
    base, _ = serve(build=lambda config: chat_coordinator(
        lambda s, p: (gate.wait(10), chat_respond(s, p))[1]))
    chat_id = new_chat(base)
    assert call(base, "POST", f"/api/chats/{chat_id}/messages", {"text": "add one"})[0] == 202
    base2, _ = serve(build=lambda config: chat_coordinator())  # a second server on the same runs/
    whole = call(base2, "GET", f"/api/chats/{chat_id}")[1]
    gate.set()
    assert whole["events"][-1]["type"] == "done" and whole["events"][-1]["status"] == "interrupted"
    assert call(base2, "POST", f"/api/chats/{chat_id}/retry") == (202, {"retrying": True})
    events = until(base2, f"/api/chats/{chat_id}/events?after={whole['events'][-1]['seq']}", done)
    assert any(e["type"] == "spec" for e in events)
