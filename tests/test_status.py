"""Readiness checks: Ollama, Claude credentials, and whether a config can run here."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hivemind.web import status


@pytest.fixture
def fake_ollama():
    class Tags(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"models": [{"name": "qwen2.5-coder:3b"}, {"name": "llama3:latest"}]}).encode()
            self.send_response(200 if self.path == "/api/tags" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Tags)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_reachable_ollama_lists_its_models(fake_ollama):
    s = status.ollama_status(fake_ollama)
    assert s == {"host": fake_ollama, "reachable": True,
                 "models": ["llama3:latest", "qwen2.5-coder:3b"], "error": None}


def test_unreachable_ollama_is_reported_not_raised():
    s = status.ollama_status("http://127.0.0.1:9", timeout=0.5)
    assert s["reachable"] is False and s["models"] == [] and s["error"]


def test_claude_credentials_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(status.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert status.claude_credentials() == {"available": False, "source": None}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert status.claude_credentials() == {"available": True, "source": "ANTHROPIC_API_KEY"}


def config_with(agents, coordinator=None, roles=None):
    return {"run": {"name": "x"}, "agents": agents, "coordinator": coordinator or agents,
            "roles": roles or {}}


OLLAMA_UP = {"host": "http://localhost:11434", "reachable": True,
             "models": ["qwen2.5-coder:3b", "llama3:latest"], "error": None}
NO_KEY = {"available": False, "source": None}


def test_ready_config_has_no_problems():
    config = config_with({"provider": "ollama", "model": "qwen2.5-coder:3b"},
                         roles={"testing": {"provider": "ollama", "model": "llama3"}})
    assert status.problems(config, OLLAMA_UP, NO_KEY, platform="darwin") == []


def test_unpulled_model_names_the_pull_command():
    config = config_with({"provider": "ollama", "model": "qwen2.5-coder:7b"})
    assert status.problems(config, OLLAMA_UP, NO_KEY, platform="darwin") == [
        "Model qwen2.5-coder:7b is not pulled. Run: ollama pull qwen2.5-coder:7b"]


def test_stopped_ollama_names_the_start_command():
    down = {**OLLAMA_UP, "reachable": False, "models": []}
    config = config_with({"provider": "ollama", "model": "qwen2.5-coder:3b"})
    assert status.problems(config, down, NO_KEY, platform="linux") == [
        "Ollama is not running. Start it with: ollama serve"]


def test_claude_without_credentials_says_how_to_add_them():
    config = config_with({"provider": "ollama", "model": "qwen2.5-coder:3b"},
                         coordinator={"provider": "anthropic", "model": "claude-opus-5-5"})
    assert status.problems(config, OLLAMA_UP, NO_KEY, platform="darwin") == [
        "Claude needs credentials. Set ANTHROPIC_API_KEY or run: ant auth login"]


def test_windows_is_flagged_until_the_sandbox_supports_it():
    config = config_with({"provider": "ollama", "model": "qwen2.5-coder:3b"})
    assert status.problems(config, OLLAMA_UP, NO_KEY, platform="win32") == [
        "The code sandbox does not support Windows yet, so runs cannot start on this machine."]
