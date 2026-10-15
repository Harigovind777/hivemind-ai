"""The web server, driven over real HTTP on a spare port."""

import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hivemind.agents import Team
from hivemind.coordinator import Coordinator
from hivemind.llm import Ledger, ScriptedLLM
from hivemind.web.server import App, make_server

from test_coordinator import TASK, respond
from test_jobs import FAKE_RUNNER

REPO = Path(__file__).resolve().parents[1]


def scripted(respond_fn=respond, ledger=None):
    llm = ScriptedLLM(respond_fn)
    return Coordinator(Team(llm, llm, ledger or Ledger()), n_candidates=3)


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(**overrides):
        configs = tmp_path / "configs"
        configs.mkdir(exist_ok=True)
        shutil.copy(REPO / "configs" / "pilot-local.toml", configs)
        options = dict(root=tmp_path, config_dir=configs, runs_dir=tmp_path / "runs", tasks=[TASK],
                       build=lambda config: scripted(), preflight=lambda config: [],
                       ollama_host="http://127.0.0.1:9")
        app = App(**{**options, **overrides})
        httpd = make_server(app, "127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}", app

    yield start
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def call(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"null")


def stream(base, path):
    """Read a Server-Sent Events stream to the end and return its JSON events."""
    events = []
    with urllib.request.urlopen(base + path, timeout=20) as resp:
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        for raw in resp:
            line = raw.decode().rstrip("\n")
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_the_page_and_its_files_are_served(serve):
    base, _ = serve()
    with urllib.request.urlopen(base + "/") as resp:
        html = resp.read().decode()
    assert resp.headers["Content-Type"].startswith("text/html")
    assert "/static/app.js" in html and "/static/app.css" in html and "/static/chat.js" in html
    assert html.index('data-tab="chat"') < html.index('data-tab="run"')
    with urllib.request.urlopen(base + "/static/app.js") as resp:
        assert resp.headers["Content-Type"].startswith("text/javascript")
    with urllib.request.urlopen(base + "/static/chat.js") as resp:
        assert resp.headers["Content-Type"].startswith("text/javascript")
    for bad in ("/static/../server.py", "/static/%2e%2e/server.py", "/static/missing.js"):
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + bad)
        assert err.value.code == 404


def test_unknown_api_routes_answer_json_404(serve):
    base, _ = serve()
    assert call(base, "GET", "/api/nope") == (404, {"error": "Not found: GET /api/nope"})


def test_status_reports_ollama_and_no_running_job(serve):
    base, _ = serve()
    code, body = call(base, "GET", "/api/status")
    assert code == 200
    assert body["ollama"]["reachable"] is False and body["current"] is None
    assert {"platform", "python", "claude"} <= set(body)


def test_tasks_are_listed_with_their_function_and_summary(serve):
    base, _ = serve()
    assert call(base, "GET", "/api/tasks") == (200, [
        {"task_id": "Toy/0", "entry_point": "add_one", "summary": "Return x plus one."}])


def test_configs_can_be_read_validated_and_saved(serve):
    base, _ = serve()
    code, listing = call(base, "GET", "/api/configs")
    assert code == 200 and [c["name"] for c in listing] == ["pilot-local"]
    code, body = call(base, "GET", "/api/configs/pilot-local")
    assert code == 200 and body["config"]["agents"]["model"] == "qwen2.5-coder:3b"

    config = body["config"]
    config["roles"] = {"testing": {"provider": "ollama", "model": "llama3"}}
    assert call(base, "PUT", "/api/configs/with-tester", {"config": config})[0] == 200
    assert call(base, "GET", "/api/configs/with-tester")[1]["config"]["roles"]["testing"]["model"] == "llama3"

    config["agents"]["provider"] = "openai"
    code, body = call(base, "PUT", "/api/configs/broken", {"config": config})
    assert code == 400 and "agents.provider" in body["errors"]
    assert call(base, "GET", "/api/configs/broken")[0] == 404
    assert call(base, "PUT", "/api/configs/..%2Fescape", {"config": config})[0] == 400


def test_a_live_run_streams_every_event_until_done(serve, tmp_path):
    base, _ = serve()
    code, body = call(base, "POST", "/api/live", {"config": "pilot-local", "task_id": "Toy/0"})
    assert code == 201
    events = stream(base, f"/api/live/{body['id']}/events")
    kinds = [e["type"] for e in events]
    assert kinds[0] == "started" and kinds[-2:] == ["outcome", "done"]
    assert {"turn_started", "turn", "evidence", "rulings"} <= set(kinds)
    assert events[0]["config"] == "pilot-local"
    assert (tmp_path / "runs" / "demos" / "Toy_0.json").exists()

    resumed = stream(base, f"/api/live/{body['id']}/events?after=3")
    assert resumed[0]["seq"] == 4 and resumed == events[3:]


def test_only_one_job_runs_at_a_time(serve):
    gate = threading.Event()

    def slow(system, prompt):
        gate.wait(10)
        return respond(system, prompt)

    base, _ = serve(build=lambda config: scripted(slow))
    first = call(base, "POST", "/api/live", {"config": "pilot-local", "task_id": "Toy/0"})[1]["id"]
    code, body = call(base, "POST", "/api/live", {"config": "pilot-local", "task_id": "Toy/0"})
    assert code == 409 and body["current"]["id"] == first
    assert call(base, "GET", "/api/status")[1]["current"]["id"] == first

    assert call(base, "POST", f"/api/live/{first}/cancel")[0] == 200
    gate.set()
    events = stream(base, f"/api/live/{first}/events")
    assert events[-1]["type"] == "done" and events[-1]["status"] == "cancelled"


def test_readiness_problems_block_a_start(serve):
    base, _ = serve(preflight=lambda config: ["Ollama is not running. Start it with: ollama serve"])
    code, body = call(base, "POST", "/api/live", {"config": "pilot-local", "task_id": "Toy/0"})
    assert code == 400 and body["problems"] == ["Ollama is not running. Start it with: ollama serve"]


def test_bad_start_requests_say_what_is_wrong(serve):
    base, _ = serve()
    assert call(base, "POST", "/api/live", {"config": "missing", "task_id": "Toy/0"})[0] == 404
    assert call(base, "POST", "/api/live", {"config": "pilot-local", "task_id": "Nope/1"})[0] == 404
    assert call(base, "POST", "/api/live", {"task_id": "Toy/0"})[0] == 400


def test_an_experiment_runs_as_a_child_process_with_progress(serve, tmp_path):
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER)

    def command(config_path, n_tasks, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        return [sys.executable, str(script), str(out_dir), str(n_tasks), "0", "0"]

    base, _ = serve(experiment_command=command)
    code, body = call(base, "POST", "/api/experiments", {"config": "pilot-local", "n_tasks": 2})
    assert code == 201
    deadline = time.time() + 20
    while (progress := call(base, "GET", f"/api/experiments/{body['id']}")[1])["status"] == "running":
        assert time.time() < deadline
        time.sleep(0.05)
    assert progress["status"] == "done" and (progress["done"], progress["total"]) == (2, 2)
    assert progress["out_dir"].endswith("pilot-local-qwen2.5-coder-3b")


def test_finished_runs_can_be_browsed(serve, tmp_path):
    run = tmp_path / "runs" / "toy-run"
    (run / "transcripts").mkdir(parents=True)
    ledger = Ledger()
    outcome, transcript = scripted(ledger=ledger).solve(TASK)
    (run / "outcomes.jsonl").write_text(json.dumps(outcome.to_dict()) + "\n")
    (run / "calls.jsonl").write_text("".join(json.dumps(r.__dict__) + "\n" for r in ledger.records))
    (run / "transcripts" / "Toy_0.json").write_text(json.dumps({"task_id": "Toy/0", "turns": transcript.turns}))
    (run / "report.md").write_text("# Toy report\n")
    base, _ = serve()

    code, listing = call(base, "GET", "/api/runs")
    assert code == 200
    row = next(r for r in listing["runs"] if r["name"] == "toy-run")
    assert row["tasks"] == 1 and row["pass"]["arbitration"] == 1 and row["pass"]["vote"] == 0

    code, detail = call(base, "GET", "/api/runs/toy-run")
    assert code == 200 and detail["report_md"] == "# Toy report\n" and detail["transcripts"] == ["Toy_0"]
    assert detail["summary"]["strategies"]["arbitration"]["passed"] == 1
    code, tr = call(base, "GET", "/api/runs/toy-run/transcripts/Toy_0")
    assert code == 200 and tr["task_id"] == "Toy/0"
    assert call(base, "GET", "/api/runs/..%2Fconfigs")[0] in (400, 404)


def test_a_configs_readiness_lists_what_to_fix(serve):
    base, _ = serve(preflight=lambda config: ["Model x is not pulled. Run: ollama pull x"])
    assert call(base, "GET", "/api/configs/pilot-local/readiness") == (
        200, {"problems": ["Model x is not pulled. Run: ollama pull x"], "errors": {}})


def test_serve_command_passes_port_and_browser_choice(monkeypatch):
    from hivemind import __main__ as cli
    from hivemind.web import server

    seen = {}
    monkeypatch.setattr(server, "serve", lambda port, open_browser: seen.update(port=port, open=open_browser))
    monkeypatch.setattr(sys, "argv", ["hivemind", "serve", "--port", "9001", "--no-open"])
    cli.main()
    assert seen == {"port": 9001, "open": False}
