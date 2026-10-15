"""The HiveMind web interface: a small JSON + Server-Sent Events API and a static page.

Standard library only. It listens on 127.0.0.1, runs one job at a time (a live run
in a thread, or an experiment as a child process), and never serves files outside
configs/, runs/ and its own static folder.
"""

from __future__ import annotations

import ast
import json
import platform
import re
import sys
import threading
import time
import tomllib
import traceback
import uuid
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlsplit

from .. import config as cfg
from ..chat import SpecError
from ..domains.code import CodeTask, load_humaneval
from ..llm import Ledger
from ..report import summarise
from ..runner import ROOT, build_coordinator
from .chats import Chat, ChatBusy, ChatError, load_chats
from .jobs import ExperimentJob
from .live import LiveRun
from .status import DEFAULT_OLLAMA, claude_credentials, ollama_status, problems

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
MAX_BODY = 1 << 20


class ApiError(Exception):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = status
        self.payload = {"error": message, **extra}


@contextmanager
def _chat_errors():
    """Chat refusals as HTTP answers: busy is 409; anything else the chat rejects is 400."""
    try:
        yield
    except ChatBusy as exc:
        raise ApiError(409, str(exc)) from None
    except SpecError as exc:
        raise ApiError(400, str(exc), line=exc.line) from None
    except ChatError as exc:
        raise ApiError(400, str(exc)) from None


def _safe(name: str, what: str) -> str:
    if not cfg.is_safe_name(name):
        raise ApiError(400, f"Invalid {what} name {name!r}")
    return name


def _summary_line(task: CodeTask) -> str:
    try:
        tree = ast.parse(task.prompt)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == task.entry_point:
            doc = ast.get_docstring(node) or ""
            return next((line.strip() for line in doc.splitlines() if line.strip()), "")
    return ""


def _default_command(config_path: Path, n_tasks: int, out_dir: Path) -> list[str]:
    return [sys.executable, "-m", "hivemind", "run", str(config_path),
            "--tasks", str(n_tasks), "--out", str(out_dir)]


class App:
    """Everything the API does, independent of HTTP so it can be tested directly."""

    def __init__(self, root: Path = ROOT, config_dir: Path | None = None,
                 runs_dir: Path | None = None, tasks: list[CodeTask] | None = None,
                 build: Callable[[dict], object] | None = None,
                 preflight: Callable[[dict], list[str]] | None = None,
                 experiment_command: Callable[[Path, int, Path], list[str]] = _default_command,
                 ollama_host: str = DEFAULT_OLLAMA):
        self.root = Path(root)
        self.config_dir = Path(config_dir or self.root / "configs")
        self.runs_dir = Path(runs_dir or self.root / "runs")
        self._tasks = tasks
        self.build = build or (lambda config: build_coordinator(config, Ledger()))
        self.preflight = preflight or (lambda config: problems(
            config, ollama_status(self.ollama_host), claude_credentials()))
        self.experiment_command = experiment_command
        self.ollama_host = ollama_host
        self.lives: dict[str, LiveRun] = {}
        self.jobs: dict[str, ExperimentJob] = {}
        self._lock = threading.Lock()
        self.chats_dir = self.runs_dir / "chats"
        self._chats: dict[str, Chat] | None = None
        self._chats_lock = threading.Lock()

    # ------------------------------------------------------------------ status

    def tasks(self) -> list[CodeTask]:
        if self._tasks is None:
            self._tasks = load_humaneval()
        return self._tasks

    def current(self) -> dict | None:
        for job in [*self.lives.values(), *self.jobs.values()]:
            if not job.finished:
                return job.summary()
        for chat in self._all_chats().values():
            if chat.busy == "round":
                return chat.summary()
        return None

    def status(self) -> dict:
        return {"platform": sys.platform, "python": platform.python_version(),
                "ollama": ollama_status(self.ollama_host), "claude": claude_credentials(),
                "current": self.current()}

    def task_list(self) -> list[dict]:
        return [{"task_id": t.task_id, "entry_point": t.entry_point, "summary": _summary_line(t)}
                for t in self.tasks()]

    # ------------------------------------------------------------------ configs

    def get_config(self, name: str) -> dict:
        _safe(name, "config")
        try:
            config = cfg.load(name, self.config_dir)
        except FileNotFoundError:
            raise ApiError(404, f"No config named {name!r}")
        except tomllib.TOMLDecodeError as exc:
            raise ApiError(400, f"{name}.toml is not valid TOML: {exc}")
        return {"name": name, "config": config, "errors": cfg.validate(config)}

    def put_config(self, name: str, body: dict) -> dict:
        _safe(name, "config")
        config = body.get("config")
        if not isinstance(config, dict):
            raise ApiError(400, "Send the config as {\"config\": {...}}")
        try:
            cfg.save(name, config, self.config_dir)
        except cfg.ConfigError as exc:
            raise ApiError(400, "The config has errors; nothing was saved", errors=exc.errors)
        return {"name": name, "saved": True}

    def readiness(self, name: str) -> dict:
        config = self.get_config(name)
        found = [] if config["errors"] else self.preflight(config["config"])
        return {"problems": found, "errors": config["errors"]}

    def _runnable_config(self, body: dict) -> tuple[str, dict]:
        name = body.get("config")
        if not isinstance(name, str):
            raise ApiError(400, "Choose a config")
        config = self.get_config(name)
        if config["errors"]:
            raise ApiError(400, f"Config {name!r} has errors; fix it on the Agents tab",
                           errors=config["errors"])
        found = self.preflight(config["config"])
        if found:
            raise ApiError(400, "This config cannot run on this machine yet", problems=found)
        return name, config["config"]

    def _ensure_idle(self) -> None:
        current = self.current()
        if current is not None:
            what = (current.get("task_id") or current.get("title") or current.get("config")
                    or current["id"])
            raise ApiError(409, f"A {current['kind']} run ({what}) is still running; cancel it first",
                           current=current)

    @staticmethod
    def _models(config: dict) -> dict:
        return {"agents": config["agents"]["model"], "coordinator": config["coordinator"]["model"],
                "roles": {r: s["model"] for r, s in config.get("roles", {}).items()}}

    # ------------------------------------------------------------------ live runs

    def start_live(self, body: dict) -> dict:
        with self._lock:
            self._ensure_idle()
            name, config = self._runnable_config(body)
            task_id = body.get("task_id")
            task = next((t for t in self.tasks() if t.task_id == task_id), None)
            if task is None:
                raise ApiError(404, f"No problem with id {task_id!r}")
            run = LiveRun(uuid.uuid4().hex[:12], task, lambda: self.build(config),
                          out_dir=self.runs_dir / "demos", review=bool(body.get("review")),
                          meta={"config": name, "models": self._models(config),
                                "candidates": config["run"].get("candidates", 3)})
            self.lives[run.id] = run
            run.start()
        return {"id": run.id}

    def live(self, run_id: str) -> LiveRun:
        run = self.lives.get(run_id)
        if run is None:
            raise ApiError(404, f"No live run {run_id!r}")
        return run

    # ------------------------------------------------------------------ experiments

    def start_experiment(self, body: dict) -> dict:
        with self._lock:
            self._ensure_idle()
            name, config = self._runnable_config(body)
            n_tasks = body.get("n_tasks", config["run"].get("n_tasks", cfg.HUMANEVAL_SIZE))
            if not (isinstance(n_tasks, int) and 1 <= n_tasks <= cfg.HUMANEVAL_SIZE):
                raise ApiError(400, f"Number of problems must be 1 to {cfg.HUMANEVAL_SIZE}")
            out_dir = self.runs_dir / config["run"]["name"]
            command = self.experiment_command(self.config_dir / f"{name}.toml", n_tasks, out_dir)
            job = ExperimentJob(uuid.uuid4().hex[:12], command, out_dir=out_dir, total=n_tasks,
                                cwd=self.root, meta={"config": name, "models": self._models(config)})
            self.jobs[job.id] = job
            job.start()
        return {"id": job.id}

    def job(self, job_id: str) -> ExperimentJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise ApiError(404, f"No experiment {job_id!r}")
        return job

    # ------------------------------------------------------------------ chats

    def _all_chats(self) -> dict[str, Chat]:
        with self._chats_lock:
            if self._chats is None:
                self._chats = {c.id: c for c in load_chats(self.chats_dir)}
            return self._chats

    def chat(self, chat_id: str) -> Chat:
        chat = self._all_chats().get(_safe(chat_id, "chat"))
        if chat is None:
            raise ApiError(404, f"No chat {chat_id!r}")
        return chat

    def chat_list(self) -> list[dict]:
        return sorted((c.summary() for c in self._all_chats().values()),
                      key=lambda s: s["updated"], reverse=True)

    def create_chat(self, body: dict) -> dict:
        name = body.get("config")
        if not isinstance(name, str):
            raise ApiError(400, "Choose a config")
        config = self.get_config(name)
        if config["errors"]:
            raise ApiError(400, f"Config {name!r} has errors; fix it on the Agents tab",
                           errors=config["errors"])
        chat = Chat.create(self.chats_dir, name)
        self._all_chats()[chat.id] = chat
        return {"id": chat.id}

    def _chat_build(self, chat: Chat):
        """Readiness first, so the chat can show what to fix; then a team for its config."""
        _, config = self._runnable_config({"config": chat.config})
        return lambda: self.build(config)

    def chat_send(self, chat_id: str, body: dict) -> dict:
        chat = self.chat(chat_id)
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, "Type a message first")
        if len(text) > 4000:
            raise ApiError(400, "Keep messages under 4,000 characters")
        build = self._chat_build(chat)
        with _chat_errors():
            return {"message_id": chat.send(text.strip(), build)}

    def chat_go(self, chat_id: str, body: dict) -> dict:
        chat = self.chat(chat_id)
        spec, spec_id = body.get("spec"), body.get("spec_id")
        if not isinstance(spec, str) or not isinstance(spec_id, str):
            raise ApiError(400, 'Send the spec as {"spec": "...", "spec_id": "s1"}')
        build = self._chat_build(chat)
        with self._lock, _chat_errors():
            self._ensure_idle()
            return {"round": chat.go(spec, spec_id, build)}

    def chat_reinterpret(self, chat_id: str, body: dict) -> dict:
        chat = self.chat(chat_id)
        message_id = body.get("message_id")
        if not isinstance(message_id, str):
            raise ApiError(400, "Say which message to switch")
        build = self._chat_build(chat)
        with _chat_errors():
            chat.reinterpret(message_id, build)
        return {"reinterpreting": message_id}

    def chat_retry(self, chat_id: str) -> dict:
        chat = self.chat(chat_id)
        action = chat.retryable()
        if action is None:
            raise ApiError(400, "There is nothing to retry")
        build = self._chat_build(chat)
        with self._lock, _chat_errors():
            if action["action"] == "round":
                self._ensure_idle()
            chat.retry(build)
        return {"retrying": True}

    # ------------------------------------------------------------------ results

    def runs(self) -> dict:
        rows = []
        if self.runs_dir.is_dir():
            for d in sorted(self.runs_dir.iterdir()):
                outcomes = d / "outcomes.jsonl"
                if d.name == "demos" or not outcomes.exists() or not cfg.is_safe_name(d.name):
                    continue
                row = {"name": d.name, "updated": outcomes.stat().st_mtime}
                try:
                    s = summarise(d)
                    row.update(tasks=s["n_tasks"], oracle=s["oracle_passed"],
                               models=sorted({m["model"] for m in s["models"]}))
                    row["pass"] = {k: v["passed"] for k, v in s["strategies"].items()}
                except (ValueError, KeyError) as exc:
                    row["error"] = f"Could not read results yet: {exc}"
                rows.append(row)
        demos = []
        demo_dir = self.runs_dir / "demos"
        if demo_dir.is_dir():
            for f in sorted(demo_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                demos.append({"name": f.stem, "updated": f.stat().st_mtime})
        return {"runs": sorted(rows, key=lambda r: r["updated"], reverse=True), "demos": demos}

    def run_detail(self, name: str) -> dict:
        d = self.runs_dir / _safe(name, "run")
        if not (d / "outcomes.jsonl").exists() or name == "demos":
            raise ApiError(404, f"No run named {name!r}")
        config = None
        if (d / "config.toml").exists():
            try:
                config = tomllib.loads((d / "config.toml").read_text())
            except tomllib.TOMLDecodeError:
                pass
        report = d / "report.md"
        stems = sorted((p.stem for p in (d / "transcripts").glob("*.json")),
                       key=lambda s: (s.rsplit("_", 1)[0], int(s.rsplit("_", 1)[-1]) if s.rsplit("_", 1)[-1].isdigit() else 0))
        return {"name": name, "summary": summarise(d), "config": config,
                "report_md": report.read_text() if report.exists() else None, "transcripts": stems}

    def transcript(self, name: str, task: str) -> dict:
        path = self.runs_dir / _safe(name, "run") / "transcripts" / f"{_safe(task, 'transcript')}.json"
        if not path.exists():
            raise ApiError(404, f"No transcript {task!r} in run {name!r}")
        return json.loads(path.read_text())

    def demo(self, name: str) -> dict:
        path = self.runs_dir / "demos" / f"{_safe(name, 'demo')}.json"
        if not path.exists():
            raise ApiError(404, f"No demo named {name!r}")
        return json.loads(path.read_text())


# ---------------------------------------------------------------------- HTTP


def _routes(app: App):
    """(method, pattern, handler(match groups..., body, query)) in match order."""
    return [
        ("GET", r"/api/status", lambda b, q: app.status()),
        ("GET", r"/api/tasks", lambda b, q: app.task_list()),
        ("GET", r"/api/configs", lambda b, q: cfg.list_configs(app.config_dir)),
        ("GET", r"/api/configs/([^/]+)", lambda name, b, q: app.get_config(name)),
        ("GET", r"/api/configs/([^/]+)/readiness", lambda name, b, q: app.readiness(name)),
        ("PUT", r"/api/configs/([^/]+)", lambda name, b, q: app.put_config(name, b)),
        ("POST", r"/api/live", lambda b, q: (201, app.start_live(b))),
        ("GET", r"/api/live/([^/]+)", lambda i, b, q: app.live(i).summary()),
        ("POST", r"/api/live/([^/]+)/cancel", lambda i, b, q: (app.live(i).cancel(), {"cancelling": True})[1]),
        ("POST", r"/api/experiments", lambda b, q: (201, app.start_experiment(b))),
        ("GET", r"/api/experiments/([^/]+)", lambda i, b, q: app.job(i).progress()),
        ("POST", r"/api/experiments/([^/]+)/cancel", lambda i, b, q: (app.job(i).cancel(), {"cancelled": True})[1]),
        ("GET", r"/api/chats", lambda b, q: app.chat_list()),
        ("POST", r"/api/chats", lambda b, q: (201, app.create_chat(b))),
        ("GET", r"/api/chats/([^/]+)", lambda i, b, q: app.chat(i).to_dict()),
        ("POST", r"/api/chats/([^/]+)/messages", lambda i, b, q: (202, app.chat_send(i, b))),
        ("POST", r"/api/chats/([^/]+)/go", lambda i, b, q: (202, app.chat_go(i, b))),
        ("POST", r"/api/chats/([^/]+)/reinterpret", lambda i, b, q: (202, app.chat_reinterpret(i, b))),
        ("POST", r"/api/chats/([^/]+)/retry", lambda i, b, q: (202, app.chat_retry(i))),
        ("POST", r"/api/chats/([^/]+)/cancel", lambda i, b, q: (app.chat(i).cancel(), {"cancelling": True})[1]),
        ("GET", r"/api/runs", lambda b, q: app.runs()),
        ("GET", r"/api/runs/([^/]+)", lambda name, b, q: app.run_detail(name)),
        ("GET", r"/api/runs/([^/]+)/transcripts/([^/]+)", lambda name, t, b, q: app.transcript(name, t)),
        ("GET", r"/api/demos/([^/]+)", lambda name, b, q: app.demo(name)),
    ]


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    routes = [(m, re.compile(p + "$"), h) for m, p, h in _routes(app)]
    events_routes = [(re.compile(r"/api/live/([^/]+)/events$"), app.live),
                     (re.compile(r"/api/chats/([^/]+)/events$"), app.chat)]

    class Handler(BaseHTTPRequestHandler):
        server_version = "HiveMind"

        def log_message(self, format, *args):  # keep the terminal for the run's own output
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ApiError(413, "Request too large")
            if not length:
                return {}
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                raise ApiError(400, "The request body is not valid JSON")
            if not isinstance(body, dict):
                raise ApiError(400, "The request body must be a JSON object")
            return body

        def _static(self, path: str) -> None:
            name = "index.html" if path == "/" else unquote(path[len("/static/"):])
            file = STATIC_DIR / name
            if path not in ("/",) and not path.startswith("/static/") \
                    or not cfg.is_safe_name(name) or not file.is_file():
                self._json(404, {"error": f"Not found: {path}"})
                return
            self._send(200, file.read_bytes(), CONTENT_TYPES.get(file.suffix, "application/octet-stream"))

        def _events(self, run, query: dict) -> None:
            """Stream a live run's or a chat's events, from `?after=` or Last-Event-ID on."""
            try:
                after = int((query.get("after") or [self.headers.get("Last-Event-ID") or 0])[0])
            except ValueError:
                after = 0
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    events = run.wait_events(after, timeout=15)
                    if not events:
                        if run.finished:
                            break
                        self.wfile.write(b": still running\n\n")
                    for event in events:
                        self.wfile.write(f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n".encode())
                        after = event["seq"]
                    self.wfile.flush()
                    if run.finished and not run.events_after(after):
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page closed or reloaded; it reconnects with ?after=

        def _dispatch(self, method: str) -> None:
            parts = urlsplit(self.path)
            path, query = parts.path, parse_qs(parts.query)
            try:
                if method == "GET" and (path == "/" or path.startswith("/static/")):
                    return self._static(path)
                if method == "GET":
                    for pattern, find in events_routes:
                        if m := pattern.fullmatch(path):
                            return self._events(find(unquote(m.group(1))), query)
                for route_method, pattern, handler in routes:
                    if route_method == method and (m := pattern.fullmatch(path)):
                        body = self._body() if method in ("POST", "PUT") else {}
                        result = handler(*[unquote(g) for g in m.groups()], body, query)
                        status, payload = result if isinstance(result, tuple) else (200, result)
                        return self._json(status, payload)
                self._json(404, {"error": f"Not found: {method} {path}"})
            except ApiError as err:
                self._json(err.status, err.payload)
            except Exception as exc:  # report, keep serving
                traceback.print_exc()
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_PUT(self):
            self._dispatch("PUT")

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def serve(port: int = 8765, open_browser: bool = True) -> None:
    try:
        httpd = make_server(App(), "127.0.0.1", port)
    except OSError as exc:
        sys.exit(f"[hivemind] cannot listen on port {port} ({exc.strerror}); try --port {port + 1}")
    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(f"[hivemind] web interface running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[hivemind] stopping", flush=True)
    finally:
        httpd.server_close()
