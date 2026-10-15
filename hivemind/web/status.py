"""Can a config run on this machine? Ollama, Claude credentials and platform checks.

Every problem is phrased as the fix, so the UI can show it as-is.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_OLLAMA = "http://localhost:11434"


def ollama_status(host: str = DEFAULT_OLLAMA, timeout: float = 1.5) -> dict:
    """Is Ollama answering, and which models has it pulled?"""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read())
        models = sorted(m["name"] for m in data.get("models", []))
        return {"host": host, "reachable": True, "models": models, "error": None}
    except (OSError, ValueError, KeyError) as exc:
        return {"host": host, "reachable": False, "models": [], "error": str(exc)}


def claude_credentials() -> dict:
    """Where Claude credentials would come from. The SDK also reads `ant auth login` profiles."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if os.environ.get(var):
            return {"available": True, "source": var}
    if (Path.home() / ".config" / "anthropic").is_dir():
        return {"available": True, "source": "ant profile"}
    return {"available": False, "source": None}


def _with_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def problems(config: dict, ollama: dict, claude: dict, platform: str = sys.platform) -> list[str]:
    """Everything that would stop `config` from running here, each written as its fix."""
    if platform == "win32":
        return ["The code sandbox does not support Windows yet, so runs cannot start on this machine."]
    sections = [config.get("agents", {}), config.get("coordinator", {}),
                *config.get("roles", {}).values()]
    found: list[str] = []
    ollama_models = [s["model"] for s in sections if s.get("provider") == "ollama" and s.get("model")]
    if ollama_models and not ollama["reachable"]:
        found.append("Ollama is not running. Start it with: ollama serve")
    elif ollama_models:
        pulled = {_with_tag(m) for m in ollama["models"]}
        for model in dict.fromkeys(ollama_models):
            if _with_tag(model) not in pulled:
                found.append(f"Model {model} is not pulled. Run: ollama pull {model}")
    if any(s.get("provider") == "anthropic" for s in sections) and not claude["available"]:
        found.append("Claude needs credentials. Set ANTHROPIC_API_KEY or run: ant auth login")
    return found
