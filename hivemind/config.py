"""Experiment configs: list, load, validate and save the TOML files in configs/.

Python can read TOML (tomllib) but not write it, so `to_toml` writes the small,
flat subset these configs use. A config is only written once it validates, and
names are restricted so a request cannot reach files outside the directory.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from pathlib import Path

from .agents import ROLE_PROMPTS

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
HUMANEVAL_SIZE = 164
AGENT_ROLES = tuple(r for r in ROLE_PROMPTS if r != "coordinator")
EFFORTS = ("low", "medium", "high", "xhigh", "max")  # output_config.effort on current Claude models
PROVIDERS = ("ollama", "anthropic")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class ConfigError(ValueError):
    """The config did not validate; `errors` maps each bad field to a message."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def is_safe_name(name: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(name or "")) and ".." not in name


def _checked(name: str) -> str:
    if not is_safe_name(name):
        raise ValueError(f"invalid name {name!r}: use letters, digits, '-', '_' and '.'")
    return name


# ------------------------------------------------------------------ validation

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int_between(lo: int, hi: int):
    return lambda v: None if _is_int(v) and lo <= v <= hi else f"Whole number from {lo} to {hi}"


RUN_FIELDS = {
    "name": lambda v: None if isinstance(v, str) and is_safe_name(v)
    else "Letters, digits, '-', '_' and '.' only",
    "benchmark": lambda v: None if v == "humaneval" else "Only 'humaneval' is supported",
    "n_tasks": _int_between(1, HUMANEVAL_SIZE),
    "sample_seed": _int_between(0, 2**31 - 1),
    "seed": _int_between(0, 2**31 - 1),
    "candidates": _int_between(1, 8),
    "debate_rounds": _int_between(0, 5),
    "repair": lambda v: None if isinstance(v, bool) else "true or false",
    "workers": _int_between(1, 16),
    "task_ids": lambda v: None if isinstance(v, list) and all(isinstance(t, str) for t in v)
    else "A list of task ids",
}

MODEL_OPTIONS = {
    "ollama": {
        "temperature": lambda v: None if _is_number(v) and 0 <= v <= 2 else "Number from 0 to 2",
        "max_tokens": _int_between(1, 131072),
        "host": lambda v: None if isinstance(v, str) and v.startswith(("http://", "https://"))
        else "A URL such as http://localhost:11434",
        "timeout_s": lambda v: None if _is_number(v) and v > 0 else "Seconds, above 0",
    },
    "anthropic": {
        "effort": lambda v: None if v in EFFORTS else f"One of {', '.join(EFFORTS)}",
        "max_tokens": _int_between(1, 128000),
    },
}


def _validate_model(section, path: str, errors: dict[str, str]) -> None:
    if not isinstance(section, dict):
        errors[path] = "Must be a table of settings"
        return
    provider = section.get("provider")
    if provider not in PROVIDERS:
        errors[f"{path}.provider"] = f"One of {', '.join(PROVIDERS)}"
    model = section.get("model")
    if not isinstance(model, str) or not model.strip():
        errors[f"{path}.model"] = "Required"
    if provider not in PROVIDERS:
        return
    options = MODEL_OPTIONS[provider]
    for key, value in section.items():
        if key in ("provider", "model"):
            continue
        check = options.get(key)
        if check is None:
            errors[f"{path}.{key}"] = f"Not a setting for {provider} models"
        elif (message := check(value)) is not None:
            errors[f"{path}.{key}"] = message


def validate(config: dict) -> dict[str, str]:
    """Return {field: message} for every problem; an empty dict means the config is valid."""
    errors: dict[str, str] = {}
    for key in config:
        if key not in ("run", "agents", "coordinator", "roles"):
            errors[key] = "Unknown section"
    run = config.get("run")
    if not isinstance(run, dict):
        errors["run"] = "Missing section"
    else:
        if "name" not in run:
            errors["run.name"] = "Required"
        for key, value in run.items():
            check = RUN_FIELDS.get(key)
            if check is None:
                errors[f"run.{key}"] = "Unknown setting"
            elif (message := check(value)) is not None:
                errors[f"run.{key}"] = message
    for section in ("agents", "coordinator"):
        if section not in config:
            errors[section] = "Missing section"
        else:
            _validate_model(config[section], section, errors)
    roles = config.get("roles", {})
    if not isinstance(roles, dict):
        errors["roles"] = "Must be a table of roles"
    else:
        for role, section in roles.items():
            if role not in AGENT_ROLES:
                errors[f"roles.{role}"] = f"Not an agent role; use one of {', '.join(AGENT_ROLES)}"
            else:
                _validate_model(section, f"roles.{role}", errors)
    return errors


# ------------------------------------------------------------------ TOML

def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # JSON string escapes are valid TOML basic-string escapes; DEL must be escaped too.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def to_toml(config: dict) -> str:
    lines = ["# Saved by the HiveMind web interface.", ""]
    tables = [(s, config[s]) for s in ("run", "agents", "coordinator") if s in config]
    tables += [(f"roles.{role}", section) for role, section in config.get("roles", {}).items()]
    for name, values in tables:
        lines.append(f"[{name}]")
        lines += [f"{key} = {_toml_value(value)}" for key, value in values.items()]
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ files

def load(name: str, directory: Path = CONFIG_DIR) -> dict:
    with open(Path(directory) / f"{_checked(name)}.toml", "rb") as f:
        return tomllib.load(f)


def save(name: str, config: dict, directory: Path = CONFIG_DIR) -> Path:
    _checked(name)
    errors = validate(config)
    if errors:
        raise ConfigError(errors)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.toml"
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".toml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(to_toml(config))
    os.replace(tmp, path)
    return path


def list_configs(directory: Path = CONFIG_DIR) -> list[dict]:
    rows = []
    for path in sorted(Path(directory).glob("*.toml")):
        if not is_safe_name(path.stem):
            continue
        try:
            config = load(path.stem, directory)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            rows.append({"name": path.stem, "errors": {"file": str(exc)}})
            continue
        agents, coordinator = config.get("agents", {}), config.get("coordinator", {})
        rows.append({
            "name": path.stem,
            "run_name": config.get("run", {}).get("name"),
            "agents_provider": agents.get("provider"), "agents_model": agents.get("model"),
            "coordinator_provider": coordinator.get("provider"),
            "coordinator_model": coordinator.get("model"),
            "role_overrides": sorted(config.get("roles", {})),
            "errors": validate(config),
        })
    return rows
