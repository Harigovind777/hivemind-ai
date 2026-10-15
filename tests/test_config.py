"""Config files: validation, saving as TOML, and safe names."""

import tomllib
from pathlib import Path

import pytest

from hivemind import config as cfg

REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

VALID = {
    "run": {"name": "pilot-ui", "benchmark": "humaneval", "n_tasks": 5, "sample_seed": 0, "seed": 0,
            "candidates": 3, "debate_rounds": 1, "repair": True, "workers": 1},
    "agents": {"provider": "ollama", "model": "qwen2.5-coder:3b", "temperature": 0.7, "max_tokens": 1024},
    "coordinator": {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "high", "max_tokens": 16000},
    "roles": {"testing": {"provider": "ollama", "model": "qwen2.5-coder:7b", "temperature": 0.2}},
}


def test_saved_config_reads_back_identically(tmp_path):
    cfg.save("pilot-ui", VALID, directory=tmp_path)
    assert tomllib.loads((tmp_path / "pilot-ui.toml").read_text()) == VALID
    assert cfg.load("pilot-ui", directory=tmp_path) == VALID


def test_strings_with_quotes_and_unicode_survive_saving(tmp_path):
    config = {**VALID, "run": {**VALID["run"], "name": "quote-run"},
              "agents": {**VALID["agents"], "host": 'http://localhost:11434/"x"\\ü'}}
    cfg.save("q", config, directory=tmp_path)
    assert cfg.load("q", directory=tmp_path)["agents"]["host"] == 'http://localhost:11434/"x"\\ü'


@pytest.mark.parametrize("path, value, field", [
    (("agents", "provider"), "openai", "agents.provider"),
    (("agents", "model"), "", "agents.model"),
    (("coordinator", "effort"), "extreme", "coordinator.effort"),
    (("agents", "temperature"), 3.5, "agents.temperature"),
    (("run", "candidates"), 0, "run.candidates"),
    (("run", "debate_rounds"), -1, "run.debate_rounds"),
    (("run", "repair"), "yes", "run.repair"),
    (("run", "n_tasks"), 165, "run.n_tasks"),
    (("run", "name"), "../escape", "run.name"),
])
def test_invalid_values_are_reported_by_field(path, value, field):
    config = {k: dict(v) for k, v in VALID.items() if k != "roles"}
    config[path[0]][path[1]] = value
    errors = cfg.validate(config)
    assert field in errors


def test_options_belong_to_their_provider():
    config = {k: dict(v) for k, v in VALID.items() if k != "roles"}
    config["agents"]["effort"] = "high"            # an Anthropic option on an Ollama model
    config["coordinator"]["temperature"] = 0.5     # Opus 5.5 takes no sampling parameters
    errors = cfg.validate(config)
    assert "agents.effort" in errors and "coordinator.temperature" in errors


def test_only_agent_roles_can_be_overridden():
    config = {**VALID, "roles": {"coordinator": {"provider": "ollama", "model": "x"},
                                 "poet": {"provider": "ollama", "model": "x"}}}
    errors = cfg.validate(config)
    assert "roles.coordinator" in errors and "roles.poet" in errors


def test_missing_sections_are_reported():
    errors = cfg.validate({"run": VALID["run"]})
    assert "agents" in errors and "coordinator" in errors


def test_invalid_config_is_not_written(tmp_path):
    bad = {**VALID, "agents": {"provider": "openai", "model": "gpt"}}
    with pytest.raises(cfg.ConfigError) as err:
        cfg.save("bad", bad, directory=tmp_path)
    assert "agents.provider" in err.value.errors
    assert not (tmp_path / "bad.toml").exists()


@pytest.mark.parametrize("name", ["../x", "a/b", ".hidden", "", "a\\b", "x..y", "con fig"])
def test_unsafe_names_are_refused(tmp_path, name):
    with pytest.raises(ValueError):
        cfg.save(name, VALID, directory=tmp_path)
    with pytest.raises(ValueError):
        cfg.load(name, directory=tmp_path)


def test_listing_summarises_each_config(tmp_path):
    cfg.save("b-claude", {**VALID, "agents": VALID["coordinator"]}, directory=tmp_path)
    cfg.save("a-local", VALID, directory=tmp_path)
    listing = cfg.list_configs(directory=tmp_path)
    assert [c["name"] for c in listing] == ["a-local", "b-claude"]
    assert listing[0]["agents_model"] == "qwen2.5-coder:3b"
    assert listing[0]["coordinator_model"] == "claude-opus-5-5"
    assert listing[0]["role_overrides"] == ["testing"]
    assert listing[0]["errors"] == {}


@pytest.mark.parametrize("name", ["pilot-local", "claude-main"])
def test_the_repositorys_configs_are_valid(name):
    assert cfg.validate(cfg.load(name, directory=REPO_CONFIGS)) == {}
