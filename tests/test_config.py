from pathlib import Path

import pytest

from automl_agent.config import parse_config


def test_baseline_cli_and_secret_exclusion(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:7777/v1")
    config = parse_config([str(tmp_path / "task.md"), "--time-budget", "120"])
    assert config.task_path == (tmp_path / "task.md").resolve()
    assert config.provider == "openai"
    assert config.model == "test-model"
    assert config.api_key == "private-test-key"
    assert "private-test-key" not in str(config.public_dict())
    assert 0 < config.reserve_seconds < config.time_budget


def test_offline_needs_no_model_or_api(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = parse_config(["--offline", "--time-budget", "10"])
    assert config.offline is True
    assert config.task_path == Path("task.md").resolve()


@pytest.mark.parametrize("args", [
    ["--time-budget", "0"], ["--time-budget", "10801"],
    ["--n-jobs", "9"], ["--max-trials", "0"],
    ["--validation-fraction", "0.99"], ["--resume"],
])
def test_invalid_configuration_is_rejected(args):
    with pytest.raises((ValueError, SystemExit)):
        parse_config(["--offline", *args])
