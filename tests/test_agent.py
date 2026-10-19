import json

import numpy as np
import pandas as pd
import pytest

from automl_agent.agent import run, select_best
from automl_agent.config import parse_config
from automl_agent.contracts import ExperimentResult


def trial(name, metric, status="ok"):
    return ExperimentResult(name, status, "/tmp/trial", name, "hypothesis", metric=metric)


def test_best_selection_ignores_claimed_success_and_non_finite_scores():
    candidates = [trial("bad", 10000, "error"), trial("nan", float("nan")), trial("a", 0.6), trial("b", 0.8)]
    assert select_best(candidates, True).trial_id == "b"
    assert select_best(candidates, False).trial_id == "a"
    assert select_best([trial("failed", 0, "error")], False) is None


def make_task(path):
    path.mkdir()
    values = np.linspace(0, 10, 60)
    pd.DataFrame({"row": range(60), "feature": values, "outcome": values * 2 + 1}).to_csv(path / "train.csv", index=False)
    pd.DataFrame({"row": range(100, 110), "feature": np.linspace(1, 4, 10)}).to_csv(path / "test.csv", index=False)
    pd.DataFrame({"row": range(100, 110), "outcome": 0.0}).to_csv(path / "sample_submission.csv", index=False)
    (path / "task.md").write_text("Regression. Predict outcome. Metric RMSE. Files: train.csv test.csv sample_submission.csv. Save submission.csv.")
    return path / "task.md"


def test_offline_run_and_resume_preserve_valid_result(tmp_path):
    task = make_task(tmp_path / "dataset")
    run_dir = tmp_path / "run"
    args = [str(task), "--offline", "--run-dir", str(run_dir), "--time-budget", "90", "--trial-timeout", "30", "--n-jobs", "1"]
    report = run(parse_config(args))
    assert report["status"] == "offline"
    assert report["llm_calls"] == 0
    assert report["best_metric"] is not None
    output = pd.read_csv(task.parent / "submission.csv")
    assert list(output.columns) == ["row", "outcome"]
    assert list(output["row"]) == list(range(100, 110))
    assert np.isfinite(output["outcome"]).all()
    resumed = run(parse_config([*args, "--resume"]))
    assert resumed["selected_trial"] == report["selected_trial"]
    assert len(resumed["trials"]) == len(report["trials"])
    frame = pd.read_csv(task.parent / "train.csv")
    frame.loc[0, "outcome"] += 10
    frame.to_csv(task.parent / "train.csv", index=False)
    with pytest.raises(ValueError, match="fingerprint|changed|измен"):
        run(parse_config([*args, "--resume"]))
