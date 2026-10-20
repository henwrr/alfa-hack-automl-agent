import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from automl_agent.contracts import ExperimentResult, TaskSpec
from automl_agent.prompts import build_messages


def _task(feature_count=200, description_size=5000):
    features = [f"feature_{index:03d}" for index in range(feature_count)]
    return TaskSpec(
        task_path="task.md", train_path="train.csv", test_path="test.csv", target="label",
        task_type="classification", metric="macro_f1", maximize=True,
        feature_columns=features, id_columns=[], categorical_columns=features[:20],
        output_columns=["label"], data_fingerprint="fingerprint", description="Task context " * description_size,
        classes=["common", "rare"], seed=42,
    )


def _diagnostics(trial_id, count=200):
    return {
        "validation": {
            "strategy": "forward_group", "coverage": 0.91, "mean": 0.73, "std": 0.04,
            "folds": [{"fold": i, "rows": 1000, "metric": 0.7 + i / 100, "training_seconds": 9000} for i in range(5)],
            "warning": f"validation marker {trial_id}",
        },
        "per_class": [
            {"label": f"class_{trial_id}_{i:03d}_" + "label" * 8, "precision": 0.2, "recall": 0.3,
             "f1": i / max(count, 1), "support": 10} for i in range(count)
        ],
        "total_classes": count,
        "feature_importance": {
            "method": "native", "interpretation": "Association only",
            "items": [{"feature": f"feature_{i:03d}", "importance": 1 / (i + 1)} for i in range(count)],
        },
    }


def _results(count=30, latest_status="invalid_code"):
    results = []
    for index in range(count):
        trial_id = f"trial_{index + 1:04d}"
        is_best = index == 0
        latest = index == count - 1
        recipe = None
        details = {"class": "sklearn.ensemble.ExtraTreesClassifier", "parameters": {"n_estimators": 180}}
        if is_best:
            recipe = {"family": "CatBoost", "parameters": {"iterations": 1234, "learning_rate": 0.48828},
                      "preprocessing": {"add_missing_indicators": False}, "fit_options": {"early_stopping_rounds": 0}}
            details = {"class": "catboost.CatBoostClassifier",
                       "parameters": {"iterations": 1234, "learning_rate": 0.48828, "auto_class_weights": "None"},
                       "parameters_source": "catboost.get_all_params",
                       "training": {"folds": [{"training_seconds": 9999, "best_iteration": 1234} for _ in range(5)]}}
        status = latest_status if latest else "ok"
        results.append(ExperimentResult(
            trial_id=trial_id, status=status, artifact_dir="/missing/artifact", code_hash=f"hash-{index}",
            hypothesis=f"hypothesis {index}: " + "detail " * 80, metric=(0.91 if is_best else 0.5 + index / 1000),
            seconds=3.5, error=("Traceback (most recent call last):\n  File \"worker.py\"\nRuntimeError: LATEST_ERROR_MARKER " + "context " * 500) if latest and latest_status != "ok" else None,
            recipe=recipe, model_details=details,
            diagnostics=_diagnostics(trial_id) if is_best or latest else _diagnostics(trial_id, 60),
        ))
    return results


def _profile(feature_count=200):
    feature_names = [f"feature_{index:03d}" for index in range(feature_count)]
    return {
        "task_type": "classification", "metric": "macro_f1", "train_shape": [50000, feature_count],
        "test_shape": [10000, feature_count], "feature_count": feature_count,
        "categorical_columns": feature_names[:20],
        "target": {"name": "label", "dtype": "object", "unique": 2,
                   "class_counts": {"common": 48000, "rare": 2000}},
        "available_packages": {"catboost": "1.2.8", "lightgbm": "4.6.0", "xgboost": "2.1.4", "scikit-learn": "1.5.2"},
        "validation_plan": {"strategy": "forward_group", "fold_count": 5, "coverage": 0.91,
                            "time_column": "feature_000", "warnings": ["forward-only marker"]},
        "features": {name: {"dtype": "float64", "missing_fraction": i / 250,
                            "unique": 500 + i, "non_finite": i % 2, "numeric": True,
                            "unused_detail": "x" * 2000} for i, name in enumerate(feature_names)},
        "dataset_diagnostics": {
            "temporal_candidates": [{"feature": "feature_000", "future_only": True, "score": 0.99}],
            "adversarial_cv": {"status": "ok", "metric": "roc_auc", "roc_auc_mean": 0.97,
                               "roc_auc_std": 0.01, "fold_roc_auc": [0.96, 0.97, 0.98],
                               "feature_names": feature_names, "feature_count": feature_count,
                               "huge_unused": "y" * 20000},
            "features": {name: {"shift": i / 100, "missing_delta": i / 1000, "unused": "z" * 1000}
                         for i, name in enumerate(feature_names)},
            "total_features": feature_count, "features_partial": False,
        },
        "research_strategy": {"facts": {"feature_count": feature_count, "unused": "q" * 20000},
                              "candidate_shortlist": [{"model_family": name, "reason": "rationale " * 100}
                                                      for name in ("CatBoost", "ExtraTrees", "Linear")]},
        "family_comparison": {
            "selected_trial": "trial_0001", "metric": "macro_f1",
            "trials": [{"family": "CatBoost", "trial_id": "trial_0001", "status": "ok", "metric": 0.91,
                        "seconds": 10, "model_details": {"parameters": {"learning_rate": 0.48828},
                                                          "training": {"folds": [{"training_seconds": 9999} for _ in range(5)]}}}],
        },
    }


@pytest.mark.parametrize(("context_chars", "compact"), [(48000, False), (24000, True)])
def test_recipe_prompt_is_hard_bounded_and_preserves_best_and_latest_evidence(context_chars, compact):
    task, profile, results = _task(), _profile(), _results()
    best = results[0]
    task_before, profile_before, results_before = copy.deepcopy(task.to_dict()), copy.deepcopy(profile), copy.deepcopy(results)
    limit = max(5000, context_chars // 2) if compact else max(5000, context_chars)

    messages = build_messages(task, profile, results, best,
                              {"n_jobs": 4, "parallel_trials": 2, "experiment_mode": "recipe"},
                              feedback="last tool feedback", context_chars=context_chars, compact=compact)
    content = json.loads(messages[1]["content"])

    assert len(messages[1]["content"]) <= limit
    assert content["task"]["task_type"] == "classification"
    assert content["task"]["metric"] == "macro_f1"
    assert "CLASSIFICATION" in content["task_constraints"]
    assert content["data_profile"]["target"]["class_counts"] == {"common": 48000, "rare": 2000}
    assert content["data_profile"]["validation_plan"]["coverage"] == 0.91
    assert content["data_profile"]["validation_plan"]["strategy"] == "forward_group"
    assert content["data_profile"]["dataset_diagnostics"]["adversarial_cv"]["roc_auc_mean"] == 0.97
    assert content["best_experiment"]["trial_id"] == "trial_0001", "best must remain available outside the last eight"
    assert content["best_experiment"]["recipe"]["parameters"]["learning_rate"] == 0.48828
    assert content["best_experiment"]["model_details"]["parameters"]["learning_rate"] == 0.48828
    assert any(row["label"].startswith("class_trial_0001") for row in content["best_experiment"]["diagnostics"]["per_class"])
    assert content["best_experiment"]["diagnostics"]["feature_importance"]["items"]
    assert "LATEST_ERROR_MARKER" in content["latest_experiment"]["error"]
    assert "training_seconds" not in json.dumps(content["family_comparison"])
    assert all("diagnostics" not in row for row in content["experiments"] if row["trial_id"] not in {"trial_0001", "trial_0030"})

    assert task.to_dict() == task_before
    assert profile == profile_before
    assert results == results_before


def test_recipe_prompt_drops_stale_errors_after_latest_success():
    task, profile, results = _task(10, 100), _profile(10), _results(10, latest_status="ok")
    best = results[0]

    messages = build_messages(task, profile, results, best,
                              {"n_jobs": 2, "experiment_mode": "recipe"}, context_chars=24000, compact=True)
    content = json.loads(messages[1]["content"])

    assert content["latest_experiment"]["status"] == "ok"
    assert "error" not in content["latest_experiment"]
    assert "LATEST_ERROR_MARKER" not in messages[1]["content"]


def test_recipe_catalog_prefers_best_measured_base_and_prompt_runs_to_reserve():
    task, profile, results = _task(10, 100), _profile(10), _results(3)
    results[1].recipe = {"family": "CatBoost", "parameters": {"iterations": 900}, "preprocessing": {}}
    results[1].model_details = {"parameters": {"iterations": 900, "learning_rate": 0.05}}

    messages = build_messages(task, profile, results, results[0],
                              {"n_jobs": 2, "experiment_mode": "recipe"}, compact=True)
    content = json.loads(messages[1]["content"])
    measured = next(row for row in content["model_catalog"]["measured_bases"] if row["family"] == "CatBoost")

    assert measured["trial_id"] == "trial_0001"
    assert measured["parameters"]["learning_rate"] == 0.48828
    assert "Continue proposing measured experiments until the reserved deadline" in messages[0]["content"]
    assert "Stop at the reserved deadline or after measured convergence" not in messages[0]["content"]


def test_code_mode_keeps_its_existing_code_context_contract(tmp_path):
    task, profile, results = _task(2, 10), _profile(2), _results(2, latest_status="ok")
    code_path = Path(results[0].artifact_dir)
    # Keep the fixture self-contained without creating or reading a real artifact.
    results[0].artifact_dir = str(tmp_path)
    (tmp_path / "code.py").write_text("def build_model(context):\n    return None\n", encoding="utf-8")

    messages = build_messages(task, profile, results, results[0],
                              {"n_jobs": 2, "experiment_mode": "code"}, context_chars=16000)
    content = json.loads(messages[1]["content"])

    assert "build_model(context)" in content["best_code"]
    assert "model_catalog" not in content
