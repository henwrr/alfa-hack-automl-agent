"""Isolated model-fitting subprocess; validation labels stay in the host."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, na_rep="")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_features(path: str, columns: list[str], categorical: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.reindex(columns=columns)
    for name in columns:
        if name in categorical:
            continue
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def _load_model(code_path: Path):
    if not code_path.is_file():
        raise FileNotFoundError(f"Не найден код модели: {code_path}")
    spec = importlib.util.spec_from_file_location(f"automl_candidate_{os.getpid()}", code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось импортировать код модели: {code_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    builder = getattr(module, "build_model", None)
    if not callable(builder):
        raise TypeError("Код модели должен определять build_model(context: dict)")
    return module, builder


def _run_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest должен быть JSON-объектом")
    mode = manifest.get("mode")
    if mode not in {"evaluate", "finalize"}:
        raise ValueError("mode должен быть evaluate или finalize")
    inputs = manifest.get("inputs")
    context = manifest.get("context")
    if not isinstance(inputs, dict) or not isinstance(context, dict):
        raise ValueError("Manifest должен содержать объекты inputs и context")
    if "validation_target" in inputs or "validation_target" in context:
        raise ValueError("Worker не принимает validation_target: его хранит и оценивает host")
    required = {"train_features", "train_target", "test_features"}
    if mode == "evaluate":
        required.add("validation_features")
    missing = required - set(inputs)
    if missing:
        raise ValueError(f"В manifest не хватает файлов: {sorted(missing)}")
    columns = context.get("feature_columns")
    target = context.get("target")
    task_type = context.get("task_type")
    if not isinstance(columns, list) or not columns or not isinstance(target, str):
        raise ValueError("context должен задавать target и непустой feature_columns")
    if task_type not in {"regression", "classification"}:
        raise ValueError("context.task_type должен быть regression или classification")
    categorical = set(context.get("categorical_columns", []))
    train_x = _load_features(inputs["train_features"], columns, categorical)
    train_y_frame = pd.read_csv(inputs["train_target"], low_memory=False)
    if train_y_frame.shape[1] != 1 or train_y_frame.columns[0] != target:
        raise ValueError(f"train_target.csv должен содержать один столбец {target!r}")
    if len(train_x) != len(train_y_frame) or train_x.empty:
        raise ValueError("Число строк X/y не совпадает или train пуст")
    train_y = train_y_frame[target]
    if train_y.isna().any():
        raise ValueError("train_target.csv содержит пропуски")

    validation_x = None
    if mode == "evaluate":
        validation_x = _load_features(inputs["validation_features"], columns, categorical)
    test_x = _load_features(inputs["test_features"], columns, categorical)
    _, builder = _load_model(Path(manifest["code_path"]).expanduser().resolve())
    model = builder(context)
    if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
        raise TypeError("build_model(context) должен вернуть estimator с fit и predict")
    model.fit(train_x, train_y.to_numpy())

    artifact_dir = Path(manifest["artifact_dir"]).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if validation_x is not None:
        predictions = np.asarray(model.predict(validation_x)).reshape(-1)
        if len(predictions) != len(validation_x):
            raise ValueError("Число validation predictions не совпадает с validation_features")
        _validate_predictions(predictions, task_type, "validation")
        _atomic_csv(pd.DataFrame({target: predictions}), artifact_dir / "valid_predictions.csv")

    test_predictions = np.asarray(model.predict(test_x)).reshape(-1)
    if len(test_predictions) != len(test_x):
        raise ValueError("Число test predictions не совпадает с test_features")
    _validate_predictions(test_predictions, task_type, "test")
    _atomic_csv(pd.DataFrame({target: test_predictions}), artifact_dir / "test_predictions.csv")

    probability_written = False
    if task_type == "classification" and callable(getattr(model, "predict_proba", None)):
        probability_method = getattr(model, "predict_proba")
        if callable(probability_method):
            probabilities = np.asarray(probability_method(validation_x)) if validation_x is not None else None
            if probabilities is not None:
                if probabilities.ndim != 2 or probabilities.shape[0] != len(validation_x):
                    raise ValueError("predict_proba вернул массив неподходящей формы")
                if not np.isfinite(probabilities).all():
                    raise ValueError("predict_proba вернул NaN или бесконечность")
                classes = getattr(model, "classes_", None)
                if classes is None or len(classes) != probabilities.shape[1]:
                    raise ValueError("Для вероятностей модель должна предоставить classes_")
                probability_path = artifact_dir / "valid_probabilities.npy"
                fd, temporary = tempfile.mkstemp(prefix=".valid_probabilities.", suffix=".npy", dir=artifact_dir)
                os.close(fd)
                try:
                    np.save(temporary, probabilities)
                    os.replace(temporary, probability_path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                _atomic_json({"classes": [_json_value(item) for item in classes]}, artifact_dir / "probability_classes.json")
                probability_written = True

    model_path = None
    if bool(manifest.get("save_model", False)):
        model_path = artifact_dir / "model.joblib"
        fd, temporary = tempfile.mkstemp(prefix=".model.", suffix=".joblib", dir=artifact_dir)
        os.close(fd)
        try:
            joblib.dump(model, temporary)
            os.replace(temporary, model_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    result = {
        "status": "ok",
        "mode": mode,
        "train_rows": int(len(train_x)),
        "validation_rows": int(len(validation_x)) if validation_x is not None else None,
        "test_rows": int(len(test_x)),
        "target": target,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "valid_predictions": "valid_predictions.csv" if validation_x is not None else None,
        "test_predictions": "test_predictions.csv",
        "probabilities": "valid_probabilities.npy" if probability_written else None,
        "model_path": str(model_path) if model_path else None,
    }
    _atomic_json(result, artifact_dir / "worker_result.json")
    return result


def _validate_predictions(predictions: np.ndarray, task_type: str, label: str) -> None:
    if pd.isna(predictions).any():
        raise ValueError(f"{label} predictions содержат пропуски")
    if task_type == "regression":
        try:
            numeric = predictions.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} regression predictions должны быть числовыми") from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label} regression predictions содержат NaN или бесконечность")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and run one AutoML candidate")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _run_manifest(args.manifest.expanduser().resolve())
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
