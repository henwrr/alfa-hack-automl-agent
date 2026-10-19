"""Host-side scoring and independent, atomic submission validation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .contracts import TaskSpec


def _read_one_column(path: Path, expected: str, label: str) -> pd.Series:
    if not Path(path).is_file():
        raise ValueError(f"Не найден файл {label}: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if list(frame.columns) != [expected]:
        raise ValueError(f"{label} должен содержать один столбец {expected!r}, получено: {list(frame.columns)}")
    return frame[expected]


def _metric_name(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"root_mean_squared_error": "rmse", "macro_f1_score": "macro_f1", "f1_macro": "macro_f1",
               "weighted_f1_score": "weighted_f1", "auc": "roc_auc"}
    return aliases.get(normalized, normalized)


def _as_float_array(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} содержит нечисловые значения, NaN или бесконечность")
    return numeric


def _classes(task: TaskSpec, actual: pd.Series, predicted: pd.Series) -> list[Any]:
    values = list(task.classes) if task.classes else list(pd.unique(pd.concat([actual, predicted], ignore_index=True)))
    values = [value for value in values if not pd.isna(value)]
    return values


def _same_label(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        return bool(left == right)
    except Exception:
        return str(left) == str(right)


def _positive_class(labels: list[Any]) -> Any:
    """Use sklearn's conventional label 1, or the last ordered class."""
    for label in labels:
        if _same_label(label, 1):
            return label
    try:
        return sorted(labels)[-1]
    except TypeError:
        return labels[-1]


def _read_probability_artifacts(trial_dir: Path) -> tuple[np.ndarray, list[Any]] | None:
    probability_path = trial_dir / "valid_probabilities.npy"
    classes_path = trial_dir / "probability_classes.json"
    if not probability_path.exists() and not classes_path.exists():
        return None
    if not probability_path.exists() or not classes_path.exists():
        raise ValueError("Вероятности должны сопровождаться probability_classes.json")
    try:
        probabilities = np.load(probability_path, allow_pickle=False)
        class_payload = json.loads(classes_path.read_text(encoding="utf-8"))
        classes = class_payload["classes"]
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать validation probabilities: {exc}") from exc
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Размер вероятностей не совпадает со списком classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Вероятности должны быть конечными и находиться в диапазоне [0, 1]")
    row_sums = probabilities.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("Вероятности содержат строку с нулевой суммой")
    probabilities = probabilities / row_sums[:, None]
    return probabilities, classes


def score_trial(task: TaskSpec, split: dict, trial_dir: Path) -> dict[str, float]:
    """Compute metrics from host-owned labels and candidate prediction files."""
    trial_dir = Path(trial_dir)
    validation_path = Path(split["validation_target"])
    actual = _read_one_column(validation_path, task.target, "validation_target.csv")
    prediction_path = trial_dir / "valid_predictions.csv"
    predicted = _read_one_column(prediction_path, task.target, "valid_predictions.csv")
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Размер validation predictions не совпадает с validation target")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Validation labels и predictions не должны содержать пропуски")
    primary = _metric_name(task.metric)
    scores: dict[str, float] = {}

    if task.task_type == "regression":
        y_true = _as_float_array(actual, "validation_target")
        y_pred = _as_float_array(predicted, "valid_predictions")
        scores.update({
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        })
    elif task.task_type == "classification":
        labels = _classes(task, actual, predicted)
        if primary == "f1" and len(labels) != 2:
            raise ValueError("Метрика f1 поддерживается только для бинарной задачи; для нескольких классов используйте macro_f1")
        unknown_predictions = [
            value for value in pd.unique(predicted)
            if not any(_same_label(value, label) for label in labels)
        ]
        if unknown_predictions:
            raise ValueError(f"Predictions содержат неизвестные классы: {unknown_predictions[:10]}")
        scores.update({
            "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(actual, predicted)),
        })
        if len(labels) == 2:
            scores["f1"] = float(f1_score(
                actual, predicted, average="binary", pos_label=_positive_class(labels), zero_division=0
            ))
        if primary in {"roc_auc", "log_loss"}:
            artifacts = _read_probability_artifacts(trial_dir)
            if artifacts is None:
                raise ValueError(f"Метрика {primary} требует valid_probabilities.npy и probability_classes.json")
            probabilities, probability_classes = artifacts
            if len(probabilities) != len(actual):
                raise ValueError("Число строк вероятностей не совпадает с validation target")
            if len(probability_classes) != len(labels) or any(
                not any(_same_label(label, found) for found in probability_classes) for label in labels
            ):
                raise ValueError("Классы вероятностей не совпадают с классами validation задачи")
            try:
                ordered_labels = sorted(labels)
            except TypeError:
                ordered_labels = labels
            aligned = np.column_stack([
                probabilities[:, next(i for i, found in enumerate(probability_classes) if _same_label(label, found))]
                for label in ordered_labels
            ])
            if primary == "log_loss":
                scores["log_loss"] = float(log_loss(actual, aligned, labels=ordered_labels))
            elif len(ordered_labels) == 2:
                scores["roc_auc"] = float(roc_auc_score(actual, aligned[:, 1]))
            else:
                scores["roc_auc"] = float(roc_auc_score(actual, aligned, labels=ordered_labels, multi_class="ovr", average="macro"))
    else:
        raise ValueError(f"Неподдерживаемый task_type: {task.task_type}")

    if primary not in scores:
        raise ValueError(f"Неподдерживаемая для {task.task_type} метрика: {task.metric}")
    scores[task.metric] = float(scores[primary])
    return scores


def _expected_output_columns(task: TaskSpec) -> list[str]:
    columns = list(task.output_columns or [*task.id_columns, task.target])
    if columns.count(task.target) != 1:
        raise ValueError(f"target должен ровно один раз входить в output_columns: {columns}")
    if len(columns) != len(set(columns)):
        raise ValueError(f"output_columns содержит повторяющиеся имена: {columns}")
    return columns


def _read_test(task: TaskSpec) -> pd.DataFrame:
    id_dtypes = {name: "string" for name in task.id_columns}
    test = pd.read_csv(task.test_path, low_memory=False, dtype=id_dtypes)
    if not task.id_columns:
        return test
    missing = [name for name in task.id_columns if name not in test]
    if missing:
        raise ValueError(f"В test.csv отсутствуют id-столбцы: {missing}")
    if test.loc[:, task.id_columns].isna().any().any():
        raise ValueError("ID в test.csv содержит пропуски")
    if test.duplicated(subset=task.id_columns).any():
        raise ValueError("ID в test.csv не уникален")
    return test


def _compare_ids(expected: pd.DataFrame, actual: pd.DataFrame, columns: list[str]) -> bool:
    for name in columns:
        left, right = expected[name], actual[name]
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
            if not np.array_equal(pd.to_numeric(left).to_numpy(), pd.to_numeric(right).to_numpy(), equal_nan=False):
                return False
        elif not left.astype("string").equals(right.astype("string")):
            return False
    return True


def _validate_frame(task: TaskSpec, frame: pd.DataFrame, test: pd.DataFrame) -> dict:
    expected_columns = _expected_output_columns(task)
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Неверные колонки/порядок submission: ожидаются {expected_columns}, получены {list(frame.columns)}")
    if len(frame) != len(test):
        raise ValueError(f"Неверное число строк submission: ожидалось {len(test)}, получено {len(frame)}")
    id_columns = [name for name in expected_columns if name != task.target]
    if any(name not in test.columns for name in id_columns):
        raise ValueError("Output содержит столбец, отсутствующий в test.csv")
    if frame[expected_columns].isna().any().any():
        raise ValueError("Submission содержит пропуски")
    if id_columns and not _compare_ids(test.loc[:, id_columns], frame.loc[:, id_columns], id_columns):
        raise ValueError("ID в submission не совпадают с test.csv или изменён их порядок")

    values = frame[task.target]
    if task.task_type == "regression":
        _as_float_array(values, "submission target")
    elif task.task_type == "classification":
        allowed = list(task.classes)
        if not allowed:
            train = pd.read_csv(task.train_path, usecols=[task.target], low_memory=False)[task.target]
            allowed = list(pd.unique(train.dropna()))
        unknown = [value for value in pd.unique(values) if not any(_same_label(value, label) for label in allowed)]
        if unknown:
            raise ValueError(f"Submission содержит неизвестные классы: {unknown[:10]}")
    else:
        raise ValueError(f"Неподдерживаемый task_type: {task.task_type}")

    if task.sample_path:
        sample = pd.read_csv(task.sample_path, nrows=0)
        if list(sample.columns) != expected_columns:
            raise ValueError("Колонки sample_submission не совпадают с output_columns")
    return {"valid": True, "rows": int(len(frame)), "columns": expected_columns, "target": task.target}


def _atomic_publish(frame: pd.DataFrame, destination: Path) -> None:
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, na_rep="")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is not available on every supported filesystem.
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_submission(task: TaskSpec, predictions_path: Path, destination: Path) -> dict:
    """Validate model predictions, attach output IDs and atomically publish."""
    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        raise ValueError(f"Не найден файл predictions: {predictions_path}")
    id_dtypes = {name: "string" for name in task.id_columns}
    predictions = pd.read_csv(predictions_path, low_memory=False, dtype=id_dtypes)
    expected_columns = _expected_output_columns(task)
    test = _read_test(task)
    if list(predictions.columns) == [task.target]:
        if len(predictions) != len(test):
            raise ValueError(f"Неверное число prediction строк: ожидалось {len(test)}, получено {len(predictions)}")
        output = pd.DataFrame(index=range(len(test)))
        for name in expected_columns:
            output[name] = predictions[task.target].to_numpy() if name == task.target else test[name].to_numpy()
    elif list(predictions.columns) == expected_columns:
        output = predictions
    else:
        raise ValueError(f"predictions должны содержать [{task.target!r}] или все колонки {expected_columns}")
    _validate_frame(task, output, test)
    _atomic_publish(output.loc[:, expected_columns], Path(destination))
    return {"valid": True, "rows": int(len(output)), "columns": expected_columns,
            "destination": str(Path(destination).expanduser().resolve()), "target": task.target}


def validate_submission(task: TaskSpec, path: Path) -> dict:
    """Independently check an already-written submission against the task files."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Не найден submission: {path}")
    id_dtypes = {name: "string" for name in task.id_columns}
    frame = pd.read_csv(path, low_memory=False, dtype=id_dtypes)
    return _validate_frame(task, frame, _read_test(task))
