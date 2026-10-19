"""Host-side validation diagnostics for candidate predictions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .evaluation import _read_one_column


_MAX_CLASSES = 30


def _json_label(value: Any) -> str | int:
    """Normalize a supported class label to a JSON-safe Python scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Validation labels must be finite")
        if value.is_integer():
            return int(value)
        return str(value)
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _label_key(value: str | int) -> tuple[type, str | int]:
    return type(value), value


def validation_diagnostics(task: Any, split: dict, trial_dir: Path) -> dict:
    """Return aggregate per-class validation metrics for classification tasks."""
    if getattr(task, "task_type", None) != "classification":
        return {}

    target = getattr(task, "target", None)
    if not isinstance(target, str) or not target:
        raise ValueError("Task target must be a non-empty column name")

    actual = _read_one_column(
        Path(split["validation_target"]), target, "validation_target.csv"
    )
    predicted = _read_one_column(
        Path(trial_dir) / "valid_predictions.csv", target, "valid_predictions.csv"
    )
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("Размер validation predictions не совпадает с validation target")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Validation labels и predictions не должны содержать пропуски")

    actual_labels = [_json_label(value) for value in actual.tolist()]
    predicted_labels = [_json_label(value) for value in predicted.tolist()]
    declared_labels = [
        _json_label(value)
        for value in (getattr(task, "classes", []) or [])
        if not pd.isna(value)
    ]

    # Keep declared class order, then append any validation or prediction label
    # not represented in the task metadata. This also reports zero-support labels.
    labels: list[str | int] = []
    label_indices: dict[tuple[type, str | int], int] = {}
    for label in [*declared_labels, *actual_labels, *predicted_labels]:
        key = _label_key(label)
        if key not in label_indices:
            label_indices[key] = len(labels)
            labels.append(label)

    if not labels:
        return {"per_class": [], "total_classes": 0, "partial": False}

    y_true = [label_indices[_label_key(label)] for label in actual_labels]
    y_pred = [label_indices[_label_key(label)] for label in predicted_labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )

    per_class = [
        {
            "label": labels[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(len(labels))
    ]
    per_class.sort(key=lambda item: item["f1"])

    total_classes = len(per_class)
    return {
        "per_class": per_class[:_MAX_CLASSES],
        "total_classes": total_classes,
        "partial": total_classes > _MAX_CLASSES,
    }
