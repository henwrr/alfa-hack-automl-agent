"""Метрики считает только хост — LLM не может «нарисовать» себе скор."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    name: str
    task_type: str
    maximize: bool
    fn: Callable


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _rmsle(y, p):
    p = np.clip(np.asarray(p, float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(np.asarray(y, float)) - np.log1p(p)) ** 2)))


def _mape(y, p):
    y = np.asarray(y, float)
    return float(np.mean(np.abs((y - np.asarray(p, float)) / np.where(y == 0, 1e-9, y))))


# Для классификации fn получает (y_true_labels, proba[n, k], classes).
def _labels(proba, classes):
    return np.asarray(classes).astype(str)[np.argmax(proba, axis=1)]


def _auc(y, proba, classes):
    if len(classes) == 2:
        return float(skm.roc_auc_score(np.asarray(y) == classes[1], proba[:, 1]))
    return float(skm.roc_auc_score(y, proba, labels=classes, multi_class="ovr", average="macro"))


METRICS = {
    "rmse": Metric("rmse", "regression", False, _rmse),
    "rmsle": Metric("rmsle", "regression", False, _rmsle),
    "mae": Metric("mae", "regression", False, lambda y, p: float(skm.mean_absolute_error(y, p))),
    "mape": Metric("mape", "regression", False, _mape),
    "r2": Metric("r2", "regression", True, lambda y, p: float(skm.r2_score(y, p))),
    "macro_f1": Metric("macro_f1", "classification", True,
                       lambda y, pr, c: float(skm.f1_score(y, _labels(pr, c), labels=c, average="macro", zero_division=0))),
    "weighted_f1": Metric("weighted_f1", "classification", True,
                          lambda y, pr, c: float(skm.f1_score(y, _labels(pr, c), labels=c, average="weighted", zero_division=0))),
    "f1": Metric("f1", "classification", True,
                 lambda y, pr, c: float(skm.f1_score(np.asarray(y) == c[1], _labels(pr, c) == c[1], zero_division=0))),
    "accuracy": Metric("accuracy", "classification", True,
                       lambda y, pr, c: float(skm.accuracy_score(y, _labels(pr, c)))),
    "roc_auc": Metric("roc_auc", "classification", True, _auc),
    "log_loss": Metric("log_loss", "classification", False,
                       lambda y, pr, c: float(skm.log_loss(y, np.clip(pr, 1e-15, 1), labels=c))),
}


def normalize_metric(name: str) -> str:
    name = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"f1_macro": "macro_f1", "auc": "roc_auc", "logloss": "log_loss", "acc": "accuracy"}
    name = aliases.get(name, name)
    if name not in METRICS:
        raise ValueError(f"Неизвестная метрика: {name}")
    return name


def score(task, y_true, pred) -> float:
    metric = METRICS[task.metric]
    if task.task_type == "regression":
        return metric.fn(y_true, pred)
    # Метки как строки: смешанные/целочисленные object-метки sklearn не понимает («unknown is not supported»).
    return metric.fn(np.asarray(y_true).astype(str), pred, [str(c) for c in task.classes])


def better(a: float | None, b: float | None, maximize: bool) -> bool:
    """True, если a строго лучше b."""
    if a is None or not np.isfinite(a):
        return False
    if b is None:
        return True
    return a > b if maximize else a < b
