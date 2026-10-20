"""Метрики. Предсказания классификации хранятся как вероятности (n, K) по закодированным классам."""
import numpy as np
from sklearn import metrics as M

# name -> (greater_is_better, task_kinds)
REGISTRY = {
    "rmse": (False, "reg"), "mse": (False, "reg"), "mae": (False, "reg"),
    "mape": (False, "reg"), "rmsle": (False, "reg"), "r2": (True, "reg"),
    "macro_f1": (True, "cls"), "weighted_f1": (True, "cls"), "micro_f1": (True, "cls"),
    "f1": (True, "cls"), "accuracy": (True, "cls"), "balanced_accuracy": (True, "cls"),
    "roc_auc": (True, "cls"), "logloss": (False, "cls"),
}
LABEL_METRICS = {"macro_f1", "weighted_f1", "micro_f1", "f1", "accuracy", "balanced_accuracy"}
PROBA_METRICS = {"roc_auc", "logloss"}


def greater_is_better(name: str) -> bool:
    return REGISTRY[name][0]


def score(name: str, y, pred) -> float:
    if name == "rmse":
        return float(np.sqrt(M.mean_squared_error(y, pred)))
    if name == "mse":
        return float(M.mean_squared_error(y, pred))
    if name == "mae":
        return float(M.mean_absolute_error(y, pred))
    if name == "mape":
        return float(M.mean_absolute_percentage_error(y, pred))
    if name == "rmsle":
        return float(np.sqrt(M.mean_squared_log_error(np.clip(y, 0, None), np.clip(pred, 0, None))))
    if name == "r2":
        return float(M.r2_score(y, pred))
    # классификация: pred — вероятности (n, K)
    proba = np.asarray(pred)
    k = proba.shape[1]
    if name in LABEL_METRICS:
        lab = proba.argmax(1)
        if name == "macro_f1":
            return float(M.f1_score(y, lab, average="macro", labels=np.arange(k), zero_division=0))
        if name == "weighted_f1":
            return float(M.f1_score(y, lab, average="weighted", zero_division=0))
        if name == "micro_f1":
            return float(M.f1_score(y, lab, average="micro", zero_division=0))
        if name == "f1":
            if k == 2:
                return float(M.f1_score(y, lab, zero_division=0))
            return float(M.f1_score(y, lab, average="macro", labels=np.arange(k), zero_division=0))
        if name == "accuracy":
            return float(M.accuracy_score(y, lab))
        if name == "balanced_accuracy":
            return float(M.balanced_accuracy_score(y, lab))
    if name == "roc_auc":
        if k == 2:
            return float(M.roc_auc_score(y, proba[:, 1]))
        return float(M.roc_auc_score(y, proba, multi_class="ovr", labels=np.arange(k)))
    if name == "logloss":
        return float(M.log_loss(y, np.clip(proba, 1e-15, 1), labels=np.arange(k)))
    raise ValueError(name)


def signed(name: str, value: float) -> float:
    """Больше — лучше для любой метрики."""
    return value if greater_is_better(name) else -value
