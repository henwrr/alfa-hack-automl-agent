"""Понимание задачи: поиск файлов, target, тип задачи, метрика, профиль данных."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRICS, normalize_metric


@dataclass
class Task:
    task_dir: Path
    description: str
    train_path: Path
    test_path: Path
    sample_path: Path | None
    dictionary_path: Path | None
    id_col: str | None
    target: str
    task_type: str  # "regression" | "classification"
    metric: str
    output_mode: str = "label"  # "label" | "proba" (бинарная вероятность)
    classes: list = field(default_factory=list)
    cat_cols: list = field(default_factory=list)
    feature_cols: list = field(default_factory=list)
    dictionary: dict = field(default_factory=dict)

    @property
    def maximize(self) -> bool:
        return METRICS[self.metric].maximize

    def summary(self) -> dict:
        return {"target": self.target, "task_type": self.task_type, "metric": self.metric,
                "maximize": self.maximize, "id_col": self.id_col, "output_mode": self.output_mode,
                "classes": [str(c) for c in self.classes][:50], "n_features": len(self.feature_cols),
                "categorical": self.cat_cols}


def _find_csv(task_dir: Path, text: str, stem: str) -> Path | None:
    """Ищем файл сначала по путям из task.md, затем рекурсивно рядом с task.md."""
    for match in re.findall(r"[`'\"]?([\w./\\-]*%s[\w-]*\.csv)" % stem, text):
        candidate = (task_dir / match).resolve()
        if candidate.is_file():
            return candidate
    found = sorted(task_dir.rglob(f"{stem}*.csv"), key=lambda p: (len(p.parts), str(p)))
    found = [p for p in found if "agent_runs" not in p.parts]
    return found[0] if found else None


def _metric_from_text(text: str, task_type: str, n_classes: int) -> str:
    lower = text.lower()
    patterns = [
        ("rmsle", r"rmsle|root mean squared log"),
        ("rmse", r"\brmse\b|root mean squared error"),
        ("mape", r"\bmape\b"),
        ("mae", r"\bmae\b|mean absolute error"),
        ("r2", r"\br2\b|r²|r\^2"),
        ("macro_f1", r"macro[\s_-]*f1|f1[\s_-]*macro|average=[\"']macro"),
        ("weighted_f1", r"weighted[\s_-]*f1|f1[\s_-]*weighted"),
        ("roc_auc", r"roc[\s_-]*auc|\bauc\b|gini"),
        ("log_loss", r"log[\s_-]*loss|logloss|cross[\s_-]*entropy"),
        ("accuracy", r"\baccuracy\b|точност"),
        ("f1", r"\bf1\b"),
    ]
    # Приоритет — строки про «основную метрику» / «метрику оценки».
    lines = [l for l in lower.splitlines() if "основн" in l or "метрик" in l or "metric" in l]
    for scope in ("\n".join(l for l in lines if "основн" in l), "\n".join(lines), lower):
        hits = []
        for name, pattern in patterns:
            m = re.search(pattern, scope)
            if m:
                hits.append((m.start(), name))
        hits = [(pos, name) for pos, name in hits
                if (METRICS[name].task_type == task_type)]
        if hits:
            name = min(hits)[1]
            if name == "f1" and n_classes > 2:
                name = "macro_f1"
            return name
    if task_type == "regression":
        return "rmse"
    return "macro_f1" if n_classes > 2 else "roc_auc"


def _read_dictionary(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    cols = {c.lower(): c for c in frame.columns}
    name_col = cols.get("feature") or cols.get("column") or frame.columns[0]
    out = {}
    for _, row in frame.iterrows():
        out[str(row[name_col])] = {k: (None if pd.isna(v) else str(v)) for k, v in row.items() if k != name_col}
    return out


def discover(task_path: Path) -> tuple[Task, pd.DataFrame, pd.DataFrame]:
    task_path = Path(task_path).resolve()
    task_dir = task_path.parent
    text = task_path.read_text(encoding="utf-8")
    train_path = _find_csv(task_dir, text, "train")
    test_path = _find_csv(task_dir, text, "test")
    if train_path is None or test_path is None:
        raise FileNotFoundError(f"Не найдены train/test CSV рядом с {task_path}")
    sample_path = _find_csv(task_dir, text, "sample_submission")
    dictionary_path = _find_csv(task_dir, text, "data_dictionary")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    dictionary = _read_dictionary(dictionary_path)

    sample_cols = list(pd.read_csv(sample_path, nrows=5).columns) if sample_path else []
    id_col = None
    if sample_cols and sample_cols[0] in test.columns:
        id_col = sample_cols[0]
    elif "id" in test.columns:
        id_col = "id"
    for name, info in dictionary.items():
        if str(info.get("role", "")).lower() in {"id", "identifier"} and name in test.columns:
            id_col = name

    candidates = [c for c in train.columns if c not in test.columns and c != id_col]
    target = next((c for c in sample_cols[1:] if c in candidates), None)
    if target is None:
        target = next((n for n, i in dictionary.items() if str(i.get("role", "")).lower() == "target" and n in train.columns), None)
    if target is None:
        if len(candidates) != 1:
            raise ValueError(f"Не удалось однозначно определить target: {candidates}")
        target = candidates[0]

    y = train[target]
    lower = text.lower()
    says_cls = bool(re.search(r"классификац|classification|класс[аы]?\b", lower))
    says_reg = bool(re.search(r"регресс|regression", lower))
    if y.dtype == object or y.dtype == bool or str(y.dtype) == "category":
        task_type = "classification"
    elif says_reg and not says_cls:
        task_type = "regression"
    elif says_cls and not says_reg:
        task_type = "classification"
    else:
        is_int = np.allclose(y.dropna(), np.round(y.dropna()))
        task_type = "classification" if is_int and y.nunique() <= 30 else "regression"

    classes = sorted(y.dropna().unique().tolist(), key=str) if task_type == "classification" else []
    metric = normalize_metric(_metric_from_text(text, task_type, len(classes)))

    output_mode = "label"
    if task_type == "classification" and len(classes) == 2 and metric in {"roc_auc", "log_loss"}:
        output_mode = "proba"

    feature_cols = [c for c in test.columns if c != id_col and c in train.columns]
    cat_cols = []
    for c in feature_cols:
        dtype = str(dictionary.get(c, {}).get("dtype", "")).lower()
        if train[c].dtype == object or train[c].dtype == bool or "categ" in dtype or "string" in dtype:
            cat_cols.append(c)

    task = Task(task_dir=task_dir, description=text, train_path=train_path, test_path=test_path,
                sample_path=sample_path, dictionary_path=dictionary_path, id_col=id_col, target=target,
                task_type=task_type, metric=metric, output_mode=output_mode, classes=classes,
                cat_cols=cat_cols, feature_cols=feature_cols, dictionary=dictionary)
    return task, train, test


def profile(task: Task, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Компактный профиль данных для LLM: только факты, без интерпретаций."""
    feats = {}
    for c in task.feature_cols:
        s, t = train[c], test[c]
        info = {"dtype": "category" if c in task.cat_cols else str(s.dtype),
                "missing_train": round(float(s.isna().mean()), 4),
                "missing_test": round(float(t.isna().mean()), 4),
                "nunique": int(s.nunique())}
        if c in task.cat_cols:
            seen = set(s.dropna().astype(str))
            info["unseen_in_test_share"] = round(float((~t.dropna().astype(str).isin(seen)).mean()) if t.notna().any() else 0.0, 4)
            info["top"] = s.astype(str).value_counts().head(5).to_dict()
        else:
            num = pd.to_numeric(s, errors="coerce")
            info.update({"min": float(np.nanmin(num)) if num.notna().any() else None,
                         "max": float(np.nanmax(num)) if num.notna().any() else None,
                         "zeros_share": round(float((num == 0).mean()), 4)})
        desc = task.dictionary.get(c, {}).get("description_ru")
        if desc:
            info["description"] = desc[:120]
        feats[c] = info
    y = train[task.target]
    if task.task_type == "regression":
        target = {"mean": float(y.mean()), "std": float(y.std()), "skew": round(float(y.skew()), 3),
                  "min": float(y.min()), "max": float(y.max()),
                  "quantiles": {str(q): float(y.quantile(q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)}}
    else:
        counts = y.value_counts()
        target = {"class_counts": {str(k): int(v) for k, v in counts.items()},
                  "imbalance_ratio": round(float(counts.max() / max(counts.min(), 1)), 2)}
    dup = train[task.feature_cols].duplicated()
    conflict = 0.0
    if task.task_type == "classification" and dup.any():
        grouped = train.groupby(task.feature_cols, dropna=False)[task.target].nunique()
        conflict = float((grouped > 1).sum() / max(len(grouped), 1))
    return {"train_rows": len(train), "test_rows": len(test), "n_features": len(task.feature_cols),
            "n_categorical": len(task.cat_cols), "target": target,
            "duplicate_feature_rows_share": round(float(dup.mean()), 4),
            "conflicting_label_groups_share": round(conflict, 4), "features": feats}
