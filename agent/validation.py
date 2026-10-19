"""Схема валидации и проверка сдвига train/test.

Главная идея: валидация должна быть похожа на то, как test отличается от train.
Если test лежит «в будущем» по какому-то столбцу (год, дата), валидируемся
на последних периодах train, а не на случайных строках.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _time_like_column(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[str, pd.Series] | None:
    """Ищем столбец, где все значения test строго больше всех значений train."""
    best = None
    for col in cols:
        tr, te = train[col], test[col]
        values_tr = pd.to_numeric(tr, errors="coerce")
        values_te = pd.to_numeric(te, errors="coerce")
        if values_tr.notna().mean() < 0.95 or values_te.notna().mean() < 0.95:
            if tr.dtype == object and any(k in col.lower() for k in ("date", "time", "дата")):
                values_tr = pd.to_datetime(tr, errors="coerce")
                values_te = pd.to_datetime(te, errors="coerce")
                if values_tr.notna().mean() < 0.95 or values_te.notna().mean() < 0.95:
                    continue
                epoch = pd.Timestamp(0)
                values_tr = (values_tr - epoch).dt.total_seconds()
                values_te = (values_te - epoch).dt.total_seconds()
            else:
                continue
        if values_te.min() > values_tr.max() and values_tr.nunique() > 1:
            share = values_tr.nunique() / len(values_tr)
            if best is None or share < best[2]:
                best = (col, values_tr, share)
    return (best[0], best[1]) if best else None


def choose_split(task, train: pd.DataFrame, test: pd.DataFrame, seed: int = 42, valid_frac: float = 0.2) -> dict:
    y = train[task.target]
    idx = np.arange(len(train))
    time_col = _time_like_column(train, test, task.feature_cols)
    if time_col is not None:
        col, values = time_col
        # Берём последние периоды, пока не наберём ~valid_frac строк.
        order = np.sort(values.dropna().unique())[::-1]
        threshold, taken = order[0], 0
        for v in order:
            taken = int((values >= v).sum())
            threshold = v
            if taken >= valid_frac * len(train):
                break
        valid_mask = (values >= threshold).to_numpy()
        if 0.05 <= valid_mask.mean() <= 0.5:
            return {"kind": "time", "column": col, "threshold": float(threshold),
                    "fit_idx": idx[~valid_mask], "valid_idx": idx[valid_mask],
                    "note": (f"test целиком позже train по '{col}' (test.min > train.max) — "
                             f"валидация на последних периодах train ({col} >= {threshold}), "
                             "случайный holdout был бы слишком оптимистичен")}
    stratify = None
    if task.task_type == "classification":
        counts = y.value_counts()
        if counts.min() >= 2:
            stratify = y
    fit_idx, valid_idx = train_test_split(idx, test_size=valid_frac, random_state=seed, stratify=stratify)
    kind = "stratified_holdout" if stratify is not None else "random_holdout"
    return {"kind": kind, "column": None, "fit_idx": np.sort(fit_idx), "valid_idx": np.sort(valid_idx),
            "note": f"{kind} {int(valid_frac * 100)}%: явного временного сдвига test относительно train не найдено"}


def adversarial_check(task, train: pd.DataFrame, test: pd.DataFrame, seed: int = 42, n_jobs: int = 8,
                      exclude: list[str] | None = None) -> dict:
    """Насколько test отличим от train: AUC классификатора train-vs-test и самые «сдвинутые» признаки."""
    import lightgbm as lgb

    cols = [c for c in task.feature_cols if c not in (exclude or [])]
    if not cols:
        return {}
    rng = np.random.default_rng(seed)
    n = min(len(train), len(test), 20000)
    a = train[cols].iloc[rng.choice(len(train), n, replace=False)]
    b = test[cols].iloc[rng.choice(len(test), n, replace=False)]
    X = pd.concat([a, b], ignore_index=True)
    for c in cols:
        if c in task.cat_cols:
            X[c] = X[c].astype(str).astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    y = np.r_[np.zeros(n), np.ones(n)]
    oof = np.zeros(len(y))
    importance = np.zeros(len(cols))
    for fit, val in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, verbose=-1,
                                   n_jobs=n_jobs, random_state=seed)
        model.fit(X.iloc[fit], y[fit])
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]
        importance += model.booster_.feature_importance("gain")
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, oof))
    share = importance / max(importance.sum(), 1e-9)
    top = sorted(zip(cols, share), key=lambda t: -t[1])[:8]
    verdict = ("сильный сдвиг: test заметно отличается от train" if auc > 0.8 else
               "умеренный сдвиг" if auc > 0.6 else "train и test похожи")
    return {"auc": round(auc, 4), "verdict": verdict, "excluded": exclude or [],
            "top_shifted_features": [{"feature": c, "importance_share": round(float(s), 3)} for c, s in top]}
