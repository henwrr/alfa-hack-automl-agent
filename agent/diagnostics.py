"""Диагностика: ПОЧЕМУ модель ошибается. Это главный вход для гипотез LLM.

Только факты на валидации хоста: где ошибки, какие классы путаются,
какие сегменты дают основную долю ошибки, есть ли переобучение,
какие признаки работают и какие нет.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .metrics import score


def _segments_regression(frame: pd.DataFrame, residual: np.ndarray, columns: list[str], top_k: int = 4) -> list[dict]:
    """Сегменты, которые дают непропорционально большую долю квадратичной ошибки."""
    sq = residual ** 2
    total = sq.sum() or 1.0
    rows = []
    for col in columns:
        key = frame[col].astype(str) if frame[col].dtype == object or str(frame[col].dtype) == "category" \
            else pd.qcut(pd.to_numeric(frame[col], errors="coerce"), 5, duplicates="drop").astype(str)
        g = pd.DataFrame({"k": key.to_numpy(), "sq": sq, "res": residual}).groupby("k")
        stats = g.agg(n=("sq", "size"), sse=("sq", "sum"), bias=("res", "mean"))
        stats["row_share"] = stats["n"] / len(frame)
        stats["error_share"] = stats["sse"] / total
        stats["lift"] = stats["error_share"] / stats["row_share"]
        stats = stats[stats["n"] >= max(20, 0.002 * len(frame))].sort_values("sse", ascending=False).head(top_k)
        for k, r in stats.iterrows():
            rows.append({"feature": col, "value": str(k)[:40], "rows_share": round(r.row_share, 4),
                         "error_share": round(r.error_share, 4), "error_lift": round(r.lift, 2),
                         "rmse": round(float(np.sqrt(r.sse / r.n)), 4), "mean_residual(y-pred)": round(r.bias, 4)})
    rows.sort(key=lambda d: -d["error_share"] * (d["error_lift"] > 1.2))
    return rows[:12]


def diagnose(task, model, X_fit, y_fit, X_valid, y_valid, valid_pred, raw_valid: pd.DataFrame) -> dict:
    out: dict = {}
    # 1) переобучение: скор на обучающей части против валидации
    sample = np.random.default_rng(0).choice(len(X_fit), min(len(X_fit), 20000), replace=False)
    train_score = score(task, np.asarray(y_fit)[sample], model.predict(X_fit.iloc[sample]))
    valid_score = score(task, y_valid, valid_pred)
    out["fit_vs_valid"] = {"train_part_score": round(train_score, 5), "valid_score": round(valid_score, 5),
                           "note": "большой разрыв = переобучение/сдвиг; маленький разрыв при плохом скоре = недообучение"}

    # 2) важность признаков
    imp = pd.Series(model.importance, dtype=float)
    if imp.sum() > 0:
        imp = imp / imp.sum()
        out["feature_importance_top"] = {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).head(15).items()}
        out["zero_importance_features"] = imp[imp <= 1e-6].index.tolist()[:30]

    if model.best_iteration is not None:
        out["best_iteration"] = model.best_iteration

    if task.task_type == "regression":
        y = np.asarray(y_valid, float)
        p = np.asarray(valid_pred, float)
        res = y - p
        out["residuals"] = {"mean_residual(y-pred)": round(float(res.mean()), 4),
                            "rmse": round(float(np.sqrt(np.mean(res ** 2))), 4),
                            "share_of_sse_from_top1pct_errors": round(float(np.sort(res ** 2)[::-1][:max(1, len(res) // 100)].sum() / max((res ** 2).sum(), 1e-12)), 4)}
        # ошибка по квантилям таргета: недопрогноз хвостов — частая причина большого RMSE
        bins = pd.qcut(y, 10, duplicates="drop")
        by_q = pd.DataFrame({"bin": bins.astype(str), "res": res, "y": y}).groupby("bin", observed=True)
        out["error_by_target_decile"] = [
            {"y_range": k, "mean_y": round(float(g.y.mean()), 3), "mean_residual(y-pred)": round(float(g.res.mean()), 3),
             "rmse": round(float(np.sqrt((g.res ** 2).mean())), 3)} for k, g in by_q]
        cols = list(dict.fromkeys(task.cat_cols + list(out.get("feature_importance_top", {}))[:5]))
        cols = [c for c in cols if c in raw_valid.columns][:8]
        out["worst_segments"] = _segments_regression(raw_valid.reset_index(drop=True), res, cols)
        worst = np.argsort(-np.abs(res))[:5]
        show = [c for c in list(out.get("feature_importance_top", {}))[:6] if c in raw_valid.columns]
        out["largest_errors"] = [{"y": round(float(y[i]), 3), "pred": round(float(p[i]), 3),
                                  **{c: (str(raw_valid.iloc[i][c])[:30]) for c in show}} for i in worst]
    else:
        classes = [str(c) for c in task.classes]
        labels = np.asarray(classes)[np.argmax(valid_pred, axis=1)]
        y = np.asarray(y_valid).astype(str)
        prec, rec, f1, sup = precision_recall_fscore_support(y, labels, labels=classes, zero_division=0)
        out["per_class"] = [{"class": str(c), "support": int(s), "precision": round(float(pr), 3),
                             "recall": round(float(r), 3), "f1": round(float(f), 3)}
                            for c, pr, r, f, s in zip(classes, prec, rec, f1, sup)]
        out["per_class"].sort(key=lambda d: d["f1"])
        cm = confusion_matrix(y, labels, labels=classes)
        pairs = []
        for i, a in enumerate(classes):
            for j, b in enumerate(classes):
                if i != j and cm[i, j] > 0:
                    pairs.append({"true": str(a), "predicted_as": str(b), "count": int(cm[i, j]),
                                  "share_of_true_class": round(float(cm[i, j] / max(cm[i].sum(), 1)), 3)})
        out["top_confusions"] = sorted(pairs, key=lambda d: -d["count"])[:10]
        pred_dist = pd.Series(labels).value_counts(normalize=True)
        true_dist = pd.Series(y).value_counts(normalize=True)
        out["predicted_vs_true_class_share"] = {str(c): [round(float(pred_dist.get(c, 0)), 4), round(float(true_dist.get(c, 0)), 4)]
                                                for c in classes}
        conf = valid_pred.max(axis=1)
        out["confidence"] = {"mean_max_proba": round(float(conf.mean()), 3),
                             "share_below_0.5": round(float((conf < 0.5).mean()), 3)}
    return out
