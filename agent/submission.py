"""Запись сабмита строго в формате task.md / sample_submission."""
import os

import numpy as np
import pandas as pd

from .config import log
from .metrics import PROBA_METRICS


def _atomic_write(df: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def write_submission(spec, ds, pred, classes=None, class_w=None, tag=""):
    ids = ds.test[spec.id_col].values
    if spec.task_type == "regression":
        vals = np.asarray(pred, dtype=float)
        if not np.all(np.isfinite(vals)):
            fill = float(np.nanmean(ds.y_raw))
            vals = np.where(np.isfinite(vals), vals, fill)
        df = pd.DataFrame({spec.id_col: ids, spec.target: vals})
    else:
        proba = np.asarray(pred, dtype=float)
        proba = np.nan_to_num(proba, nan=1.0 / proba.shape[1])
        if spec.proba_output:
            cols = {}
            names = [str(c) for c in classes]
            for j, col in enumerate(spec.proba_columns):
                idx = names.index(str(col)) if str(col) in names else min(j, proba.shape[1] - 1)
                cols[col] = proba[:, idx]
            df = pd.DataFrame({spec.id_col: ids, **cols})
        elif spec.metric in PROBA_METRICS and proba.shape[1] == 2:
            df = pd.DataFrame({spec.id_col: ids, spec.target: proba[:, 1]})
        else:
            adj = proba * class_w if class_w is not None else proba
            labels = np.asarray(classes)[adj.argmax(1)]
            df = pd.DataFrame({spec.id_col: ids, spec.target: labels})
    order = [c for c in spec.submission_columns if c in df.columns]
    if len(order) == df.shape[1]:
        df = df[order]
    assert len(df) == len(ds.test) and not df[spec.id_col].duplicated().any()
    assert not df.isna().any().any()
    _atomic_write(df, spec.output_path)
    log(f"Submission written ({tag}): {spec.output_path} rows={len(df)}")


def write_fallback(spec, ds):
    if spec.task_type == "regression":
        v = float(ds.y_raw.mean()) if spec.metric in ("rmse", "mse", "r2") else float(ds.y_raw.median())
        pred = np.full(len(ds.test), v)
        write_submission(spec, ds, pred, tag="fallback-constant")
    else:
        classes = np.sort(ds.y_raw.unique())
        mode = ds.y_raw.mode().iloc[0]
        proba = np.zeros((len(ds.test), len(classes)))
        proba[:, list(classes).index(mode)] = 1.0
        if spec.metric in PROBA_METRICS:
            freq = ds.y_raw.value_counts(normalize=True).reindex(classes).values
            proba = np.tile(freq, (len(ds.test), 1))
        write_submission(spec, ds, proba, classes=classes, tag="fallback-prior")
