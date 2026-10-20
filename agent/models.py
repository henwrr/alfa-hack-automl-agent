"""Обучение моделей по фолдам: OOF-предсказания + усреднённый прогноз на тест."""
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from .config import N_JOBS, SEED
from .data import encode_for
from . import metrics as MT

warnings.filterwarnings("ignore")


def make_folds(y, task_type, n_splits):
    if task_type == "regression":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(kf.split(np.zeros(len(y))))
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(skf.split(np.zeros(len(y)), y))
    except ValueError:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(kf.split(np.zeros(len(y))))


def _reg_objective(metric):
    return {"mae": "l1", "mape": "mape"}.get(metric, "regression")


def _cb_loss(metric, task_type):
    if task_type == "regression":
        return {"mae": "MAE", "mape": "MAPE"}.get(metric, "RMSE")
    return "Logloss" if task_type == "binary" else "MultiClass"


def _transform(y, kind):
    if kind == "log1p":
        return np.log1p(y)
    return y


def _inverse(p, kind):
    if kind == "log1p":
        return np.expm1(p)
    return p


def _full_proba(m, X, k):
    pr = m.predict_proba(X)
    cls = np.asarray(m.classes_).astype(int)
    if pr.shape[1] == k and np.array_equal(cls, np.arange(k)):
        return pr
    out = np.zeros((len(X), k))
    out[:, cls] = pr
    return out


def fit_cv(cfg: dict, X: pd.DataFrame, Xt: pd.DataFrame, cat_cols: list, y: np.ndarray,
           task_type: str, n_classes: int, metric: str, folds: list) -> dict:
    """cfg: model, params, target_transform, class_weight."""
    model = cfg.get("model", "lgbm")
    params = dict(cfg.get("params", {}))
    tt = cfg.get("target_transform")
    cw = cfg.get("class_weight", False)
    t0 = time.time()

    X, cats = encode_for(model, X, cat_cols)
    Xt, _ = encode_for(model, Xt, cat_cols)
    if model == "lgbm" or model == "xgb":
        for c in cats:  # единые категории train/test
            allcat = pd.api.types.union_categoricals([X[c], Xt[c]]).categories
            X[c] = X[c].cat.set_categories(allcat)
            Xt[c] = Xt[c].cat.set_categories(allcat)

    orig_cols = list(X.columns)
    safe, seen = [], set()
    for i, c in enumerate(orig_cols):
        sc = re.sub(r"[^0-9a-zA-Z_]", "_", str(c))[:60] or f"f{i}"
        if sc in seen:
            sc = f"{sc}_{i}"
        seen.add(sc)
        safe.append(sc)
    name_map = dict(zip(orig_cols, safe))
    X.columns, Xt.columns = safe, safe
    cats = [name_map[c] for c in cats]

    is_reg = task_type == "regression"
    oof = np.zeros(len(X)) if is_reg else np.zeros((len(X), n_classes))
    test_pred = np.zeros(len(Xt)) if is_reg else np.zeros((len(Xt), n_classes))
    fold_scores, iters = [], []

    for tr, va in folds:
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        sw = compute_sample_weight("balanced", ytr) if (cw and not is_reg) else None
        ytr_f = _transform(ytr, tt) if is_reg else ytr
        yva_f = _transform(yva, tt) if is_reg else yva

        if model == "lgbm":
            import lightgbm as lgb
            p = dict(n_estimators=10000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                     n_jobs=N_JOBS, random_state=SEED, verbose=-1)
            p.update(params)
            if is_reg:
                m = lgb.LGBMRegressor(objective=_reg_objective(metric), **p)
            else:
                m = lgb.LGBMClassifier(**p)
            m.fit(Xtr, ytr_f, sample_weight=sw, eval_set=[(Xva, yva_f)],
                  categorical_feature=cats or "auto",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            iters.append(m.best_iteration_ or p["n_estimators"])
        elif model == "xgb":
            import xgboost as xgb
            p = dict(n_estimators=10000, learning_rate=0.05, max_depth=7, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, tree_method="hist",
                     enable_categorical=True, max_cat_to_onehot=1, n_jobs=N_JOBS,
                     random_state=SEED, early_stopping_rounds=100, verbosity=0)
            p.update(params)
            if is_reg:
                obj = {"mae": "reg:absoluteerror"}.get(metric, "reg:squarederror")
                m = xgb.XGBRegressor(objective=obj, **p)
            else:
                m = xgb.XGBClassifier(**p)
            m.fit(Xtr, ytr_f, sample_weight=sw, eval_set=[(Xva, yva_f)], verbose=False)
            iters.append(getattr(m, "best_iteration", p["n_estimators"]))
        elif model == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            p = dict(iterations=5000, learning_rate=0.06, depth=6, thread_count=N_JOBS,
                     random_seed=SEED, verbose=False, od_type="Iter", od_wait=150,
                     allow_writing_files=False)
            p.update(params)
            if is_reg:
                m = CatBoostRegressor(loss_function=_cb_loss(metric, task_type), **p)
            else:
                m = CatBoostClassifier(loss_function=_cb_loss(metric, task_type),
                                       auto_class_weights="Balanced" if cw else None, **p)
            m.fit(Xtr, ytr_f, cat_features=cats, eval_set=(Xva, yva_f), use_best_model=True)
            iters.append(m.get_best_iteration() or p["iterations"])
        else:
            raise ValueError(model)

        if is_reg:
            pv = _inverse(m.predict(Xva), tt)
            pt = _inverse(m.predict(Xt), tt)
        else:
            pv = _full_proba(m, Xva, n_classes)
            pt = _full_proba(m, Xt, n_classes)
        oof[va] = pv
        test_pred += pt / len(folds)
        fold_scores.append(MT.score(metric, yva, pv))

    if is_reg:
        lo, hi = float(np.min(y)), float(np.max(y))
        oof = np.clip(oof, lo, hi)
        test_pred = np.clip(test_pred, lo, hi)

    back = dict(zip(safe, orig_cols))
    importance = None
    try:
        if model == "lgbm":
            importance = dict(zip(X.columns, m.booster_.feature_importance("gain")))
        elif model == "catboost":
            importance = dict(zip(X.columns, m.get_feature_importance()))
        elif model == "xgb":
            importance = dict(zip(X.columns, m.feature_importances_))
    except Exception:  # noqa: BLE001
        pass

    if importance is not None:
        importance = {back.get(k, k): float(v) for k, v in importance.items()}
    return {
        "oof": oof, "test": test_pred, "fold_scores": fold_scores,
        "score": MT.score(metric, y, oof), "time": time.time() - t0,
        "iters": iters, "importance": importance, "n_features": X.shape[1],
    }
