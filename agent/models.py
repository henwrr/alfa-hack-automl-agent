"""Модели: единый интерфейс fit с весами, early stopping и преобразованием таргета.

Спецификация эксперимента (её пишет LLM):
{
  "family": "lightgbm" | "catboost" | "xgboost" | "extratrees" | "linear",
  "params": {...},                 # гиперпараметры семейства (поверх разумных дефолтов)
  "target_transform": "none" | "log1p" | "sqrt",   # только регрессия
  "class_weight": "none" | "balanced" | "sqrt_balanced",  # только классификация
  "features": [ ...операции из features.OPS_DOC... ]
}
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FAMILIES = ("lightgbm", "catboost", "xgboost", "extratrees", "linear")

DEFAULTS = {
    "lightgbm": {"n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
                 "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "catboost": {"iterations": 3000, "learning_rate": 0.08, "depth": 6},
    "xgboost": {"n_estimators": 3000, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_weight": 1.0},
    "extratrees": {"n_estimators": 400, "min_samples_leaf": 2, "max_features": 0.7},
    "linear": {"alpha": 1.0, "C": 1.0},
}
EARLY_STOPPING = 100


class TimeBudget:
    """Колбэки останавливают обучение, если эксперимент выходит за свой лимит времени."""

    def __init__(self, seconds: float):
        self.deadline = time.time() + max(5.0, seconds)

    @property
    def expired(self) -> bool:
        return time.time() > self.deadline


def _sample_weight(y: np.ndarray, mode: str) -> np.ndarray | None:
    if mode in (None, "none"):
        return None
    values, inverse, counts = np.unique(y.astype(str), return_inverse=True, return_counts=True)
    w = len(y) / (len(values) * counts)
    if mode == "sqrt_balanced":
        w = np.sqrt(w)
    return w[inverse].astype(float)


def _prepare(X: pd.DataFrame, cats: list[str], family: str) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if c in cats:
            if family == "catboost":
                X[c] = X[c].astype(str).fillna("__nan__")
            else:
                X[c] = X[c].astype(str).astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def _align_categories(frames: list[pd.DataFrame], cats: list[str]) -> list[pd.DataFrame]:
    """Одинаковые коды категорий во всех частях (fit/valid/test)."""
    for c in cats:
        if c in frames[0].columns and str(frames[0][c].dtype) == "category":
            union = pd.api.types.union_categoricals([f[c] for f in frames]).categories
            for f in frames:
                f[c] = f[c].cat.set_categories(union)
    return frames


class Model:
    """Обёртка: одна спецификация -> обучение/предсказание; помнит лучшую итерацию."""

    def __init__(self, spec: dict, task, seed: int = 42, n_jobs: int = 8):
        self.spec = spec
        self.task = task
        self.family = spec.get("family", "lightgbm")
        if self.family not in FAMILIES:
            raise ValueError(f"неизвестное семейство {self.family}; доступны {FAMILIES}")
        self.params = {**DEFAULTS[self.family], **(spec.get("params") or {})}
        self.seed = seed
        self.n_jobs = n_jobs
        self.best_iteration: int | None = None
        self.model = None
        self.importance: dict[str, float] = {}

    # --- таргет ---
    def _y_forward(self, y):
        if self.task.task_type == "classification":
            index = {str(c): i for i, c in enumerate(self.task.classes)}
            return np.array([index[str(v)] for v in np.asarray(y)], dtype=int)
        y = np.asarray(y, float)
        mode = self.spec.get("target_transform", "none")
        if mode == "log1p":
            if (y <= -1).any():
                raise ValueError("log1p: таргет содержит значения <= -1")
            return np.log1p(y)
        if mode == "sqrt":
            if (y < 0).any():
                raise ValueError("sqrt: таргет содержит отрицательные значения")
            return np.sqrt(y)
        return y

    def _y_inverse(self, p):
        mode = self.spec.get("target_transform", "none")
        if mode == "log1p":
            return np.expm1(p)
        if mode == "sqrt":
            return np.square(np.clip(p, 0, None))
        return p

    @property
    def n_classes(self):
        return len(self.task.classes)

    # --- обучение ---
    def fit(self, X: pd.DataFrame, y, cats: list[str], X_extra: list[pd.DataFrame] = (), seconds: float = 600,
            n_estimators: int | None = None) -> "Model":
        """Если n_estimators не задан — early stopping на внутреннем куске (10%) обучающей части.

        Валидационная выборка хоста в обучении НЕ участвует — иначе её скор оптимистичен.
        """
        budget = TimeBudget(seconds)
        self.cats = [c for c in cats if c in X.columns]
        self.columns = list(X.columns)
        y_enc = self._y_forward(y)
        Xp = _prepare(X, self.cats, self.family)
        self._extra_template = [_prepare(e, self.cats, self.family) for e in X_extra]
        if self.family != "catboost":
            _align_categories([Xp, *self._extra_template], self.cats)
            self._categories = {c: Xp[c].cat.categories for c in self.cats if str(Xp[c].dtype) == "category"}
        weights = _sample_weight(np.asarray(y), self.spec.get("class_weight", "none")) \
            if self.task.task_type == "classification" else None

        use_es = n_estimators is None and self.family in {"lightgbm", "catboost", "xgboost"}
        if use_es:
            strat = y_enc if self.task.task_type == "classification" and np.bincount(y_enc).min() >= 2 else None
            tr, es = train_test_split(np.arange(len(Xp)), test_size=0.1, random_state=self.seed, stratify=strat)
        else:
            tr, es = np.arange(len(Xp)), None
        w_tr = weights[tr] if weights is not None else None
        getattr(self, f"_fit_{self.family}")(Xp, y_enc, tr, es, w_tr, weights, budget, n_estimators)
        return self

    def _fit_lightgbm(self, X, y, tr, es, w_tr, w, budget, n_estimators):
        import lightgbm as lgb

        params = dict(self.params, random_state=self.seed, n_jobs=self.n_jobs, verbose=-1)
        if n_estimators:
            params["n_estimators"] = n_estimators
        if self.task.task_type == "classification":
            params.setdefault("objective", "multiclass" if self.n_classes > 2 else "binary")
            est = lgb.LGBMClassifier(**params)
        else:
            est = lgb.LGBMRegressor(**params)

        def stop_on_time(env):
            if budget.expired:
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

        callbacks = [stop_on_time]
        fit_kw = {"sample_weight": w_tr, "categorical_feature": [c for c in self.cats if c in X.columns] or "auto"}
        if es is not None:
            fit_kw["eval_set"] = [(X.iloc[es], y[es])]
            fit_kw["eval_sample_weight"] = [w[es]] if w is not None else None
            callbacks.append(lgb.early_stopping(EARLY_STOPPING, verbose=False))
        est.fit(X.iloc[tr], y[tr], callbacks=callbacks, **fit_kw)
        self.model = est
        self.best_iteration = int(est.best_iteration_ or est.n_estimators) if es is not None else est.n_estimators
        self.importance = dict(zip(X.columns, est.booster_.feature_importance("gain").astype(float)))

    def _fit_catboost(self, X, y, tr, es, w_tr, w, budget, n_estimators):
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        params = dict(self.params, random_seed=self.seed, thread_count=self.n_jobs, verbose=False,
                      allow_writing_files=False)
        if n_estimators:
            params["iterations"] = n_estimators
        if self.task.task_type == "classification":
            params.setdefault("loss_function", "MultiClass" if self.n_classes > 2 else "Logloss")
            est = CatBoostClassifier(**params)
        else:
            params.setdefault("loss_function", "RMSE")
            est = CatBoostRegressor(**params)

        class _Stop:
            def after_iteration(self, info):
                return not budget.expired

        cats = [c for c in self.cats if c in X.columns]
        train_pool = Pool(X.iloc[tr], y[tr], cat_features=cats, weight=w_tr)
        kw = {"callbacks": [_Stop()]}
        if es is not None:
            kw["eval_set"] = Pool(X.iloc[es], y[es], cat_features=cats, weight=w[es] if w is not None else None)
            kw["early_stopping_rounds"] = EARLY_STOPPING
            kw["use_best_model"] = True
        est.fit(train_pool, **kw)
        self.model = est
        self.best_iteration = int(est.get_best_iteration() + 1) if es is not None else int(est.tree_count_)
        self.importance = dict(zip(X.columns, est.get_feature_importance().astype(float)))

    def _fit_xgboost(self, X, y, tr, es, w_tr, w, budget, n_estimators):
        import xgboost as xgb

        class _Stop(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                return budget.expired

        # Категории -> числовые коды: нативные категории XGBoost ломаются на значениях,
        # которых не было в обучении (на временном сплите давало RMSE x6).
        X = self._codes(X)
        params = dict(self.params, random_state=self.seed, n_jobs=self.n_jobs, tree_method="hist",
                      callbacks=[_Stop()])
        if n_estimators:
            params["n_estimators"] = n_estimators
        if es is not None:
            params["early_stopping_rounds"] = EARLY_STOPPING
        est = xgb.XGBClassifier(**params) if self.task.task_type == "classification" else xgb.XGBRegressor(**params)
        kw = {"sample_weight": w_tr, "verbose": False}
        if es is not None:
            kw["eval_set"] = [(X.iloc[es], y[es])]
            kw["sample_weight_eval_set"] = [w[es]] if w is not None else None
        est.fit(X.iloc[tr], y[tr], **kw)
        self.model = est
        self.best_iteration = int(est.best_iteration + 1) if es is not None else int(params["n_estimators"])
        self.importance = {k: float(v) for k, v in zip(X.columns, est.feature_importances_)}

    def _fit_extratrees(self, X, y, tr, es, w_tr, w, budget, n_estimators):
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        Xn = self._numeric_matrix(X)
        params = dict(self.params, random_state=self.seed, n_jobs=self.n_jobs)
        if n_estimators:
            params["n_estimators"] = n_estimators
        est = ExtraTreesClassifier(**params) if self.task.task_type == "classification" else ExtraTreesRegressor(**params)
        est.fit(Xn[tr], y[tr], sample_weight=w_tr)
        self.model = est
        self.best_iteration = params["n_estimators"]
        self.importance = dict(zip(X.columns, est.feature_importances_.astype(float)))

    def _fit_linear(self, X, y, tr, es, w_tr, w, budget, n_estimators):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        Xn = self._linear_matrix(X, fit=True)
        if self.task.task_type == "classification":
            est = make_pipeline(StandardScaler(), LogisticRegression(C=self.params.get("C", 1.0), max_iter=500))
            est.fit(Xn[tr], y[tr], logisticregression__sample_weight=w_tr)
        else:
            est = make_pipeline(StandardScaler(), Ridge(alpha=self.params.get("alpha", 1.0)))
            est.fit(Xn[tr], y[tr])
        self.model = est
        self.best_iteration = None
        coef = np.abs(est[-1].coef_).reshape(-1, Xn.shape[1]).mean(axis=0)
        self.importance = {f"lin_{i}": float(v) for i, v in enumerate(coef)}

    def _codes(self, X):
        out = X.copy()
        for c in self.cats:
            if str(out[c].dtype) == "category":
                out[c] = out[c].cat.codes.replace(-1, np.nan).astype(float)
        return out

    def _numeric_matrix(self, X):
        out = X.copy()
        for c in self.cats:
            out[c] = out[c].cat.codes if str(out[c].dtype) == "category" else out[c]
        return out.astype(float).fillna(-999.0).to_numpy()

    def _linear_matrix(self, X, fit=False):
        num = [c for c in X.columns if c not in self.cats]
        A = X[num].astype(float)
        if fit:
            self._lin_median = A.median()
            self._lin_dummies = {c: X[c].astype(str).value_counts().index[:30].tolist() for c in self.cats}
        A = A.fillna(self._lin_median).fillna(0.0)
        parts = [A.to_numpy(), X[num].isna().to_numpy().astype(float)]
        for c in self.cats:
            vals = X[c].astype(str)
            parts.append(np.stack([(vals == v).to_numpy() for v in self._lin_dummies[c]], axis=1).astype(float)
                         if self._lin_dummies[c] else np.zeros((len(X), 0)))
        return np.hstack(parts)

    # --- предсказание ---
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Регрессия: значения в исходной шкале. Классификация: proba [n, k] в порядке task.classes."""
        Xp = _prepare(X[self.columns], self.cats, self.family)
        if self.family != "catboost":
            for c, cats in getattr(self, "_categories", {}).items():
                Xp[c] = Xp[c].cat.set_categories(cats)
        if self.family == "xgboost":
            Xp = self._codes(Xp)
        elif self.family == "extratrees":
            Xp = self._numeric_matrix(Xp)
        elif self.family == "linear":
            Xp = self._linear_matrix(Xp)
        if self.task.task_type == "classification":
            proba = self.model.predict_proba(Xp)
            full = np.zeros((len(X), self.n_classes))
            seen = getattr(self.model, "classes_", np.arange(proba.shape[1]))
            full[:, np.asarray(seen, int)] = proba
            return full
        return self._y_inverse(np.asarray(self.model.predict(Xp), float))
