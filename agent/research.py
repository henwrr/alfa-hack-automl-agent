"""Цикл AutoResearch: гипотеза -> эксперимент -> оценка на CV -> принять/отклонить."""
import copy
import json
import re

import numpy as np
from scipy.stats import skew

from . import metrics as MT
from .config import log, remaining
from .data import apply_actions, validate_action
from .journal import Journal
from .models import fit_cv, make_folds
from .postprocess import blend, tune_class_weights
from .submission import write_submission

LGBM_PARAM_BOUNDS = {
    "num_leaves": (4, 512, int), "min_child_samples": (1, 1000, int), "max_depth": (-1, 16, int),
    "learning_rate": (0.01, 0.2, float), "colsample_bytree": (0.1, 1.0, float),
    "subsample": (0.3, 1.0, float), "reg_lambda": (0.0, 100.0, float), "reg_alpha": (0.0, 100.0, float),
    "min_split_gain": (0.0, 10.0, float), "max_bin": (31, 1023, int),
}

SYSTEM = (
    "Ты сильный ML-инженер (Kaggle Grandmaster) и ведёшь исследование табличной задачи. "
    "Ты не пишешь код, а предлагаешь гипотезы в виде действий из фиксированного набора. "
    "Опирайся на смысл признаков, журнал экспериментов и важности признаков. "
    "Не повторяй отклонённые гипотезы. Предлагай конкретные, проверяемые изменения."
)

ACTIONS_DOC = """Допустимые действия (JSON-объекты), имена столбцов — только из списка признаков:
- {"type":"ratio","a":NUM,"b":NUM}            новый признак a/b
- {"type":"product","a":NUM,"b":NUM}          a*b
- {"type":"diff","a":NUM,"b":NUM}             a-b
- {"type":"log","col":NUM}                    sign(x)*log1p(|x|)
- {"type":"sum","cols":[NUM,...]}             сумма по строке
- {"type":"row_stats","cols":[NUM,...],"stats":["mean","std","min","max","median"]}  статистики по строке (напр. для помесячных величин)
- {"type":"group_agg","group":COL,"col":NUM,"agg":"mean|std|median|min|max"}  агрегат признака по группе (без таргета)
- {"type":"freq","col":COL}                   частота значения
- {"type":"count_nan","cols":[COL,...]}       число пропусков в строке
- {"type":"drop","cols":[COL,...]}            удалить признаки
- {"type":"lgbm_params","params":{...}}       гиперпараметры LightGBM: num_leaves, min_child_samples, max_depth, learning_rate, colsample_bytree, subsample, reg_lambda, reg_alpha, min_split_gain, max_bin
- {"type":"target_transform","value":"log1p|none"}  только регрессия, таргет >= 0
- {"type":"class_weight","value":true|false}  балансировка классов
Ответ: {"hypotheses":[{"hypothesis":"кратко, почему это должно помочь","actions":[...]}]} — от 1 до 3 гипотез."""


def cfg_key(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


class Researcher:
    def __init__(self, spec, ds, llm):
        self.spec, self.ds, self.llm = spec, ds, llm
        self.metric = spec.metric
        self.is_reg = spec.task_type == "regression"
        if self.is_reg:
            self.y = ds.y_raw.astype(float).values
            self.classes = None
            self.n_classes = 0
        else:
            self.classes = np.sort(ds.y_raw.unique())
            idx = {c: i for i, c in enumerate(self.classes)}
            self.y = ds.y_raw.map(idx).astype(int).values
            self.n_classes = len(self.classes)
        n = len(self.y)
        self.n_splits = 5 if n <= 200_000 else 3
        self.folds = make_folds(self.y, spec.task_type, self.n_splits)
        self.journal = Journal(spec.task_dir / "agent_journal.json", self.metric)
        self.best_cfg = {"model": "lgbm", "params": {}, "target_transform": None,
                         "class_weight": False, "fe": []}
        if spec.task_type == "multiclass":
            # мультикласс в LightGBM дорогой: исследуем на более высоком lr, финал — на низком
            self.best_cfg["params"]["learning_rate"] = 0.1
        self.best_res = None
        self.pool = {}  # model -> (cfg, result)
        self.best_written = -np.inf
        self.tried = set()

    # ---------- базовые операции ----------
    def evaluate(self, cfg):
        X_all = apply_actions(self.ds.base, cfg["fe"])
        n = len(self.y)
        X, Xt = X_all.iloc[:n].reset_index(drop=True), X_all.iloc[n:].reset_index(drop=True)
        cats = [c for c in self.ds.cat_cols if c in X.columns]
        return fit_cv(cfg, X, Xt, cats, self.y, self.spec.task_type, self.n_classes, self.metric, self.folds)

    def s(self, res):
        return MT.signed(self.metric, res["score"])

    def is_better(self, new, old):
        if old is None:
            return True
        d = np.array([MT.signed(self.metric, a) - MT.signed(self.metric, b)
                      for a, b in zip(new["fold_scores"], old["fold_scores"])])
        mean = d.mean()
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        return mean > 0 and mean > 0.5 * se and self.s(new) > self.s(old)

    def maybe_write(self, res, tag):
        if self.s(res) > self.best_written:
            write_submission(self.spec, self.ds, res["test"], classes=self.classes, tag=tag)
            self.best_written = self.s(res)

    def est_time(self, factor=1.0):
        base = self.best_res["time"] if self.best_res else 60.0
        return base * factor

    def final_reserve(self):
        return self.est_time(4.0) + self.est_time(3.0) + 90

    def time_ok(self, factor=1.0):
        return remaining() > self.final_reserve() + 1.5 * self.est_time(factor)

    def try_cfg(self, cfg, name, hypothesis, force=False):
        key = cfg_key(cfg)
        if key in self.tried:
            return False
        self.tried.add(key)
        if not force and not self.time_ok():
            return False
        try:
            res = self.evaluate(cfg)
        except Exception as e:  # noqa: BLE001
            log(f"  ✗ {name}: error {repr(e)[:200]}")
            self.journal.add(name, hypothesis, cfg, None, False, note=f"error: {repr(e)[:150]}")
            return False
        acc = self.is_better(res, self.best_res)
        old = self.best_res["score"] if self.best_res else None
        log(f"  {'✓' if acc else '·'} {name}: {self.metric}={res['score']:.5f} (best {old}) t={res['time']:.0f}s")
        self.journal.add(name, hypothesis, cfg, res, acc)
        if acc:
            self.best_cfg, self.best_res = cfg, res
            self.pool["lgbm"] = (cfg, res)
            self.maybe_write(res, name)
        return acc

    def with_delta(self, actions):
        cfg = copy.deepcopy(self.best_cfg)
        for a in actions:
            t = a.get("type") if isinstance(a, dict) else None
            if t == "lgbm_params":
                for k, v in (a.get("params") or {}).items():
                    if k in LGBM_PARAM_BOUNDS:
                        lo, hi, tp = LGBM_PARAM_BOUNDS[k]
                        try:
                            cfg["params"][k] = tp(min(max(float(v), lo), hi))
                        except (TypeError, ValueError):
                            pass
            elif t == "target_transform":
                if self.is_reg and a.get("value") in ("log1p", "none") and self.y.min() >= 0:
                    cfg["target_transform"] = None if a["value"] == "none" else "log1p"
            elif t == "class_weight":
                if not self.is_reg:
                    cfg["class_weight"] = bool(a.get("value"))
            else:
                v = validate_action(a, self.ds)
                if v is not None and v not in cfg["fe"]:
                    cfg["fe"].append(v)
        return cfg

    # ---------- эвристические гипотезы ----------
    def heuristic_hypotheses(self):
        H = []
        ds = self.ds
        if self.is_reg and self.y.min() >= 0 and skew(self.y) > 1.0:
            H.append(("target_log1p", "Таргет скошен вправо — обучение на log1p стабилизирует ошибку",
                      [{"type": "target_transform", "value": "log1p"}]))
        if not self.is_reg and self.metric in ("macro_f1", "balanced_accuracy", "f1"):
            cnt = np.bincount(self.y)
            if cnt.max() / max(cnt.min(), 1) > 3:
                H.append(("class_weight", "Сильный дисбаланс классов, метрика чувствительна к редким классам",
                          [{"type": "class_weight", "value": True}]))
        if ds.cat_cols:
            H.append(("freq_cats", "Частотное кодирование категорий",
                      [{"type": "freq", "col": c} for c in ds.cat_cols[:30]]))
        # группы похожих по имени признаков -> row_stats
        groups = {}
        for c in ds.num_cols:
            toks = re.split(r"[_\-\s]+", c.lower())
            if len(toks) >= 2:
                groups.setdefault("_".join(toks[1:]), []).append(c)
                groups.setdefault("_".join(toks[:-1]), []).append(c)
        seen = set()
        bundles = []
        for k, cols in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            key = tuple(sorted(cols))
            if len(cols) >= 4 and key not in seen:
                seen.add(key)
                bundles.append({"type": "row_stats", "cols": cols, "stats": ["mean", "std", "min", "max"]})
        if bundles:
            H.append(("row_stats_groups", "Агрегаты по группам однотипных признаков", bundles[:6]))
        na_cols = [c for c in ds.feature_cols if ds.all[c].isna().mean() > 0.05]
        if len(na_cols) >= 2:
            H.append(("count_nan", "Число пропусков в строке как признак",
                      [{"type": "count_nan", "cols": na_cols}]))
        top = self.top_features(6)
        top_num = [c for c in top if c in ds.base.columns and c not in ds.cat_cols][:4]
        if len(top_num) >= 2:
            acts = [{"type": "ratio", "a": a, "b": b} for i, a in enumerate(top_num) for b in top_num[i + 1:]]
            H.append(("ratios_top", "Отношения самых важных числовых признаков", acts))
        top_cat = [c for c in top if c in ds.cat_cols][:2] or ds.cat_cols[:2]
        if top_cat and top_num:
            acts = [{"type": "group_agg", "group": g, "col": c, "agg": "mean"} for g in top_cat for c in top_num[:3]]
            H.append(("group_means", "Средние важных признаков внутри категорий", acts))
        H.append(("lgbm_bigger", "Более сложные деревья",
                  [{"type": "lgbm_params", "params": {"num_leaves": 127, "min_child_samples": 10}}]))
        H.append(("lgbm_regularized", "Более регуляризованная модель",
                  [{"type": "lgbm_params", "params": {"num_leaves": 15, "min_child_samples": 50, "reg_lambda": 5}}]))
        H.append(("lgbm_colsample", "Сильнее сэмплировать признаки",
                  [{"type": "lgbm_params", "params": {"colsample_bytree": 0.5}}]))
        return H

    def top_features(self, k):
        imp = (self.best_res or {}).get("importance") or {}
        return [c for c, _ in sorted(imp.items(), key=lambda kv: -kv[1])][:k]

    def drop_useless(self):
        imp = (self.best_res or {}).get("importance") or {}
        zero = [c for c, v in imp.items() if v <= 0 and c in self.ds.base.columns]
        if zero:
            self.try_cfg(self.with_delta([{"type": "drop", "cols": zero}]), "drop_zero_importance",
                         "Удалить признаки с нулевой важностью")

    # ---------- LLM-гипотезы ----------
    def llm_round(self):
        if self.llm is None or not self.llm.available():
            return 0
        imp = self.top_features(20)
        user = (
            f"Задача (из описания):\n{self.spec.text[:3500]}\n\n"
            f"Тип: {self.spec.task_type}, метрика: {self.metric} "
            f"({'больше' if MT.greater_is_better(self.metric) else 'меньше'} — лучше). "
            f"train={len(self.y)} строк.\n\nПризнаки:\n{self.ds.column_summary()}\n\n"
            f"Текущая лучшая конфигурация: {cfg_key(self.best_cfg)[:1500]}\n"
            f"Лучший {self.metric} на CV: {self.best_res['score']:.5f}\n"
            f"Топ признаков по важности: {imp}\n\n"
            f"Журнал экспериментов:\n{self.journal.summary(20)}\n\n{ACTIONS_DOC}"
        )
        ans = self.llm.chat_json(SYSTEM, user, max_tokens=3000)
        hyps = ans.get("hypotheses", []) if isinstance(ans, dict) else (ans if isinstance(ans, list) else [])
        n = 0
        for h in hyps[:3]:
            if not isinstance(h, dict) or not isinstance(h.get("actions"), list):
                continue
            cfg = self.with_delta(h["actions"])
            if cfg_key(cfg) == cfg_key(self.best_cfg):
                continue
            n += 1
            self.try_cfg(cfg, "llm", h.get("hypothesis", ""))
            if not self.time_ok():
                break
        return n

    # ---------- основной цикл ----------
    def run(self):
        log(f"Folds: {self.n_splits}, classes: {self.n_classes}")
        self.try_cfg(self.best_cfg, "baseline_lgbm", "LightGBM с дефолтными параметрами", force=True)
        if self.best_res is None:
            raise RuntimeError("Baseline не обучился")

        heur = self.heuristic_hypotheses()
        # сначала «дешёвые и почти всегда полезные» эвристики
        for name, hyp, acts in heur[:2]:
            if not self.time_ok():
                break
            self.try_cfg(self.with_delta(acts), name, hyp)

        # LLM-раунды чередуем с эвристиками
        rest = heur[2:]
        empty_llm_rounds = 0
        while self.time_ok():
            progressed = False
            if self.llm is not None and self.llm.available() and empty_llm_rounds < 3:
                k = self.llm_round()
                empty_llm_rounds = empty_llm_rounds + 1 if k == 0 else 0
                progressed = k > 0
            if rest:
                name, hyp, acts = rest.pop(0)
                self.try_cfg(self.with_delta(acts), name, hyp)
                progressed = True
            if not progressed:
                break
        if self.time_ok():
            self.drop_useless()

        self.final_phase()
        log(f"LLM calls={getattr(self.llm, 'calls', 0)} tokens in/out="
            f"{getattr(self.llm, 'prompt_tokens', 0)}/{getattr(self.llm, 'completion_tokens', 0)}")

    def final_phase(self):
        log("Final phase. Best research cfg:", cfg_key(self.best_cfg)[:500])
        candidates = [
            ("catboost", 4.0, {**copy.deepcopy(self.best_cfg), "model": "catboost", "params": {}}),
            ("xgb", 3.0, {**copy.deepcopy(self.best_cfg), "model": "xgb", "params": {}}),
        ]
        slow = copy.deepcopy(self.best_cfg)
        cur_lr = slow["params"].get("learning_rate", 0.05)
        slow_lr = 0.02 if cur_lr <= 0.05 else 0.03
        slow["params"]["learning_rate"] = slow_lr
        candidates.append(("lgbm_slow", 1.2 * cur_lr / slow_lr, slow))
        for name, factor, cfg in candidates:
            if remaining() < 1.3 * self.est_time(factor) + 30:
                log(f"  skip {name}: not enough time")
                continue
            try:
                res = self.evaluate(cfg)
            except Exception as e:  # noqa: BLE001
                log(f"  ✗ {name}: {repr(e)[:200]}")
                continue
            log(f"  {name}: {self.metric}={res['score']:.5f} t={res['time']:.0f}s")
            self.journal.add(name, "финальная модель для ансамбля", cfg, res, True, note="final")
            fam = "lgbm" if name == "lgbm_slow" else name
            if fam not in self.pool or self.s(res) > self.s(self.pool[fam][1]):
                self.pool[fam] = (cfg, res)
            self.maybe_write(res, name)

        results = sorted((r for _, r in self.pool.values()), key=self.s, reverse=True)
        w, oof, test = blend(results, self.y, self.metric, self.is_reg)
        bscore = MT.score(self.metric, self.y, oof)
        log(f"Blend weights={w} {self.metric}={bscore:.5f} (single best {results[0]['score']:.5f})")
        if MT.signed(self.metric, bscore) < self.s(results[0]):
            oof, test = results[0]["oof"], results[0]["test"]

        class_w = None
        if not self.is_reg and self.metric in MT.LABEL_METRICS and not self.spec.proba_output:
            class_w, gain = tune_class_weights(oof, self.y, self.metric)
            if class_w is not None:
                log(f"Class weights tuned, OOF gain={gain:.5f}: {np.round(class_w, 2).tolist()}")
                self.journal.add("class_weight_postprocess", "подбор весов классов под метрику",
                                 {"weights": np.round(class_w, 3).tolist()}, None, True,
                                 note=f"gain={gain:.5f}")
        write_submission(self.spec, self.ds, test, classes=self.classes, class_w=class_w, tag="final")
