"""Исследователь: предлагает следующие эксперименты.

LLMResearcher — основной: видит диагностику ошибок и историю, пишет гипотезы.
OfflineResearcher — эвристический плейбук: работает без LLM (разработка, сбой API)
и задаёт нижнюю планку, которую LLM должна обгонять.
"""

from __future__ import annotations

import copy
import json

from .features import OPS_DOC
from .llm import LLM, extract_json
from .models import FAMILIES

_BANNED_PARAMS = {"n_jobs", "thread_count", "nthread", "num_threads", "device", "device_type", "task_type",
                  "gpu_id", "random_state", "random_seed", "seed", "verbose", "verbosity", "callbacks",
                  "train_dir", "allow_writing_files", "early_stopping_rounds", "eval_metric", "cat_features"}


def normalize_spec(spec: dict, task, best_spec: dict | None) -> dict:
    """Проверяем и чистим спецификацию от LLM. Ошибки -> ValueError с понятным текстом."""
    if not isinstance(spec, dict):
        raise ValueError("spec должен быть объектом")
    if spec.get("extends_best") and best_spec:
        base = copy.deepcopy(best_spec)
        base["params"] = {**base.get("params", {}), **(spec.get("params") or {})}
        base["features"] = base.get("features", []) + list(spec.get("features") or [])
        for key in ("family", "target_transform", "class_weight"):
            if key in spec:
                base[key] = spec[key]
        if spec.get("family") and spec["family"] != best_spec.get("family"):
            base["params"] = dict(spec.get("params") or {})
        spec = base
    family = spec.get("family", "lightgbm")
    if family not in FAMILIES:
        raise ValueError(f"family должен быть одним из {FAMILIES}")
    params = {k: v for k, v in (spec.get("params") or {}).items()
              if k not in _BANNED_PARAMS and isinstance(v, (int, float, str, bool))}
    for key in ("n_estimators", "iterations"):
        if key in params:
            params[key] = int(min(max(int(params[key]), 50), 10000))
    out = {"family": family, "params": params, "features": list(spec.get("features") or [])}
    unknown = [op.get("op") for op in out["features"] if not isinstance(op, dict) or op.get("op") not in OPS_DOC]
    if unknown:
        raise ValueError(f"неизвестные операции признаков: {unknown}")
    if task.task_type == "regression":
        out["target_transform"] = spec.get("target_transform", "none")
        if out["target_transform"] not in {"none", "log1p", "sqrt"}:
            raise ValueError("target_transform: none | log1p | sqrt")
    else:
        out["class_weight"] = spec.get("class_weight", "none")
        if out["class_weight"] not in {"none", "balanced", "sqrt_balanced"}:
            raise ValueError("class_weight: none | balanced | sqrt_balanced")
    return out


SYSTEM_PROMPT = """Ты — ведущий ML-исследователь в автономном агенте для табличных задач.
Хост сам обучает модели, сам считает метрику на СВОЕЙ валидации и сам выбирает лучшую — ты не видишь меток валидации и не можешь «объявить» скор.
Твоя работа — цикл AutoResearch: посмотреть, ПОЧЕМУ текущая лучшая модель ошибается (диагностика ниже), сформулировать проверяемые гипотезы и описать эксперименты.

Как думать:
- Начинай с диагностики: какие сегменты/классы дают основную ошибку, есть ли смещение (mean_residual), недопрогноз хвостов, путаница классов, переобучение (train_part_score vs valid_score), сдвиг train/test.
- Каждая гипотеза должна опираться на конкретный факт из диагностики или профиля и менять ОДНУ понятную вещь относительно лучшей модели (признаки ИЛИ модель ИЛИ таргет ИЛИ веса). Так результат интерпретируем.
- Не повторяй уже проверенные спецификации. Если гипотеза не подтвердилась — учти это, не долби одно направление.
- Признаки, сильно отличающиеся между train и test (adversarial check), могут вредить — проверяй их удаление.
- Не используй id и не пытайся восстановить таргет из test.

Формат спецификации эксперимента (spec):
{
  "family": "lightgbm" | "catboost" | "xgboost" | "extratrees" | "linear",
  "params": {гиперпараметры семейства; число деревьев подбирается early stopping, n_jobs/seed задаёт хост},
  "target_transform": "none" | "log1p" | "sqrt",            (только регрессия)
  "class_weight": "none" | "balanced" | "sqrt_balanced",   (только классификация)
  "features": [операции признаков, применяются по порядку],
  "extends_best": true   — взять спецификацию лучшей модели и ДОБАВИТЬ твои features / заменить указанные поля
}
Операции признаков:
%s

Ответ — ТОЛЬКО JSON без пояснений вокруг:
{
  "analysis": "2-5 предложений: что говорит диагностика о причинах ошибок",
  "experiments": [
    {"hypothesis": "что меняем и почему (со ссылкой на факт)", "expected": "какой эффект ожидаем", "spec": {...}}
  ],
  "stop": false
}
Число экспериментов — не больше max_experiments из budget. "stop": true — только если идеи исчерпаны.
Пиши analysis и hypothesis по-русски.""" % "\n".join(f"- {v}" for v in OPS_DOC.values())


def _compact_profile(profile: dict, limit: int = 60) -> dict:
    feats = dict(list(profile.get("features", {}).items())[:limit])
    return {**{k: v for k, v in profile.items() if k != "features"}, "features": feats}


class LLMResearcher:
    name = "llm"

    def __init__(self, llm: LLM, log):
        self.llm = llm
        self.log = log

    def propose(self, state: dict) -> dict:
        payload = {
            "task": state["task"], "task_description": state["description"][:6000],
            "data_profile": _compact_profile(state["profile"]), "validation": state["validation"],
            "train_test_shift": state["shift"], "history": state["history"][-25:],
            "best": state["best"], "budget": state["budget"],
            "previous_errors": state.get("errors", [])[-3:],
        }
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
        text = self.llm.chat(messages)
        try:
            answer = extract_json(text)
        except ValueError:
            messages += [{"role": "assistant", "content": text[:4000]},
                         {"role": "user", "content": "Ответ не распознан. Верни ТОЛЬКО JSON в описанном формате."}]
            answer = extract_json(self.llm.chat(messages))
        experiments = answer.get("experiments") or []
        if not isinstance(experiments, list):
            raise ValueError("experiments должен быть списком")
        return {"analysis": str(answer.get("analysis", ""))[:2000], "experiments": experiments,
                "stop": bool(answer.get("stop", False))}


class OfflineResearcher:
    """Детерминированный плейбук: следующий шаг выбирается по профилю и диагностике."""

    name = "offline"

    def __init__(self, task, profile):
        self.task = task
        self.profile = profile
        self.done: set[str] = set()

    def _plan(self, state) -> list[tuple[str, dict]]:
        task, prof = self.task, self.profile
        best = (state.get("best") or {}).get("spec") or {}
        diag = (state.get("best") or {}).get("diagnostics") or {}
        missing = [c for c, i in prof["features"].items() if i.get("missing_train", 0) > 0.05]
        cats = [c for c in task.cat_cols if prof["features"].get(c, {}).get("nunique", 0) > 2]
        plan: list[tuple[str, dict]] = []  # бейзлайн запускает сам хост
        if task.task_type == "regression":
            skew = prof["target"].get("skew", 0)
            if skew > 1:
                plan.append((f"skew таргета {skew}: log1p должен уменьшить влияние хвоста на обучение",
                             {"extends_best": True, "target_transform": "log1p"}))
            plan.append(("CatBoost с нативными категориями может лучше использовать категориальные признаки",
                         {"extends_best": True, "family": "catboost"}))
        else:
            plan.append(("CatBoost с нативными категориями", {"extends_best": True, "family": "catboost"}))
            worst = (diag.get("per_class") or [{}])[0]
            plan.append((f"самый слабый класс {worst.get('class')} (F1={worst.get('f1')}): мягкие веса классов поднимут recall редких классов",
                         {"extends_best": True, "class_weight": "sqrt_balanced"}))
            plan.append(("полные balanced-веса классов под Macro F1", {"extends_best": True, "class_weight": "balanced"}))
        if missing:
            plan.append((f"{len(missing)} признаков с пропусками >5%: явные индикаторы пропусков",
                         {"extends_best": True, "features": [{"op": "missing_flags", "cols": missing[:20]}]}))
        if cats:
            plan.append(("частотное кодирование категорий даёт деревьям порядок по популярности значения",
                         {"extends_best": True, "features": [{"op": "freq_encode", "cols": cats[:10]}]}))
        plan.append(("более медленное и глубокое обучение бустинга",
                     {"extends_best": True, "params": {"learning_rate": 0.03, "num_leaves": 127} if best.get("family", "lightgbm") == "lightgbm"
                      else {"learning_rate": 0.04, "depth": 8} if best.get("family") == "catboost" else {"learning_rate": 0.03}}))
        plan.append(("другое семейство бустинга для разнообразия", {"extends_best": True, "family": "xgboost"}))
        return plan

    def propose(self, state: dict) -> dict:
        out = []
        best_spec = (state.get("best") or {}).get("spec")
        for hypothesis, spec in self._plan(state):
            key = json.dumps(spec, sort_keys=True)
            try:
                resolved = json.dumps(normalize_spec(spec, self.task, best_spec), sort_keys=True)
            except ValueError:
                continue
            if key in self.done or resolved in state.get("seen_specs", ()):
                continue
            self.done.add(key)
            out.append({"hypothesis": hypothesis, "expected": "проверяется на валидации хоста", "spec": spec})
            if len(out) >= state["budget"]["max_experiments"]:
                break
        return {"analysis": "эвристический плейбук (без LLM)", "experiments": out, "stop": not out}
