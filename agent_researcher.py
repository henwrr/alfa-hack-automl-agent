"""Исследователь: предлагает следующие эксперименты.

LLMResearcher — основной: видит диагностику ошибок и историю, пишет гипотезы.
OfflineResearcher — эвристический плейбук: работает без LLM (разработка, сбой API)
и задаёт нижнюю планку, которую LLM должна обгонять.
"""

from __future__ import annotations

import copy
import difflib
import json
import os
import time

import numpy as np

from agent_features import OPS_DOC, TARGET_OPS_DOC
from agent_llm import LLM, extract_json
from agent_models import valid_params
from agent_schemas import ExperimentSpec, Proposal, ValidationError, format_validation_error, proposal_json_schema

def normalize_spec(spec: dict, task, best_spec: dict | None) -> dict:
    """Спецификация от LLM/плейбука -> проверенный словарь. Ошибки -> ValueError с понятным текстом.

    Проверку делает pydantic-схема ExperimentSpec (schemas.py); здесь — только слияние с лучшей
    моделью (extends_best / remove_features) и выбор полей под тип задачи.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec должен быть объектом")
    if spec.get("extends_best") and best_spec:
        base = copy.deepcopy(best_spec)
        base["params"] = {**base.get("params", {}), **(spec.get("params") or {})}
        removed = [json.dumps(op, sort_keys=True) for op in spec.get("remove_features") or [] if isinstance(op, dict)]
        kept = [op for op in base.get("features", []) if json.dumps(op, sort_keys=True) not in removed]
        base["features"] = kept + list(spec.get("features") or [])
        for key in ("family", "target_transform", "class_weight", "early_stopping_rounds", "offset"):
            if key in spec:
                base[key] = spec[key]
        if spec.get("family") and spec["family"] != best_spec.get("family"):
            base["params"] = dict(spec.get("params") or {})
        spec = base
    if spec.get("target_transform") == "gamma" and spec.get("family", "lightgbm") != "lightgbm":
        spec = {**spec, "target_transform": "log1p"}  # gamma есть только у LightGBM; ближайшая замена
    try:
        checked = ExperimentSpec.model_validate(spec)
    except ValidationError as exc:
        raise ValueError(f"spec не прошёл проверку: {format_validation_error(exc)}") from None
    unknown = [k for k in checked.params if k not in valid_params(checked.family)]
    if unknown:
        hints = {k: difflib.get_close_matches(k, sorted(valid_params(checked.family)), n=2) for k in unknown}
        raise ValueError(f"неизвестные параметры для {checked.family}: "
                         + "; ".join(f"{k} (возможно {', '.join(v)})" if v else k for k, v in hints.items()))
    out = {"family": checked.family, "params": checked.params,
           "features": [op.model_dump() for op in checked.features]}
    if task.task_type == "regression":
        out["target_transform"] = checked.target_transform
        if checked.offset != "none":
            out["offset"] = checked.offset
    else:
        out["class_weight"] = checked.class_weight
    if checked.early_stopping_rounds is not None:
        out["early_stopping_rounds"] = checked.early_stopping_rounds
    return out


SYSTEM_PROMPT = """Ты — ведущий ML-исследователь в автономном агенте для табличных задач.
Хост сам обучает модели, сам считает метрику на СВОЕЙ валидации и сам выбирает лучшую — ты не видишь меток валидации и не можешь «объявить» скор.
Твоя работа — цикл AutoResearch: посмотреть, ПОЧЕМУ текущая лучшая модель ошибается (диагностика ниже), сформулировать проверяемые гипотезы и описать эксперименты.

Как думать:
- Сначала прочитай host_findings: это факты, которые хост уже проверил (схема валидации, ключи сущностей, сдвиг).
- Начинай с диагностики: какие сегменты/классы дают основную ошибку, есть ли смещение (mean_residual), недопрогноз хвостов, путаница классов, переобучение (train_part_score vs valid_score), сдвиг train/test.
- Каждая гипотеза должна опираться на конкретный факт из диагностики или профиля и менять ОДНУ понятную вещь относительно лучшей модели (признаки ИЛИ модель ИЛИ таргет ИЛИ веса). Так результат интерпретируем.
- Не повторяй уже проверенные спецификации. Если гипотеза не подтвердилась — учти это, не долби одно направление.
- Признаки, сильно отличающиеся между train и test (adversarial check), могут вредить — проверяй их удаление.
- Не используй id и не пытайся восстановить таргет из test.

Формат спецификации эксперимента (spec):
{
  "family": "lightgbm" | "catboost" | "xgboost" | "extratrees" | "linear",
  "params": {гиперпараметры семейства; число деревьев подбирается early stopping, n_jobs/seed задаёт хост},
  "target_transform": "none" | "log1p" | "sqrt" | "gamma",  (только регрессия; gamma — LightGBM с objective=gamma: прогноз среднего, а не медианы, для положительного скошенного таргета; только при таргете > 0)
  "offset": "none" | "lag",  (только регрессия) — "lag": модель учит поправку к прошлому значению той же сущности (lag_* от target_lag, если пусто — te_* от target_encode); имеет смысл только вместе с target_lag/target_encode
  "class_weight": "none" | "balanced" | "sqrt_balanced",   (только классификация)
  "early_stopping_rounds": 20..1000 — терпение early stopping (по умолчанию 100; больше — для медленного learning_rate),
  "features": [операции признаков, применяются по порядку],
  "extends_best": true,  — взять спецификацию лучшей модели и ДОБАВИТЬ твои features / заменить указанные поля
  "remove_features": [операции лучшей модели, которые надо убрать]  (только вместе с extends_best)
}
Не повторяй в features операции, которые уже есть в лучшей модели — они и так сохраняются.
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
Пиши analysis и hypothesis по-русски.""" % "\n".join(f"- {v}" for v in {**OPS_DOC, **TARGET_OPS_DOC}.values())


def _compact_profile(profile: dict, limit: int = 80) -> dict:
    """Профиль в одну строку на признак — промпт должен влезать в контекст небольшой модели."""
    lines = []
    for name, info in list(profile.get("features", {}).items())[:limit]:
        parts = [info["dtype"], f"nuniq={info['nunique']}"]
        if info.get("missing_train") or info.get("missing_test"):
            parts.append(f"miss={info['missing_train']:.2f}/{info['missing_test']:.2f}")
        if info.get("unseen_in_test_share"):
            parts.append(f"unseen_in_test={info['unseen_in_test_share']:.2f}")
        if info.get("min") is not None:
            parts.append(f"range=[{info['min']:.4g}, {info['max']:.4g}]")
        if info.get("zeros_share", 0) > 0.05:
            parts.append(f"zeros={info['zeros_share']:.2f}")
        if info.get("description"):
            parts.append(info["description"][:70])
        lines.append(f"{name}: " + "; ".join(parts))
    rest = {k: v for k, v in profile.items() if k != "features"}
    return {**rest, "features (dtype; nunique; missing train/test; ...)": lines}


def _compact_history(history: list[dict], maximize: bool, best_n: int = 6, recent_n: int = 10) -> list[dict]:
    """В промпт — лучшие и последние опыты, а не всё подряд: за час набирается 200+ опытов."""
    ok = [h for h in history if h.get("status") == "ok" and h.get("score") is not None]
    top = sorted(ok, key=lambda h: h["score"], reverse=maximize)[:best_n]
    recent = history[-recent_n:]
    keep = {id(h) for h in top + recent}
    out = []
    for h in [h for h in history if id(h) in keep]:
        item = {k: h.get(k) for k in ("id", "hypothesis", "status", "score", "delta_vs_best", "verdict") if h.get(k) is not None}
        if h.get("spec"):
            item["spec"] = h["spec"]
        if h.get("error"):
            item["error"] = h["error"][:300]
        out.append(item)
    return out


class LLMResearcher:
    name = "llm"

    def __init__(self, llm: LLM, log, run_dir=None):
        self.llm = llm
        self.log = log
        self.run_dir = run_dir

    def _chat(self, messages):
        started = time.time()
        text = self.llm.chat(messages, schema=proposal_json_schema())
        if self.run_dir is not None:
            record = {"seconds": round(time.time() - started, 1), "prompt_chars": sum(len(m["content"]) for m in messages),
                      "prompt_tokens": self.llm.last_prompt_tokens, "request": messages[-1]["content"], "response": text}
            with open(self.run_dir / "llm_calls.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.log(f"LLM ответила за {time.time() - started:.0f}s (prompt ~{self.llm.last_prompt_tokens} токенов; "
                 f"потрачено {self.llm.cost:.2f} ₽ из {self.llm.max_cost:.0f})")
        return text

    @staticmethod
    def _fit_prompt(payload: dict, state: dict) -> str:
        """Промпт должен влезать в контекст (у Ollama 16k токенов, ~0.33 токена на символ).
        Урезаем по очереди: историю, детали диагностики, описание задачи, профиль."""
        limit = int(os.getenv("AGENT_PROMPT_CHARS", 36000)) - len(SYSTEM_PROMPT)
        dump = lambda: json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))  # noqa: E731
        maximize = state["task"]["maximize"]
        steps = [
            lambda: payload.update(history=_compact_history(state["history"], maximize, best_n=4, recent_n=6)),
            lambda: payload["best"] and payload["best"].get("diagnostics", {}).pop("largest_errors", None),
            lambda: payload.update(task_description=state["description"][:1500]),
            lambda: payload["best"] and payload["best"].get("diagnostics", {}).pop("error_by_target_decile", None),
            lambda: payload.update(history=_compact_history(state["history"], maximize, best_n=3, recent_n=3)),
            lambda: payload.update(data_profile=_compact_profile(state["profile"], limit=25)),
        ]
        text = dump()
        for step in steps:
            if len(text) <= limit:
                break
            step()
            text = dump()
        return text

    def propose(self, state: dict) -> dict:
        payload = {
            "host_findings": state.get("host_findings", []),
            "task": state["task"], "task_description": state["description"][:4000],
            "data_profile": _compact_profile(state["profile"]), "validation": state["validation"],
            "train_test_shift": state["shift"],
            "history": _compact_history(state["history"], state["task"]["maximize"]),
            "history_note": f"показаны лучшие и последние опыты из {len(state['history'])}",
            "best": copy.deepcopy(state["best"]), "budget": state["budget"],
            "previous_errors": state.get("errors", [])[-3:],
        }
        content = self._fit_prompt(payload, state)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        text = self._chat(messages)
        try:
            answer = Proposal.model_validate(extract_json(text))
        except (ValueError, ValidationError) as exc:
            detail = format_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
            messages += [{"role": "assistant", "content": text[:4000]},
                         {"role": "user", "content": f"Ответ не прошёл проверку ({detail}). Верни ТОЛЬКО JSON в описанном формате."}]
            answer = Proposal.model_validate(extract_json(self._chat(messages)))
        return {"analysis": answer.analysis[:2000],
                "experiments": [e.model_dump() for e in answer.experiments],
                "stop": answer.stop}


class OfflineResearcher:
    """Детерминированный плейбук: следующий шаг выбирается по профилю и диагностике."""

    name = "offline"

    def __init__(self, task, profile):
        self.task = task
        self.profile = profile
        self.done: set[str] = set()

    def _plan(self, state) -> list[tuple[str, dict, bool]]:
        """(гипотеза, spec, strong). strong — «сильные ходы» с большим ожидаемым эффектом: хост делает их до LLM."""
        task, prof = self.task, self.profile
        best = (state.get("best") or {}).get("spec") or {}
        diag = (state.get("best") or {}).get("diagnostics") or {}
        missing = [c for c, i in prof["features"].items() if i.get("missing_train", 0) > 0.05]
        cats = [c for c in task.cat_cols if prof["features"].get(c, {}).get("nunique", 0) > 2]
        plan: list[tuple[str, dict, bool]] = []  # бейзлайн запускает сам хост

        def add(hypothesis, spec, strong=False):
            plan.append((hypothesis, spec, strong))

        for key in prof.get("entity_keys", []):
            add(f"ключ сущности {key['cols']} покрывает {key['test_rows_with_known_key']:.0%} test: "
                "средний таргет той же сущности в train (out-of-fold)",
                {"extends_best": True, "features": [{"op": "target_encode", "cols": key["cols"]}]}, strong=True)
            if state.get("validation", {}).get("kind") == "time":
                add(f"сущность {key['cols']} повторяется по периодам: её таргет в прошлом периоде (лаг)",
                    {"extends_best": True, "features": [{"op": "target_lag", "cols": key["cols"]}]}, strong=True)
        if task.task_type == "regression":
            skew = prof["target"].get("skew", 0)
            if skew > 1:
                add(f"skew таргета {skew}: log1p должен уменьшить влияние хвоста на обучение",
                    {"extends_best": True, "target_transform": "log1p"}, strong=True)
                if prof["target"].get("min", 0) > 0 and task.metric in {"rmse", "mse", "r2"}:
                    # log1p оценивает ближе к медиане, а RMSE минимизирует среднее: на скошенном таргете хвост
                    # занижается. Gamma (лог-связь) оценивает среднее. Помогает не везде — решает валидация.
                    add(f"skew таргета {skew} и метрика {task.metric}: objective=gamma прогнозирует среднее, "
                        "а не медиану, и меньше занижает большие значения",
                        {"extends_best": True, "family": "lightgbm", "target_transform": "gamma"}, strong=True)
            if prof.get("entity_keys") and state.get("validation", {}).get("kind") == "time":
                # rakovskiy_branch: на 6 панелях «сущность × период» −1…−22% RMSE на четырёх, хуже на двух —
                # гипотеза для валидации. Ставится после выбора преобразования таргета (offset в его шкале).
                add("прошлое значение сущности как стартовая точка: модель учит поправку к нему, а не таргет с нуля",
                    {"extends_best": True, "offset": "lag"}, strong=True)
            add("CatBoost с нативными категориями может лучше использовать категориальные признаки",
                {"extends_best": True, "family": "catboost"}, strong=True)
        else:
            imbalance = prof["target"].get("imbalance_ratio", 1)
            worst = (diag.get("per_class") or [{}])[0]
            if imbalance >= 3 and task.metric in {"macro_f1", "f1", "weighted_f1"}:
                add(f"дисбаланс классов x{imbalance}, слабейший класс {worst.get('class')} (F1={worst.get('f1')}): "
                    "мягкие веса классов поднимут recall редких классов",
                    {"extends_best": True, "class_weight": "sqrt_balanced"}, strong=True)
            add("CatBoost с нативными категориями", {"extends_best": True, "family": "catboost"})
            if imbalance >= 3:
                add("полные balanced-веса классов", {"extends_best": True, "class_weight": "balanced"})
        # Дешёвая проверка: если зависимость близка к линейной, бустинги ей проигрывают
        # (синтетика из сравнения агентов в main: Ridge 20.4 RMSE против 38.8 у нашего плейбука без linear).
        add("линейная модель (Ridge / логистическая регрессия): проверка, не линейна ли зависимость",
            {"extends_best": True, "family": "linear"})
        if best.get("family") == "linear":
            for value in (0.1, 10.0):
                key = "alpha" if task.task_type == "regression" else "C"
                add(f"линейная модель лидирует: регуляризация {key}={value}",
                    {"extends_best": True, "family": "linear", "params": {key: value}})
        if missing:
            add(f"{len(missing)} признаков с пропусками >5%: явные индикаторы пропусков",
                {"extends_best": True, "features": [{"op": "missing_flags", "cols": missing[:20]}]})
        if cats:
            add("частотное кодирование категорий даёт деревьям порядок по популярности значения",
                {"extends_best": True, "features": [{"op": "freq_encode", "cols": cats[:10]}]})
        shift = state.get("shift") or {}
        shifted = [d["feature"] for d in shift.get("top_shifted_features", [])[:5] if d["importance_share"] > 0.05]
        if shift.get("auc", 0) > 0.8 and shifted:
            add(f"сильный сдвиг train/test (AUC {shift['auc']}): убрать самые сдвинутые признаки {shifted}",
                {"extends_best": True, "features": [{"op": "drop", "cols": shifted}]})
        family = best.get("family", "lightgbm")
        slow = ({"learning_rate": 0.03, "num_leaves": 127} if family == "lightgbm" else
                {"learning_rate": 0.04, "depth": 8} if family == "catboost" else {"learning_rate": 0.03})
        add("более медленное и глубокое обучение бустинга", {"extends_best": True, "params": slow})
        add("другое семейство бустинга для разнообразия", {"extends_best": True, "family": "xgboost"})
        if family != "lightgbm":
            add("LightGBM как альтернативное семейство (разнообразие для ансамбля)", {"extends_best": True, "family": "lightgbm"})
        return plan

    SPACE = {
        "lightgbm": {"num_leaves": [15, 31, 63, 127, 255], "min_child_samples": [5, 10, 20, 50, 100],
                     "colsample_bytree": [0.4, 0.6, 0.8, 1.0], "subsample": [0.6, 0.8, 1.0],
                     "reg_lambda": [0.0, 1.0, 5.0, 20.0], "learning_rate": [0.02, 0.03, 0.05, 0.08]},
        "catboost": {"depth": [4, 6, 8, 10], "l2_leaf_reg": [1, 3, 10, 30], "learning_rate": [0.03, 0.05, 0.08],
                     "random_strength": [0.5, 1, 3]},
        "xgboost": {"max_depth": [4, 6, 8, 10], "min_child_weight": [1, 5, 20], "colsample_bytree": [0.5, 0.7, 0.9],
                    "subsample": [0.6, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 10.0], "learning_rate": [0.02, 0.04, 0.08]},
    }

    def _tuning_moves(self, state, n: int = 4) -> list[tuple[str, dict, bool]]:
        """Когда идеи плейбука кончились: случайный поиск вокруг лучшей модели и в других семействах.

        Полезно и само по себе, и как источник разнообразных сильных моделей для финального ансамбля.
        """
        best = (state.get("best") or {}).get("spec") or {}
        rng = np.random.default_rng(len(state.get("seen_specs", ())) + 7)
        moves = []
        for i in range(n):
            family = best.get("family", "lightgbm") if i % 2 == 0 else str(rng.choice(list(self.SPACE)))
            if family not in self.SPACE:
                family = "lightgbm"
            space = self.SPACE[family]
            keys = rng.choice(list(space), size=min(3, len(space)), replace=False)
            params = {str(k): (v.item() if hasattr(v, "item") else v) for k in keys for v in [rng.choice(space[k])]}
            spec = {"extends_best": True, "family": family, "params": params}
            if family == best.get("family"):
                spec["params"] = {**best.get("params", {}), **params}
            moves.append((f"тюнинг {family}: {params}", spec, False))
        return moves

    def propose(self, state: dict, strong_only: bool = False) -> dict:
        out = []
        best_spec = (state.get("best") or {}).get("spec")
        plan = self._plan(state)
        if not strong_only:
            plan = plan + self._tuning_moves(state, n=8)
        for hypothesis, spec, strong in plan:
            if strong_only and not strong:
                continue
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
        return {"analysis": "сильные ходы хоста" if strong_only else "эвристический плейбук (без LLM)",
                "experiments": out, "stop": not out}
