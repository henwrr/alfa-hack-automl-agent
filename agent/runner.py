"""Главный цикл агента: понять задачу -> бейзлайн -> [диагностика -> гипотезы -> эксперименты] -> финальная модель."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostics import diagnose
from .features import apply_ops, apply_target_ops, find_entity_keys, split_target_ops
from .llm import LLM, LLMError
from .metrics import better, score
from .models import Model
from .researcher import LLMResearcher, OfflineResearcher, normalize_spec
from .task import discover, profile
from .validation import adversarial_check, choose_split


class Agent:
    def __init__(self, task_path, time_budget: float = 600, seed: int = 42, n_jobs: int | None = None,
                 offline: bool = False, run_dir: str | None = None, experiments_per_round: int = 2, n_folds: int = 2):
        self.t0 = time.time()
        self.task_path = Path(task_path).resolve()
        self.time_budget = float(time_budget)
        self.seed = seed
        self.n_jobs = n_jobs or min(8, os.cpu_count() or 4)
        self.offline = offline
        self.per_round = experiments_per_round
        self.n_folds = n_folds
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir) if run_dir else self.task_path.parent / "agent_runs" / stamp
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.run_dir = Path(tempfile.mkdtemp(prefix="agent_run_"))
        self.history: list[dict] = []
        self.errors: list[str] = []
        self.best: dict | None = None
        self.seen_specs: set[str] = set()
        self.llm_seconds: list[float] = []
        self.fold_preds: dict[str, list] = {}  # exp_id -> предсказания на валидации каждого фолда (для ансамбля)
        self.submission_path = self.task_path.parent / "submission.csv"

    # ---------- служебное ----------
    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    @property
    def remaining(self) -> float:
        return self.time_budget - self.elapsed

    def log(self, message: str) -> None:
        line = f"[{self.elapsed:7.1f}s] {message}"
        print(line, file=sys.stderr, flush=True)
        with open(self.run_dir / "agent.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def journal(self, record: dict) -> None:
        with open(self.run_dir / "experiments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # ---------- подготовка ----------
    def setup(self) -> None:
        self.task, self.train, self.test = discover(self.task_path)
        t = self.task
        self.log(f"Задача: {t.task_type}, target={t.target}, метрика={t.metric} "
                 f"({'max' if t.maximize else 'min'}), признаков={len(t.feature_cols)}, категориальных={len(t.cat_cols)}")
        self.y = self.train[t.target].reset_index(drop=True)
        try:  # запасной ответ до любых моделей: если все обучения упадут, файл всё равно будет
            if t.task_type == "regression":
                prior = np.full(len(self.test), float(pd.to_numeric(self.y, errors="coerce").mean()))
            else:
                share = self.y.astype(str).value_counts(normalize=True)
                prior = np.tile([share.get(str(c), 0.0) for c in t.classes], (len(self.test), 1))
            self.write_submission(prior, note="prior (запасной, до моделей)")
        except Exception as exc:
            self.log(f"Запасной submission не записан: {exc}")
        self.profile = profile(t, self.train, self.test)
        self.split = choose_split(t, self.train, self.test, seed=self.seed, n_folds=self.n_folds)
        self.log(f"Валидация: {self.split['note']}")
        self.y = self.train[t.target].reset_index(drop=True)
        self.all_X = pd.concat([self.train[t.feature_cols], self.test[t.feature_cols]], ignore_index=True)
        self.n_train = len(self.train)
        col = self.split["column"]
        self.time_values = (pd.to_numeric(self.all_X[col], errors="coerce").to_numpy() if col else None)
        try:
            exclude = [self.split["column"]] if self.split["column"] else []
            self.shift = adversarial_check(t, self.train, self.test, seed=self.seed, n_jobs=self.n_jobs, exclude=exclude)
            if self.split["column"]:
                self.shift["note"] = f"'{self.split['column']}' исключён: test целиком в другом периоде по нему"
            self.log(f"Сдвиг train/test: AUC={self.shift.get('auc')} — {self.shift.get('verdict')}; "
                     f"топ: {[d['feature'] for d in self.shift.get('top_shifted_features', [])[:5]]}")
        except Exception as exc:  # диагностика не должна ронять агента
            self.shift = {"error": str(exc)[:300]}
        try:
            exclude = [self.split["column"]] if self.split["column"] else []
            self.entity_keys, self.te_candidates = find_entity_keys(self.train, self.test, t.feature_cols, t.target,
                                                                    t.task_type, exclude=exclude)
            for k in self.entity_keys:
                self.log(f"Найден ключ сущности {k['cols']}: {k['test_rows_with_known_key']:.0%} строк test встречаются в train, "
                         f"однородность таргета внутри ключа {k['target_purity_within_key']}")
        except Exception as exc:
            self.entity_keys, self.te_candidates = [], []
            self.log(f"Поиск ключа сущности не удался: {exc}")
        self.profile["entity_keys"] = self.entity_keys
        llm = LLM()
        if not self.offline and llm.available:
            self.researcher = LLMResearcher(llm, self.log, self.run_dir)
            self.llm = llm
            self.log(f"Исследователь: LLM {llm.model}")
        else:
            self.llm = None
            self.researcher = OfflineResearcher(t, self.profile)
            self.log("Исследователь: offline-плейбук (LLM не задана или --offline)")
        self.fallback = self.researcher if self.llm is None else OfflineResearcher(t, self.profile)
        (self.run_dir / "context.json").write_text(json.dumps(
            {"task": t.summary(), "validation": {k: v for k, v in self.split.items() if not k.endswith("_idx")},
             "shift": self.shift, "profile": self.profile}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---------- эксперимент ----------
    def _plain_features(self, spec):
        plain, target_ops = split_target_ops(spec.get("features"))
        # drop — в самом конце: иначе удалённый столбец может понадобиться ключу target_encode.
        drops = [op for op in plain if op.get("op") == "drop"]
        plain = [op for op in plain if op.get("op") != "drop"]
        feats, cats = apply_ops(self.all_X, self.task.cat_cols, plain)
        self._pending_drops = sorted({c for op in drops for c in op.get("cols", [])})
        cats = [c for c in cats if c not in self._pending_drops]
        return feats, cats, target_ops

    def _with_target_ops(self, feats, target_ops, train_rows):
        """train_rows — строки train, чьи таргеты можно использовать (fit-часть фолда / весь train в финале)."""
        if target_ops:
            classes = self.task.classes if self.task.task_type == "classification" else None
            feats = apply_target_ops(feats, target_ops, self.y.to_numpy(), self.n_train, np.asarray(train_rows), classes,
                                     self.seed, time_values=self.time_values)
        drops = [c for c in getattr(self, "_pending_drops", []) if c in feats.columns]
        if drops:
            feats = feats.drop(columns=drops)
        return feats.iloc[:self.n_train].reset_index(drop=True), feats.iloc[self.n_train:].reset_index(drop=True)

    def _avg_experiment_seconds(self) -> float:
        times = [h["seconds"] for h in self.history if h["status"] == "ok"]
        return float(np.median(times)) if times else 60.0

    def _refit_estimate(self) -> float:
        if not self.best:
            return 30.0
        ratio = self.n_train / max(len(self.split["fit_idx"]), 1)
        single = self.best["seconds"] / len(self.split["folds"]) * ratio * 1.3
        # финал: несколько seed и (в длинном запуске) несколько моделей ансамбля
        seeds = max(1, int(os.getenv("AGENT_FINAL_SEEDS", 3)))
        members = 3 if self.time_budget >= 1200 and len(self.fold_preds) >= 3 else 1
        # на коротком бюджете не отдаём финалу больше трети времени: лишние seed просто не успеют
        return min(single * seeds * members, max(single, self.time_budget / 3)) + 15

    def run_experiment(self, hypothesis: str, raw_spec: dict, source: str, base_spec: dict | None = None) -> dict:
        """base_spec — от чего считать extends_best (по умолчанию текущая лучшая модель)."""
        exp_id = f"exp_{len(self.history) + 1:03d}"
        record = {"id": exp_id, "source": source, "hypothesis": hypothesis, "status": "error"}
        started = time.time()
        try:
            base = base_spec if base_spec is not None else (self.best["spec"] if self.best else None)
            spec = normalize_spec(raw_spec, self.task, base)
            record["spec"] = spec
            key = json.dumps(spec, sort_keys=True)
            if key in self.seen_specs:
                raise ValueError("такая спецификация уже проверялась")
            self.seen_specs.add(key)
            feats, cats, target_ops = self._plain_features(spec)
            folds = self.split["folds"]
            limit = max(30.0, min(self.remaining - self._refit_estimate() - 10, max(90.0, 0.35 * self.time_budget)))
            fold_scores, iterations, first, preds = [], [], None, []
            for i, (fit_idx, valid_idx) in enumerate(folds):
                X, X_test = self._with_target_ops(feats, target_ops, fit_idx)
                model = Model(spec, self.task, seed=self.seed, n_jobs=self.n_jobs)
                model.fit(X.iloc[fit_idx], self.y.iloc[fit_idx], cats, seconds=limit / len(folds))
                pred = model.predict(X.iloc[valid_idx])
                value = score(self.task, self.y.iloc[valid_idx], pred)
                if not np.isfinite(value):
                    raise ValueError("метрика не конечна")
                fold_scores.append(value)
                preds.append(np.asarray(pred, dtype=np.float32))
                if model.best_iteration:
                    iterations.append(model.best_iteration)
                if i == 0:
                    first = (model, X, X_test, fit_idx, valid_idx, pred)
            value = float(np.mean(fold_scores))
            previous = self.best["score"] if self.best else None
            improved = better(value, previous, self.task.maximize)
            best_iteration = int(np.mean(iterations)) if iterations else None
            record.update(score=round(value, 6), fold_scores=[round(v, 6) for v in fold_scores],
                          best_iteration=best_iteration,
                          delta_vs_best=None if previous is None else round(value - previous, 6),
                          verdict="улучшение — новая лучшая модель" if improved else "не лучше текущей лучшей")
            record["seconds"] = round(time.time() - started, 1)
            if improved:
                model, X, X_test, fit_idx, valid_idx, pred = first
                diag = diagnose(self.task, model, X.iloc[fit_idx], self.y.iloc[fit_idx], X.iloc[valid_idx],
                                self.y.iloc[valid_idx], pred, self.train.iloc[valid_idx][self.task.feature_cols])
                self.best = {"id": exp_id, "score": value, "fold_scores": fold_scores, "spec": spec,
                             "seconds": record["seconds"], "best_iteration": best_iteration, "diagnostics": diag}
                self.write_submission(model.predict(X_test), note=f"safety ({exp_id}, обучена на fit-части)")
            record["status"] = "ok"
            self.fold_preds[exp_id] = preds
            if os.getenv("AGENT_SAVE_PREDS"):
                (self.run_dir / "fold_preds").mkdir(exist_ok=True)
                np.savez_compressed(self.run_dir / "fold_preds" / f"{exp_id}.npz", *preds)
        except Exception as exc:
            record.pop("verdict", None)
            record["error"] = f"{type(exc).__name__}: {exc}"[:1500]
            record["seconds"] = round(time.time() - started, 1)
            if not isinstance(exc, (ValueError, KeyError)):
                record["traceback"] = traceback.format_exc()[-1500:]
        self.history.append(record)
        self.journal(record)
        mark = "★" if record.get("verdict", "").startswith("улучшение") else " "
        self.log(f"{mark} {exp_id} [{source}] {record['status']} "
                 f"{self.task.metric}={record.get('score')} {record.get('fold_scores', '')} ({record['seconds']}s) — {hypothesis[:140]}"
                 + (f" | ошибка: {record['error'][:200]}" if record["status"] != "ok" else ""))
        return record

    # ---------- исследование ----------
    def _state(self, max_experiments: int) -> dict:
        history = [{k: h.get(k) for k in ("id", "source", "hypothesis", "spec", "status", "score", "fold_scores", "delta_vs_best",
                                          "verdict", "seconds", "best_iteration", "error")} for h in self.history]
        best = None
        if self.best:
            best = {k: self.best[k] for k in ("id", "score", "fold_scores", "spec", "best_iteration", "diagnostics")}
        return {"task": self.task.summary(), "description": self.task.description, "profile": self.profile,
                "host_findings": self._findings(),
                "validation": {"kind": self.split["kind"], "note": self.split["note"], "folds": len(self.split["folds"]),
                               "fit_rows": len(self.split["fit_idx"]), "valid_rows": len(self.split["valid_idx"]),
                               "how_to_read": "score — среднее по фолдам; fold_scores показывают шум: разница между "
                                              "экспериментами меньше разброса фолдов может быть случайной"},
                "shift": self.shift, "history": history, "best": best, "errors": self.errors,
                "seen_specs": self.seen_specs,
                "budget": {"remaining_seconds": round(self.remaining), "typical_experiment_seconds": round(self._avg_experiment_seconds()),
                           "max_experiments": max_experiments}}

    def _strong_phase(self, first: dict) -> None:
        """Фаза 1: сильные ходы (ключ сущности, лаг, log-таргет, веса классов, смена семейства).

        Проверяются цепочкой на быстрой модели (LightGBM), независимо от того, с какого семейства начат запуск:
        жадный цикл со старта CatBoost застревал — поверх CatBoost+TE лаг и log-таргет не принимались,
        хотя на LightGBM они давали −2.5 и −1.1 RMSE. Найденный набор признаков переносится на лучшее семейство.
        """
        fast = "lightgbm"
        chain = first if first.get("status") == "ok" and first["spec"]["family"] == fast else \
            self.run_experiment(f"быстрая модель для отбора признаков: {fast}", {"family": fast}, "host")
        if chain.get("status") != "ok":
            chain = first
        if chain.get("status") != "ok":
            return
        chain_spec, chain_score = chain["spec"], chain["score"]
        while self._can_start_experiment():
            state = self._state(1)
            state["best"] = {**(state["best"] or {}), "spec": chain_spec}
            proposal = self.fallback.propose(state, strong_only=True)
            if not proposal["experiments"]:
                break
            exp = proposal["experiments"][0]
            record = self.run_experiment(exp["hypothesis"], exp["spec"], "host-strong", base_spec=chain_spec)
            if record.get("status") == "ok" and better(record["score"], chain_score, self.task.maximize):
                chain_spec, chain_score = record["spec"], record["score"]
        # Перенос: признаки цепочки — на семейство лучшей модели, если оно другое (и наоборот).
        if self.best and self._can_start_experiment():
            best_spec = self.best["spec"]
            for target in {best_spec["family"], first.get("spec", {}).get("family")} - {chain_spec["family"], None}:
                spec = {**chain_spec, "family": target, "params": best_spec["params"] if target == best_spec["family"] else {}}
                try:
                    if json.dumps(normalize_spec(spec, self.task, None), sort_keys=True) in self.seen_specs:
                        continue
                except ValueError:
                    continue
                self.run_experiment(f"перенос признаков быстрой цепочки на {target}", spec, "host-strong")

    def baseline_family(self) -> str:
        """Бейзлайн — LightGBM + early stopping: быстрый «движок» для отбора признаков.

        Проверено: сам по себе CatBoost-бейзлайн лучше (регрессия 46.2 vs 49.1 RMSE, бинарная 0.825 vs 0.820 AUC),
        но жадный цикл с него застревает: поверх CatBoost+TE лаг и log-таргет уже не дают прироста, опыты в
        2–4 раза медленнее, и за 600 с итог хуже (35.38 vs 34.05). CatBoost пробуется сильным ходом сразу после
        признаков. В мультиклассе CatBoost хуже и медленнее в 34 раза (0.560 vs 0.601, 888 с vs 26 с).
        """
        return os.getenv("AGENT_BASELINE", "lightgbm")

    def _playbook_move(self, reason: str) -> bool:
        proposal = self.fallback.propose(self._state(1))
        if proposal["experiments"] and self._can_start_experiment():
            exp = proposal["experiments"][0]
            self.run_experiment(exp["hypothesis"], exp["spec"], "offline")
            return True
        return False

    def _ask(self, researcher, state) -> dict:
        """Спрашиваем исследователя. Если это LLM — пока она думает, хост сам гоняет ходы плейбука/тюнинга:
        ответ LLM идёт десятки секунд–минуты, эксперимент — секунды, процессор не должен простаивать."""
        if researcher is self.fallback:
            return researcher.propose(state)
        started = time.time()
        # daemon-нить: если LLM не успеет к концу бюджета, она не задержит выход процесса
        future: Future = Future()

        def call():
            try:
                future.set_result(researcher.propose(state))
            except BaseException as exc:  # noqa: BLE001 — передаём в основной поток
                future.set_exception(exc)

        threading.Thread(target=call, daemon=True).start()
        busy = 0
        while not future.done() and self._can_start_experiment():
            if not self._playbook_move("пока LLM думает"):
                break
            busy += 1
        wait = max(1.0, self.remaining - self._refit_estimate() - 10)
        try:
            proposal = future.result(timeout=wait)
        except FuturesTimeout:
            raise LLMError(f"LLM не успела ответить до резерва времени ({wait:.0f}s)") from None
        self.llm_seconds.append(time.time() - started)
        if busy:
            self.log(f"Пока LLM думала ({time.time() - started:.0f}s), хост провёл {busy} эксп.")
        return proposal

    def _findings(self) -> list[str]:
        """Главные факты, которые хост нашёл сам, — коротко и на первом месте в промпте."""
        out = [self.split["note"]]
        for k in self.entity_keys:
            out.append(f"Ключ сущности {k['cols']}: {k['test_rows_with_known_key']:.0%} строк test имеют такой же ключ в train, "
                       f"таргет внутри ключа однороден (purity {k['target_purity_within_key']}). "
                       "Операции target_encode / target_lag по этому ключу обычно дают самый большой выигрыш.")
        if self.te_candidates:
            out.append("Кандидаты для target_encode (столбец: однородность таргета внутри значения, покрытие test): "
                       + ", ".join(f"{d['col']} ({d['purity']}, {d['test_coverage']:.0%})" for d in self.te_candidates))
        if self.shift.get("auc"):
            top = [d["feature"] for d in self.shift.get("top_shifted_features", [])[:5]]
            out.append(f"Adversarial AUC train vs test = {self.shift['auc']} ({self.shift.get('verdict')}); сильнее всего сдвинуты {top}.")
        if self.best:
            applied = [op.get("op") + str(op.get("cols", ""))[:60] for op in self.best["spec"].get("features", [])]
            out.append(f"В лучшей модели {self.best['id']} уже применено: family={self.best['spec']['family']}, "
                       f"features={applied}; extends_best=true сохраняет всё это.")
        return out

    def _can_start_experiment(self) -> bool:
        return self.remaining > self._refit_estimate() + self._avg_experiment_seconds() * 1.1 + 10

    def research(self) -> None:
        family = self.baseline_family()
        first = self.run_experiment(f"бейзлайн: {family} с дефолтами и early stopping", {"family": family}, "host")
        for other in ("catboost", "extratrees", "linear", "xgboost"):
            if first.get("status") == "ok" or not self._can_start_experiment():
                break
            if other != family:  # бейзлайн упал на этих данных — другое семейство может справиться
                first = self.run_experiment(f"бейзлайн {family} упал: пробуем {other}", {"family": other}, "host")
        self._strong_phase(first)
        llm_failures = 0
        rounds = 0
        while self._can_start_experiment():
            rounds += 1
            fit = int((self.remaining - self._refit_estimate()) // max(self._avg_experiment_seconds(), 1))
            per_round = self.per_round
            if self.llm is not None and self.llm_seconds:
                # Ответ LLM дорогой, эксперимент дешёвый — проверяем за раунд больше гипотез.
                per_round = int(np.clip(np.median(self.llm_seconds) / max(self._avg_experiment_seconds(), 1), self.per_round, 4))
            state = self._state(max(1, min(per_round, fit)))
            researcher = self.researcher
            try:
                proposal = self._ask(researcher, state)
            except (LLMError, ValueError) as exc:
                llm_failures += 1
                self.errors.append(f"researcher: {exc}")
                self.log(f"Исследователь {researcher.name} не ответил ({exc}); раунд делает offline-плейбук")
                if llm_failures >= 3:
                    self.researcher = self.fallback
                proposal = self.fallback.propose(state)
                researcher = self.fallback
            self.journal({"round": rounds, "researcher": researcher.name, "analysis": proposal["analysis"],
                          "experiments": proposal["experiments"]})
            if proposal["analysis"]:
                self.log(f"Анализ ({researcher.name}): {proposal['analysis'][:400]}")
            if not proposal["experiments"] and researcher is not self.fallback:
                # LLM остановилась или ничего не предложила — время ещё есть, добираем плейбуком.
                self.log("LLM не предложила экспериментов; дальше работает offline-плейбук")
                if proposal["stop"]:
                    self.researcher = self.fallback
                proposal = self.fallback.propose(state)
                researcher = self.fallback
            if not proposal["experiments"] and self.researcher is not self.fallback:
                # Плейбук пуст, но LLM ещё в строю (сбой был разовым) — спрашиваем её снова в следующем раунде.
                continue
            if not proposal["experiments"]:
                self.log("Идеи закончились — переходим к финальной модели")
                break
            for exp in proposal["experiments"][:state["budget"]["max_experiments"]]:
                if not self._can_start_experiment():
                    break
                record = self.run_experiment(str(exp.get("hypothesis", ""))[:500], exp.get("spec") or {}, researcher.name)
                if "уже проверялась" in record.get("error", "") and researcher is not self.fallback:
                    self._playbook_move("повтор от LLM заменён ходом плейбука")

    # ---------- финал ----------
    # ---------- финал ----------
    def _blend_scores(self, members: list[str], folds=None) -> list[float]:
        out = []
        for i, (_, valid_idx) in enumerate(self.split["folds"]):
            if folds is not None and i not in folds:
                continue
            pred = np.mean([self.fold_preds[m][i] for m in members], axis=0)
            out.append(score(self.task, self.y.iloc[valid_idx], pred))
        return out

    def _greedy(self, candidates: list[str], folds, max_steps: int) -> list[str]:
        """Жадный отбор с повторами (Caruana) по среднему скору на заданных фолдах."""
        maximize = self.task.maximize
        mean = lambda ms: float(np.mean(self._blend_scores(ms, folds)))  # noqa: E731
        chosen = [(max if maximize else min)(candidates, key=lambda c: mean([c]))]
        current = mean(chosen)
        for _ in range(max_steps):
            value, cid = (max if maximize else min)((mean(chosen + [c]), c) for c in candidates)
            if not better(value, current, maximize):
                break
            chosen.append(cid)
            current = value
        return chosen

    def select_ensemble(self, max_candidates: int = 10, max_steps: int = 15) -> list[str]:
        """Ансамбль принимается, если он честно обобщается: отбор на одном фолде, проверка на другом.

        Для каждого фолда A: жадный отбор только по A и лучшая одиночная модель по A; обе оцениваются на
        остальных фолдах. Если в среднем ансамбль выигрывает на НЕвиденных фолдах — берём ансамбль,
        отобранный по всем фолдам. Иначе выигрыш — подгонка под валидацию, берём одну модель.
        """
        maximize = self.task.maximize
        ok = [h for h in self.history if h["status"] == "ok" and h["id"] in self.fold_preds]
        ok.sort(key=lambda h: h["score"], reverse=maximize)
        candidates = [h["id"] for h in ok[:max_candidates]]
        n_folds = len(self.split["folds"])
        if len(candidates) < 2:
            return [self.best["id"]]
        gains = []
        if n_folds >= 2:
            for a in range(n_folds):
                others = [i for i in range(n_folds) if i != a]
                members = self._greedy(candidates, [a], max_steps)
                single = self._greedy(candidates, [a], 0)
                blend_score = float(np.mean(self._blend_scores(members, others)))
                single_score = float(np.mean(self._blend_scores(single, others)))
                gains.append((blend_score - single_score) * (1 if maximize else -1))
        chosen = self._greedy(candidates, None, max_steps)
        single = self._blend_scores([self.best["id"]])
        blend = self._blend_scores(chosen)
        if len(set(chosen)) < 2 or (gains and np.mean(gains) <= 0):
            self.log(f"Ансамбль не обобщается: выигрыш на невиденных фолдах {[round(g, 5) for g in gains]} — берём одну модель")
            return [self.best["id"]]
        self.ensemble_scores = {"single": single, "ensemble": blend, "cross_fold_gains": gains}
        self.log(f"Ансамбль: {dict(pd.Series(chosen).value_counts())} — {self.task.metric} "
                 f"{np.mean(single):.5f} → {np.mean(blend):.5f} (по фолдам {[round(v, 5) for v in blend]}); "
                 f"выигрыш на невиденных фолдах {[round(g, 5) for g in gains]}")
        return chosen

    @staticmethod
    def _best_f1_threshold(truth_pos: np.ndarray, p: np.ndarray) -> tuple[float, float]:
        """Точный максимум бинарной F1 по всем порогам: (порог, F1)."""
        order = np.argsort(-p, kind="stable")
        tp = np.cumsum(truth_pos[order])
        ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(order) - 1]
        f1 = 2 * tp[ends] / np.maximum(1, truth_pos.sum() + ends + 1)
        k = int(np.argmax(f1))
        nxt = p[order[ends[k] + 1]] if ends[k] + 1 < len(order) else 0.0
        return float((p[order[ends[k]]] + nxt) / 2), float(f1[k])

    def _binary_threshold(self, members: list[str]) -> float | None:
        """Бинарная F1: порог вместо argmax (0.5), если он честно обобщается — выбор на одном фолде, проверка на других."""
        t = self.task
        if t.metric != "f1" or len(t.classes) != 2 or t.output_mode != "label" or len(self.split["folds"]) < 2:
            return None
        pos = str(t.classes[1])
        folds = []
        for i, (_, valid_idx) in enumerate(self.split["folds"]):
            p = np.mean([self.fold_preds[m][i] for m in members], axis=0)[:, 1].astype(float)
            folds.append((self.y.iloc[valid_idx].astype(str).to_numpy() == pos, p))

        def f1_at(truth, p, thr):
            pred = p >= thr
            denom = truth.sum() + pred.sum()
            return 2 * (truth & pred).sum() / denom if denom else 0.0

        gains, chosen = [], []
        for a, (truth_a, p_a) in enumerate(folds):
            thr, _ = self._best_f1_threshold(truth_a, p_a)
            chosen.append(thr)
            for b, (truth_b, p_b) in enumerate(folds):
                if b != a:
                    gains.append(f1_at(truth_b, p_b, thr) - f1_at(truth_b, p_b, 0.5))
        thr = float(np.median(chosen))
        if np.mean(gains) <= 0:
            self.log(f"Порог F1 не обобщается (выигрыш на невиденных фолдах {np.round(gains, 4).tolist()}) — оставляем 0.5")
            return None
        self.log(f"Порог F1 для класса {pos}: {thr:.4f} вместо 0.5; выигрыш на невиденных фолдах {np.round(gains, 4).tolist()}")
        return thr

    def _refit(self, record: dict, seconds: float, seed: int):
        spec = record["spec"]
        feats, cats, target_ops = self._plain_features(spec)
        X, X_test = self._with_target_ops(feats, target_ops, np.arange(self.n_train))
        n_est = None
        if record.get("best_iteration"):
            scale = self.n_train / max(len(self.split["fit_idx"]), 1)
            n_est = max(50, int(record["best_iteration"] * min(scale, 1.3)))
        model = Model(spec, self.task, seed=seed, n_jobs=self.n_jobs)
        model.fit(X, self.y, cats, seconds=seconds, n_estimators=n_est)
        return model.predict(X_test), n_est

    def finalize(self) -> None:
        """Дообучение на всём train. Каждый член ансамбля — на нескольких seed (среднее): разброс одной модели
        от seed заметный, усреднение 3 seed дало −0.5 RMSE и +0.015 Macro F1 на обоих фолдах."""
        if not self.best:
            self.log("Ни один эксперимент не завершился успешно — остаётся запасной prior-submission")
            self.final_members = []
            return
        by_id = {h["id"]: h for h in self.history}
        try:
            members = self.select_ensemble()
        except Exception as exc:
            self.log(f"Отбор ансамбля не удался ({exc}); берём одну модель")
            members = [self.best["id"]]
        counts = pd.Series(members).value_counts()
        order = list(counts.index)  # самые весомые — первыми
        try:
            self.threshold = self._binary_threshold(list(members))
        except Exception as exc:
            self.log(f"Подбор порога не удался ({exc}); оставляем argmax")
            self.threshold = None
        n_seeds = max(1, int(os.getenv("AGENT_FINAL_SEEDS", 3)))
        self.log(f"Финал: {order} {self.task.metric}={self.best['score']:.6f} (лучшая одиночная); "
                 f"дообучаем на всём train, до {n_seeds} seed на модель")
        ratio = self.n_train / max(len(self.split["fit_idx"]), 1)
        per_member: dict[str, list] = {m: [] for m in order}
        # Сначала по одному seed каждому члену, потом добавляем seed по кругу — пока есть время.
        for round_ in range(n_seeds):
            for exp_id in order:
                record = by_id[exp_id]
                need = record["seconds"] / len(self.split["folds"]) * ratio * 1.3 + 5
                if any(per_member.values()) and self.remaining - 10 < need:
                    continue
                try:
                    pred, n_est = self._refit(record, seconds=max(20.0, self.remaining - 10), seed=self.seed + round_)
                    per_member[exp_id].append(pred)
                    self.log(f"  дообучена {exp_id} seed={self.seed + round_} ({n_est} итераций)")
                except Exception as exc:
                    self.log(f"  дообучение {exp_id} seed={self.seed + round_} не удалось: {exc}")
        done = [m for m in order if per_member[m]]
        if done:
            preds = [np.mean(per_member[m], axis=0) for m in done]
            weights = np.asarray([counts[m] for m in done], float)
            blend = np.average(np.stack(preds), axis=0, weights=weights)
            seeds_used = {m: len(per_member[m]) for m in done}
            self.write_submission(blend, note=f"final ({seeds_used} seed, весь train)")
        else:
            self.log("Финальное дообучение не удалось; остаётся safety-submission")
        order = done
        self.final_members = order

    def write_submission(self, pred, note: str) -> None:
        t = self.task
        columns = list(pd.read_csv(t.sample_path, nrows=0).columns) if t.sample_path else [t.id_col, t.target]
        ids = self.test[t.id_col].to_numpy() if t.id_col else np.arange(len(self.test))
        if t.output_mode == "proba_multi":
            proba = np.asarray(pred, float)
            by_name = {str(c): j for j, c in enumerate(t.classes)}
            out = pd.DataFrame({columns[0]: ids, **{c: proba[:, by_name[str(c)]] for c in columns[1:]}})
        else:
            if t.task_type == "regression":
                y = self.y.astype(float)
                values = np.clip(np.asarray(pred, float), y.min(), y.max())
            elif t.output_mode == "proba":
                values = np.asarray(pred)[:, 1]
            elif getattr(self, "threshold", None) is not None:
                values = np.asarray(t.classes, dtype=object)[(np.asarray(pred)[:, 1] >= self.threshold).astype(int)]
            else:
                values = np.asarray(t.classes, dtype=object)[np.argmax(pred, axis=1)]
            out = pd.DataFrame({columns[0]: ids, columns[1]: values})
        if len(out) != len(self.test) or out.isna().any().any():
            raise ValueError("некорректные предсказания для submission")
        tmp = self.submission_path.with_suffix(".tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, self.submission_path)
        self.log(f"submission.csv записан: {note}")

    def run(self) -> dict:
        self.setup()
        try:
            self.research()
        except Exception as exc:
            self.log(f"Цикл исследования прерван: {exc}")
        self.finalize()
        report = {"task": self.task.summary(), "validation": self.split["note"], "shift": self.shift,
                  "final_members": getattr(self, "final_members", None),
                  "ensemble_scores": getattr(self, "ensemble_scores", None),
                  "best": {k: v for k, v in (self.best or {}).items() if k != "diagnostics"},
                  "experiments": len(self.history), "elapsed_seconds": round(self.elapsed, 1),
                  "llm": {"model": self.llm.model, "calls": self.llm.calls, "tokens": self.llm.used_tokens} if self.llm else None,
                  "submission": str(self.submission_path), "run_dir": str(self.run_dir)}
        (self.run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self.log(f"Готово за {self.elapsed:.0f}s: лучший {report['best'].get('id')} = {report['best'].get('score')}")
        return report
