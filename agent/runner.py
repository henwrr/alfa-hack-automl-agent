"""Главный цикл агента: понять задачу -> бейзлайн -> [диагностика -> гипотезы -> эксперименты] -> финальная модель."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostics import diagnose
from .features import apply_ops
from .llm import LLM, LLMError
from .metrics import better, score
from .models import Model
from .researcher import LLMResearcher, OfflineResearcher, normalize_spec
from .task import discover, profile
from .validation import adversarial_check, choose_split


class Agent:
    def __init__(self, task_path, time_budget: float = 600, seed: int = 42, n_jobs: int | None = None,
                 offline: bool = False, run_dir: str | None = None, experiments_per_round: int = 2):
        self.t0 = time.time()
        self.task_path = Path(task_path).resolve()
        self.time_budget = float(time_budget)
        self.seed = seed
        self.n_jobs = n_jobs or min(8, os.cpu_count() or 4)
        self.offline = offline
        self.per_round = experiments_per_round
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
        self.profile = profile(t, self.train, self.test)
        self.split = choose_split(t, self.train, self.test, seed=self.seed)
        self.log(f"Валидация: {self.split['note']}")
        self.y = self.train[t.target].reset_index(drop=True)
        self.all_X = pd.concat([self.train[t.feature_cols], self.test[t.feature_cols]], ignore_index=True)
        self.n_train = len(self.train)
        try:
            exclude = [self.split["column"]] if self.split["column"] else []
            self.shift = adversarial_check(t, self.train, self.test, seed=self.seed, n_jobs=self.n_jobs, exclude=exclude)
            if self.split["column"]:
                self.shift["note"] = f"'{self.split['column']}' исключён: test целиком в другом периоде по нему"
            self.log(f"Сдвиг train/test: AUC={self.shift.get('auc')} — {self.shift.get('verdict')}; "
                     f"топ: {[d['feature'] for d in self.shift.get('top_shifted_features', [])[:5]]}")
        except Exception as exc:  # диагностика не должна ронять агента
            self.shift = {"error": str(exc)[:300]}
        llm = LLM()
        if not self.offline and llm.available:
            self.researcher = LLMResearcher(llm, self.log)
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
    def _features(self, spec):
        feats, cats = apply_ops(self.all_X, self.task.cat_cols, spec.get("features"))
        return feats.iloc[:self.n_train].reset_index(drop=True), feats.iloc[self.n_train:].reset_index(drop=True), cats

    def _avg_experiment_seconds(self) -> float:
        times = [h["seconds"] for h in self.history if h["status"] == "ok"]
        return float(np.median(times)) if times else 60.0

    def _refit_estimate(self) -> float:
        if not self.best:
            return 30.0
        ratio = self.n_train / max(len(self.split["fit_idx"]), 1)
        return self.best["seconds"] * ratio * 1.3 + 15

    def run_experiment(self, hypothesis: str, raw_spec: dict, source: str) -> dict:
        exp_id = f"exp_{len(self.history) + 1:03d}"
        record = {"id": exp_id, "source": source, "hypothesis": hypothesis, "status": "error"}
        started = time.time()
        try:
            spec = normalize_spec(raw_spec, self.task, self.best["spec"] if self.best else None)
            record["spec"] = spec
            key = json.dumps(spec, sort_keys=True)
            if key in self.seen_specs:
                raise ValueError("такая спецификация уже проверялась")
            self.seen_specs.add(key)
            X, X_test, cats = self._features(spec)
            fit_idx, valid_idx = self.split["fit_idx"], self.split["valid_idx"]
            limit = max(30.0, min(self.remaining - self._refit_estimate() - 10, max(90.0, 0.35 * self.time_budget)))
            model = Model(spec, self.task, seed=self.seed, n_jobs=self.n_jobs)
            model.fit(X.iloc[fit_idx], self.y.iloc[fit_idx], cats, seconds=limit)
            pred = model.predict(X.iloc[valid_idx])
            value = score(self.task, self.y.iloc[valid_idx], pred)
            if not np.isfinite(value):
                raise ValueError("метрика не конечна")
            previous = self.best["score"] if self.best else None
            improved = better(value, previous, self.task.maximize)
            record.update(status="ok", score=round(value, 6), best_iteration=model.best_iteration,
                          delta_vs_best=None if previous is None else round(value - previous, 6),
                          verdict="улучшение — новая лучшая модель" if improved else "не лучше текущей лучшей")
            record["seconds"] = round(time.time() - started, 1)
            if improved:
                diag = diagnose(self.task, model, X.iloc[fit_idx], self.y.iloc[fit_idx], X.iloc[valid_idx],
                                self.y.iloc[valid_idx], pred, self.train.iloc[valid_idx][self.task.feature_cols])
                self.best = {"id": exp_id, "score": value, "spec": spec, "seconds": record["seconds"],
                             "best_iteration": model.best_iteration, "diagnostics": diag}
                self.write_submission(model.predict(X_test), note=f"safety ({exp_id}, обучена на fit-части)")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:1500]
            record["seconds"] = round(time.time() - started, 1)
            if not isinstance(exc, (ValueError, KeyError)):
                record["traceback"] = traceback.format_exc()[-1500:]
        self.history.append(record)
        self.journal(record)
        mark = "★" if record.get("verdict", "").startswith("улучшение") else " "
        self.log(f"{mark} {exp_id} [{source}] {record['status']} "
                 f"{self.task.metric}={record.get('score')} ({record['seconds']}s) — {hypothesis[:140]}"
                 + (f" | ошибка: {record['error'][:200]}" if record["status"] != "ok" else ""))
        return record

    # ---------- исследование ----------
    def _state(self, max_experiments: int) -> dict:
        history = [{k: h.get(k) for k in ("id", "source", "hypothesis", "spec", "status", "score", "delta_vs_best",
                                          "verdict", "seconds", "best_iteration", "error")} for h in self.history]
        best = None
        if self.best:
            best = {k: self.best[k] for k in ("id", "score", "spec", "best_iteration", "diagnostics")}
        return {"task": self.task.summary(), "description": self.task.description, "profile": self.profile,
                "validation": {"kind": self.split["kind"], "note": self.split["note"],
                               "fit_rows": len(self.split["fit_idx"]), "valid_rows": len(self.split["valid_idx"])},
                "shift": self.shift, "history": history, "best": best, "errors": self.errors,
                "seen_specs": self.seen_specs,
                "budget": {"remaining_seconds": round(self.remaining), "typical_experiment_seconds": round(self._avg_experiment_seconds()),
                           "max_experiments": max_experiments}}

    def _can_start_experiment(self) -> bool:
        return self.remaining > self._refit_estimate() + self._avg_experiment_seconds() * 1.1 + 10

    def research(self) -> None:
        self.run_experiment("бейзлайн: LightGBM с дефолтами и early stopping", {"family": "lightgbm"}, "host")
        llm_failures = 0
        rounds = 0
        while self._can_start_experiment():
            rounds += 1
            fit = int((self.remaining - self._refit_estimate()) // max(self._avg_experiment_seconds(), 1))
            state = self._state(max(1, min(self.per_round, fit)))
            researcher = self.researcher
            try:
                proposal = researcher.propose(state)
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
            if not proposal["experiments"]:
                self.log("Идеи закончились — переходим к финальной модели")
                break
            for exp in proposal["experiments"][:state["budget"]["max_experiments"]]:
                if not self._can_start_experiment():
                    break
                self.run_experiment(str(exp.get("hypothesis", ""))[:500], exp.get("spec") or {}, researcher.name)

    # ---------- финал ----------
    def finalize(self) -> None:
        if not self.best:
            raise RuntimeError("ни один эксперимент не завершился успешно")
        spec = self.best["spec"]
        self.log(f"Финал: {self.best['id']} {self.task.metric}={self.best['score']:.6f}; дообучаем на всём train")
        try:
            X, X_test, cats = self._features(spec)
            n_est = None
            if self.best["best_iteration"]:
                scale = self.n_train / max(len(self.split["fit_idx"]), 1)
                n_est = max(50, int(self.best["best_iteration"] * min(scale, 1.3)))
            model = Model(spec, self.task, seed=self.seed, n_jobs=self.n_jobs)
            model.fit(X, self.y, cats, seconds=max(20.0, self.remaining - 10), n_estimators=n_est)
            self.write_submission(model.predict(X_test), note=f"final ({self.best['id']}, весь train, {n_est} итераций)")
        except Exception as exc:
            self.log(f"Финальное дообучение не удалось ({exc}); остаётся safety-submission")

    def write_submission(self, pred, note: str) -> None:
        t = self.task
        if t.task_type == "regression":
            y = self.y.astype(float)
            values = np.clip(np.asarray(pred, float), y.min(), y.max())
        elif t.output_mode == "proba":
            values = np.asarray(pred)[:, 1]
        else:
            values = np.asarray(t.classes, dtype=object)[np.argmax(pred, axis=1)]
        if len(values) != len(self.test) or pd.isna(values).any():
            raise ValueError("некорректные предсказания для submission")
        columns = list(pd.read_csv(t.sample_path, nrows=0).columns) if t.sample_path else [t.id_col, t.target]
        out = pd.DataFrame({columns[0]: self.test[t.id_col].to_numpy() if t.id_col else np.arange(len(self.test)),
                            columns[1]: values})
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
                  "best": {k: v for k, v in (self.best or {}).items() if k != "diagnostics"},
                  "experiments": len(self.history), "elapsed_seconds": round(self.elapsed, 1),
                  "llm": {"model": self.llm.model, "calls": self.llm.calls, "tokens": self.llm.used_tokens} if self.llm else None,
                  "submission": str(self.submission_path), "run_dir": str(self.run_dir)}
        (self.run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self.log(f"Готово за {self.elapsed:.0f}s: лучший {report['best'].get('id')} = {report['best'].get('score')}")
        return report
