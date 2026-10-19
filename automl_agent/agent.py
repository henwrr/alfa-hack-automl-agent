"""Research orchestration; the host owns scoring and model selection."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4

from .contracts import ExperimentResult


def select_best(results, maximize):
    valid = [r for r in results if r.status == "ok" and r.metric is not None and math.isfinite(r.metric)]
    return (max if maximize else min)(valid, key=lambda r: r.metric) if valid else None


@contextmanager
def _run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process already owns this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


class ResearchRun:
    def __init__(self, config):
        initialized_at = time.monotonic()
        from .data import discover_task
        from .journal import EventJournal
        from .runtime import Budget

        self.config = config
        self.task = discover_task(config.task_path, seed=config.seed, validation_fraction=config.validation_fraction,
                                  overrides={k: v for k, v in {"target": config.target, "task_type": config.task_type, "metric": config.metric}.items() if v is not None})
        self.directory = config.run_dir or config.task_path.parent / ".automl" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
        self.output = config.task_path.parent / "submission.csv"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.state = _read_json(self.state_path, {})
        if self.state and not config.resume:
            raise ValueError("Run directory already contains state; use --resume or a new --run-dir")
        if config.resume and not self.state:
            raise ValueError("No saved state in --run-dir")
        signature = {"data": self.task.data_fingerprint, "seed": config.seed, "validation_fraction": config.validation_fraction,
                     "target": self.task.target, "metric": self.task.metric, "task_type": self.task.task_type}
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        if self.state and self.state.get("settings_fingerprint") != self.signature:
            raise ValueError("Input data or split settings changed (fingerprint mismatch); start a new run")
        self.journal = EventJournal(self.directory / "events.jsonl", secrets=[config.api_key] if config.api_key else [])
        self.results = []
        # The event log is authoritative if the process died before state.json was updated.
        completed = {r["trial_id"]: r for r in self.state.get("trials", [])}
        events = [{**event, **event.get("payload", {})} for event in self.journal.read_events()]
        for event in events:
            if event.get("event") == "trial_completed":
                result = event.get("result", {})
                if "trial_id" in result:
                    completed[result["trial_id"]] = result
        self.results = [ExperimentResult.from_dict(r) for r in completed.values()]
        spent = float(self.state.get("elapsed_seconds", 0))
        if self.state.get("active"):
            spent += max(0.0, time.time() - self.state.get("checkpoint_epoch", time.time()))
        spent += time.monotonic() - initialized_at
        self.budget = Budget(config.time_budget, reserve_seconds=config.reserve_seconds, spent_seconds=spent)
        self.tokens_used = int(self.state.get("total_tokens", 0))
        self.steps = int(self.state.get("llm_calls", 0))
        self.feedback = ""
        self.stop_reason = "offline" if config.offline else "max_trials"
        self.finish_requested = False
        self.recent_steps = []
        self.split = None
        self.profile = None
        self._recover_calls(events)

    def _recover_calls(self, events):
        settled = {e.get("call_id") for e in events if e.get("event") == "tool_result"}
        for event in events:
            if event.get("event") == "tool_call" and event.get("call_id") not in settled:
                self.journal.append("tool_result", call_id=event.get("call_id"), name=event.get("name"),
                                    result={"ok": False, "error": {"code": "INTERRUPTED", "message": "Previous process ended before a durable result; not replayed"}})
        finished = {r.trial_id for r in self.results}
        for event in events:
            if event.get("event") == "trial_started" and event.get("trial_id") not in finished:
                trial_id = event["trial_id"]
                result = ExperimentResult(trial_id, "interrupted", str(self.directory / trial_id), event.get("code_hash", ""), event.get("hypothesis", ""), error="Outcome unknown after interrupted process; not replayed")
                self.results.append(result)
                self.journal.append("trial_completed", result=result.to_dict())
                finished.add(trial_id)

    def checkpoint(self, active=True):
        from .journal import atomic_write_json

        best = select_best(self.results, self.task.maximize)
        state = {"version": 1, "data_fingerprint": self.task.data_fingerprint, "settings_fingerprint": self.signature,
                 "elapsed_seconds": self.budget.elapsed(), "checkpoint_epoch": time.time(), "active": active,
                 "total_tokens": self.tokens_used, "llm_calls": self.steps, "trials": [r.to_dict() for r in self.results],
                 "selected_trial": best.trial_id if best else None, "stop_reason": self.stop_reason}
        atomic_write_json(self.state_path, state)
        atomic_write_json(self.directory / "experiments.json", {"trials": state["trials"], "selected_trial": state["selected_trial"]})

    def _emergency_submission(self):
        import pandas as pd
        from .evaluation import publish_submission

        y = pd.read_csv(self.split["train_target"])[self.task.target]
        constant = y.mode().iloc[0] if self.task.task_type == "classification" else float(y.mean())
        rows = len(pd.read_csv(self.split["test_features"]))
        path = self.directory / "fallback_predictions.csv"
        pd.DataFrame({self.task.target: [constant] * rows}).to_csv(path, index=False)
        if not self.config.resume or not self.output.exists():
            publish_submission(self.task, path, self.output)
            self.journal.append("fallback_written", submission_path=str(self.output), strategy="mode" if self.task.task_type == "classification" else "mean")

    def _next_trial_id(self):
        indices = [int(p.name.split("_")[1]) for p in self.directory.glob("trial_*") if p.is_dir() and p.name.split("_")[1].isdigit()]
        return f"trial_{max(indices, default=0) + 1:04d}"

    def experiment(self, hypothesis, code, is_baseline=False):
        from .evaluation import publish_submission, score_trial
        from .journal import atomic_write_json
        from .runtime import run_process
        from .tools import validate_candidate_code

        if len(self.results) >= self.config.max_trials:
            raise ValueError("Experiment budget exhausted")
        remaining = self.budget.research_remaining()
        if remaining <= 0:
            raise ValueError("Research deadline reached; finalization time is reserved")
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if any(r.code_hash == code_hash for r in self.results):
            raise ValueError("Duplicate code: propose a different hypothesis or fix the previous error")
        trial_id = self._next_trial_id()
        directory = self.directory / trial_id
        directory.mkdir()
        (directory / "code.py").write_text(code, encoding="utf-8")
        result = ExperimentResult(trial_id, "error", str(directory), code_hash, hypothesis, is_baseline=is_baseline)
        self.journal.append("trial_started", trial_id=trial_id, code_hash=code_hash, hypothesis=hypothesis, is_baseline=is_baseline)
        self.checkpoint()
        try:
            validate_candidate_code(code)
            manifest = {"mode": "evaluate", "context": self.task.model_context(self.config.n_jobs),
                        "inputs": {key: self.split[key] for key in ("train_features", "train_target", "validation_features", "test_features")},
                        "code_path": str(directory / "code.py"), "artifact_dir": str(directory), "save_model": True}
            atomic_write_json(directory / "manifest.json", manifest)
            process = run_process([sys.executable, "-m", "automl_agent.worker", "--manifest", str(directory / "manifest.json")],
                                  cwd=directory, timeout=min(self.config.trial_timeout, remaining), n_jobs=self.config.n_jobs,
                                  memory_limit_mb=self.config.memory_limit_mb, env={"PYTHONPATH": str(Path(__file__).resolve().parent.parent)})
            atomic_write_json(directory / "process.json", process.to_dict())
            result.seconds = process.seconds
            result.status = process.status
            if process.status == "ok":
                result.metrics = score_trial(self.task, self.split, directory)
                result.metric = float(result.metrics[self.task.metric])
                if not math.isfinite(result.metric):
                    raise ValueError("Evaluator returned a non-finite metric")
                result.prediction_path = str(directory / "test_predictions.csv")
                publish_submission(self.task, Path(result.prediction_path), directory / "submission.csv")
                if (directory / "model.joblib").exists():
                    result.model_path = str(directory / "model.joblib")
            else:
                result.error = (process.stderr or process.stdout or process.status)[-4000:]
        except ValueError as exc:
            result.status = "invalid_output" if (directory / "process.json").exists() else "invalid_code"
            result.error = str(exc)[-4000:]
        except Exception as exc:
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"[-4000:]
        self.results.append(result)
        self.journal.append("trial_completed", result=result.to_dict())
        best = select_best(self.results, self.task.maximize)
        if best is result:
            publish_submission(self.task, Path(result.prediction_path), self.output)
            self.journal.append("candidate_selected", trial_id=trial_id, metric=result.metric, submission_path=str(self.output))
        self.checkpoint()
        print(f"{trial_id}: {result.status}, {self.task.metric}={result.metric}", file=sys.stderr, flush=True)
        if result.status == "cancelled":
            raise KeyboardInterrupt
        return result

    def _llm_experiment(self, hypothesis, code):
        result = self.experiment(hypothesis, code)
        if result.status != "ok":
            raise ValueError(f"{result.trial_id} {result.status}: {result.error}")
        best = select_best(self.results, self.task.maximize)
        return {"trial_id": result.trial_id, "metrics": result.metrics, "seconds": result.seconds,
                "selected": best.trial_id == result.trial_id, "best_metric": best.metric}

    def _finish(self, reason):
        if not any(r.status == "ok" and not r.is_baseline for r in self.results):
            raise ValueError("Run at least one successful generated experiment before finishing")
        self.finish_requested = True
        self.stop_reason = "model_finished"
        return {"finished": True, "reason": reason}

    def research(self):
        from .prompts import build_messages
        from .providers import OllamaProvider, OpenAICompatibleProvider, ProviderError
        from .tools import ToolRegistry, parse_response_calls

        provider_type = OllamaProvider if self.config.provider == "ollama" else OpenAICompatibleProvider
        provider = provider_type(base_url=self.config.base_url, api_key=self.config.api_key, model=self.config.model, extra_body=self.config.extra_body)
        registry = ToolRegistry()
        registry.register("run_experiment", "Test a concrete ML hypothesis with complete build_model(context) Python source.",
                          {"type": "object", "properties": {"hypothesis": {"type": "string", "minLength": 1, "maxLength": 6000}, "code": {"type": "string", "minLength": 1, "maxLength": 64000}}, "required": ["hypothesis", "code"], "additionalProperties": False}, self._llm_experiment)
        registry.register("finish_research", "Stop research after a successful generated candidate; host refits the best model.",
                          {"type": "object", "properties": {"reason": {"type": "string", "minLength": 1, "maxLength": 4000}}, "required": ["reason"], "additionalProperties": False}, self._finish)
        compact = False
        while len(self.results) < self.config.max_trials and self.steps < self.config.max_steps and not self.finish_requested:
            if self.budget.research_remaining() <= 1:
                self.stop_reason = "time_budget"
                break
            best = select_best(self.results, self.task.maximize)
            messages = build_messages(self.task, self.profile, self.results, best,
                                      {"n_jobs": self.config.n_jobs, "remaining_seconds": round(self.budget.research_remaining(), 2),
                                       "trial_timeout": self.config.trial_timeout, "remaining_trials": self.config.max_trials - len(self.results)},
                                      feedback=self.feedback, context_chars=self.config.context_chars, compact=compact)
            # Keep complete recent assistant/tool groups. Older attempts remain
            # available through the structured ledger and the durable journal.
            for step_messages in self.recent_steps[-(1 if compact else 2):]:
                messages.extend(step_messages)
            estimate = max(1, len(json.dumps(messages, ensure_ascii=False)) // 3)
            available_tokens = self.config.max_tokens - self.tokens_used - estimate
            if available_tokens < 128:
                self.stop_reason = "token_budget"
                break
            output_cap = min(self.config.response_tokens, available_tokens)
            self.steps += 1
            request_id = f"request_{self.steps:04d}"
            self.journal.append("llm_request", request_id=request_id, provider=self.config.provider, model=self.config.model,
                                messages=messages, tools=registry.schemas(), max_tokens=output_cap)
            self.checkpoint()
            try:
                response = provider.complete(messages, registry.schemas(), timeout=min(self.config.llm_timeout, self.budget.research_remaining()), max_tokens=output_cap)
            except ProviderError as exc:
                # Failed requests may still be billable; reserve their estimated input.
                self.tokens_used += estimate
                self.journal.append("llm_error", request_id=request_id, code=exc.code, error=str(exc))
                if exc.code in {"context_overflow", "CONTEXT_WINDOW_EXCEEDED", "context_length_exceeded"} and not compact:
                    compact = True
                    self.recent_steps.clear()
                    self.feedback = "Context overflow: the host reduced auxiliary context. Return one compact tool call."
                    self.journal.append("context_compacted", reason="provider_context_overflow")
                    continue
                self.stop_reason = "provider_error"
                break
            self.tokens_used += response.total_tokens or response.prompt_tokens + response.completion_tokens or estimate + len(json.dumps(response.message)) // 3
            self.journal.append("llm_response", request_id=request_id, response=response.to_dict())
            try:
                calls = parse_response_calls(response.message)
            except ValueError as exc:
                self.feedback = str(exc)
                self.journal.append("llm_protocol_error", request_id=request_id, error=str(exc))
                self.checkpoint()
                continue
            step_messages = [{"role": "assistant", "content": response.message.get("content") or "", "tool_calls": calls}]
            history_safe = True
            for call in calls:
                function = call["function"]
                try:
                    logged_arguments = json.loads(function["arguments"])
                except ValueError:
                    logged_arguments = function["arguments"]
                    history_safe = False
                self.journal.append("tool_call", call_id=call["id"], name=function["name"], arguments=logged_arguments)
                self.checkpoint()
                if self.finish_requested:
                    result = {"ok": False, "error": {"code": "CANCELLED", "message": "Research already finished earlier in this step"}}
                else:
                    result = registry.execute(function["name"], function["arguments"])
                self.journal.append("tool_result", call_id=call["id"], name=function["name"], result=result)
                self.feedback = json.dumps(result, ensure_ascii=False, default=str)
                step_messages.append({"role": "tool", "tool_call_id": call["id"], "name": function["name"], "content": self.feedback})
                self.checkpoint()
            if history_safe:
                self.recent_steps.append(step_messages)
        if self.steps >= self.config.max_steps and not self.finish_requested:
            self.stop_reason = "max_steps"
        if len(self.results) >= self.config.max_trials:
            self.stop_reason = "max_trials"

    def finalize(self, best):
        from .evaluation import publish_submission
        from .journal import atomic_write_json
        from .runtime import run_process

        if best is None or self.budget.remaining() <= 2:
            return {"status": "skipped", "reason": "No candidate or no remaining time"}
        directory = self.directory / ("final_" + uuid4().hex[:8])
        directory.mkdir()
        source = Path(best.artifact_dir) / "code.py"
        (directory / "code.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = {"mode": "finalize", "context": self.task.model_context(self.config.n_jobs), "inputs": {
            "train_features": self.split["full_train_features"], "train_target": self.split["full_train_target"], "test_features": self.split["test_features"]},
            "code_path": str(directory / "code.py"), "artifact_dir": str(directory), "save_model": True}
        atomic_write_json(directory / "manifest.json", manifest)
        self.journal.append("final_fit_started", trial_id=best.trial_id, artifact_dir=str(directory))
        self.checkpoint()
        process = run_process([sys.executable, "-m", "automl_agent.worker", "--manifest", str(directory / "manifest.json")],
                              cwd=directory, timeout=min(self.config.trial_timeout, self.budget.remaining() - 1), n_jobs=self.config.n_jobs,
                              memory_limit_mb=self.config.memory_limit_mb, env={"PYTHONPATH": str(Path(__file__).resolve().parent.parent)})
        atomic_write_json(directory / "process.json", process.to_dict())
        outcome = {"status": process.status, "artifact_dir": str(directory), "seconds": process.seconds}
        if process.status == "ok":
            try:
                publish_submission(self.task, directory / "test_predictions.csv", self.output)
                outcome["model_path"] = str(directory / "model.joblib") if (directory / "model.joblib").exists() else None
            except Exception as exc:
                outcome.update(status="invalid_output", error=str(exc))
        else:
            outcome["error"] = (process.stderr or process.stdout)[-3000:]
        self.journal.append("final_fit_completed", result=outcome)
        return outcome

    def execute(self):
        from .data import prepare_split, profile_task
        from .evaluation import validate_submission
        from .journal import atomic_write_json
        from .modeling import baseline_code

        with _run_lock(self.directory / "run.lock"):
            if self.config.resume and not self.state.get("active") and (self.directory / "report.json").exists():
                validate_submission(self.task, self.output)
                return _read_json(self.directory / "report.json")
            self.journal.append("run_resumed" if self.config.resume else "run_started", config=self.config.public_dict(), data_fingerprint=self.task.data_fingerprint)
            atomic_write_json(self.directory / "config.json", self.config.public_dict())
            atomic_write_json(self.directory / "task.json", self.task.to_dict())
            versions = {}
            for name in ("numpy", "pandas", "scikit-learn", "catboost", "httpx", "psutil", "cloudpickle"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = "unavailable"
            atomic_write_json(self.directory / "environment.json", {"python": sys.version, "platform": platform.platform(), "packages": versions})
            self.profile = profile_task(self.task)
            atomic_write_json(self.directory / "profile.json", self.profile)
            self.split = prepare_split(self.task, self.directory)
            atomic_write_json(self.directory / "split_manifest.json", self.split)
            self._emergency_submission()
            self.checkpoint()
            try:
                if not self.results and self.budget.research_remaining() > 0:
                    self.experiment("Reliable baseline before LLM research", baseline_code(self.task, n_jobs=self.config.n_jobs), is_baseline=True)
                if not self.config.offline:
                    self.research()
                best = select_best(self.results, self.task.maximize)
                final_fit = self.finalize(best)
            except KeyboardInterrupt:
                self.stop_reason = "interrupted"
                self.journal.append("run_interrupted")
                self.checkpoint()
                raise
            except Exception as exc:
                self.stop_reason = "orchestration_error"
                self.journal.append("run_error", error=f"{type(exc).__name__}: {exc}")
                best = select_best(self.results, self.task.maximize)
                final_fit = {"status": "skipped", "reason": str(exc)}
            validation = validate_submission(self.task, self.output)
            llm_trials = sum(r.status == "ok" and not r.is_baseline for r in self.results)
            status = "offline" if self.config.offline else ("completed" if llm_trials else "degraded")
            report = {"status": status, "stop_reason": self.stop_reason, "task_type": self.task.task_type, "metric": self.task.metric,
                      "best_metric": best.metric if best else None, "selected_trial": best.trial_id if best else None,
                      "trials": [r.to_dict() for r in self.results], "llm_calls": self.steps, "llm_trials": llm_trials,
                      "total_tokens": self.tokens_used, "elapsed_seconds": round(self.budget.elapsed(), 3),
                      "submission_path": str(self.output), "run_dir": str(self.directory), "final_fit": final_fit, "validation": validation}
            atomic_write_json(self.directory / "report.json", report)
            rows = ["# AutoResearch run", "", f"Status: {status}; stop: {self.stop_reason}", f"Metric: {self.task.metric}; best validation: {report['best_metric']}",
                    f"Submission: {self.output}", "", "| Trial | Status | Validation metric | Seconds |", "|---|---|---:|---:|"]
            rows += [f"| {r.trial_id} | {r.status} | {r.metric} | {r.seconds:.2f} |" for r in self.results]
            rows += ["", "The validation metric belongs to the fixed holdout experiment. The final refit uses all training rows.",
                     "Generated code runs in a supervised subprocess. Use the deployment container for OS-level filesystem/network isolation."]
            (self.directory / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
            self.journal.append("run_completed", report=report)
            self.checkpoint(active=False)
            return report


def run(config):
    return ResearchRun(config).execute()
