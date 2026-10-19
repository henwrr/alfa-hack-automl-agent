"""Human and structured projections of the redacted event journal.

The JSONL event journal remains authoritative. These two files are deliberately
rebuildable views and only expose allowlisted public facts and decision
summaries; provider messages, tool arguments, source code, and response bodies
never enter either projection.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterator
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover - MVP platforms are macOS/Linux
    fcntl = None

from .journal import EventJournal


_PRIVATE_KEY_PARTS = (
    "prompt", "message", "code", "raw", "thinking", "thought", "reasoning",
    "chain_of_thought", "internal", "authorization", "credential", "api_key",
    "secret", "password", "argument", "response", "stderr", "stdout", "traceback",
)
_LLM_EVENTS = {
    "llm_request", "llm_response", "llm_error", "llm_protocol_error",
    "llm_response_consumed", "llm_response_recovered",
}
_WORKER_EVENTS = {"trial_completed", "final_fit_completed"}


class AuditJournal(EventJournal):
    """Append canonical events, then maintain Russian and JSONL audit views."""

    def __init__(self, path: Path, secrets=(), echo: bool = True) -> None:
        super().__init__(path, secrets=secrets)
        self.echo = bool(echo)
        self.agent_log_path = self.path.parent / "agent.log"
        self.decisions_path = self.path.parent / "decisions.jsonl"
        self._projection_lock_path = self.path.parent / ".audit-projections.lock"
        try:
            self._rebuild_projections()
        except Exception as exc:  # projections must not prevent canonical recovery/use
            self._warn(None, "проекции при запуске", exc)

    def append(self, event: str, **payload: Any) -> dict:
        # Make the durable event first. If a derived view, lock, or terminal
        # fails, callers still receive the canonical record and a clear warning.
        record = super().append(event, **payload)
        projection_error = None
        try:
            self._rebuild_projections()
        except Exception as exc:
            projection_error = exc
        echo_error = None
        if self.echo:
            try:
                line = _render_human(_project_event(record, {}))
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
            except Exception as exc:
                echo_error = exc
        if projection_error is not None:
            self._warn(record["seq"], "обновление проекций", projection_error)
        if echo_error is not None:
            self._warn(record["seq"], "вывод в stderr", echo_error)
        return record

    @contextmanager
    def _projection_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._projection_lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with self._thread_lock:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _rebuild_projections(self) -> None:
        # Re-read under a cross-process projection lock. Rewriting from the
        # source of truth prevents duplicate lines after interrupted writes.
        with self._projection_lock():
            events = self.read_events()
            records = _project_events(events)
            human = "".join(_render_human(record) + "\n" for record in records)
            structured = "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
                for record in records
            )
            _write_projection(self.agent_log_path, human)
            _write_projection(self.decisions_path, structured)

    @staticmethod
    def _warn(sequence: int | None, stage: str, exc: Exception) -> None:
        if sequence is None:
            message = (
                f"Предупреждение: не удалось завершить {stage} ({type(exc).__name__}). "
                "events.jsonl остаётся источником истины; повторно откройте AuditJournal "
                "для восстановления проекций."
            )
        else:
            message = (
                f"Предупреждение: событие №{sequence} сохранено в events.jsonl, "
                f"но не удалось завершить {stage} ({type(exc).__name__}). "
                "Повторно откройте AuditJournal для восстановления проекций из events.jsonl."
            )
        try:
            sys.stderr.write(message + "\n")
            sys.stderr.flush()
        except Exception:
            # Canonical durability takes precedence when stderr itself is broken.
            pass


def _project_events(events: list[dict]) -> list[dict]:
    baseline_trials: dict[str, bool] = {}
    projected = []
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event.get("event") == "trial_started" and isinstance(payload.get("trial_id"), str):
            baseline_trials[payload["trial_id"]] = bool(payload.get("is_baseline"))
        projected.append(_project_event(event, baseline_trials))
    return projected


def _project_event(event: dict, baseline_trials: dict[str, bool]) -> dict:
    event_name = _bounded_text(event.get("event", "unknown"), 100)
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    source = _source_for(event_name, payload, baseline_trials)
    details = _details_for(event_name, payload)
    return {
        "seq": event.get("seq"),
        "timestamp": _bounded_text(event.get("timestamp", ""), 80),
        "event": event_name,
        "source": source,
        "details": details,
    }


def _source_for(event: str, payload: dict, baseline_trials: dict[str, bool]) -> str:
    explicit = str(payload.get("source", "")).strip().lower()
    if explicit in {"llm", "model", "provider"}:
        return "LLM"
    if explicit == "worker":
        return "worker"
    if explicit == "harness":
        return "harness"
    if event == "decision_recorded":
        return "harness"
    if event in _LLM_EVENTS or event == "tool_call":
        return "LLM"
    if event == "trial_started":
        return "harness" if payload.get("is_baseline") else "LLM"
    if event == "trial_completed":
        result = payload.get("result")
        trial_id = result.get("trial_id") if isinstance(result, dict) else None
        if (isinstance(result, dict) and result.get("is_baseline")) or (
            trial_id in baseline_trials and baseline_trials[trial_id]
        ):
            return "harness"
        return "worker"
    if event in _WORKER_EVENTS:
        return "worker"
    return "harness"


def _details_for(event: str, payload: dict) -> dict:
    if event in {"data_profiled", "validation_planned"}:
        return _selected(payload, ("summary",))
    if event == "research_strategy":
        return _selected(payload, ("strategy",))
    if event == "decision_recorded":
        return _selected(payload, (
            "trial_id", "hypothesis", "decision", "explanation_provided", "model_family",
            "observations", "rationale", "alternatives", "expected_effect", "risks",
            "previous_result_assessment",
        ))
    if event == "experiment_assessed":
        return _selected(payload, (
            "trial_id", "metric_name", "metric", "previous_best", "maximize",
            "selected", "delta", "reason", "model_details",
        ))
    if event == "research_finished":
        return _selected(payload, ("reason",))
    if event == "research_stopped":
        return _selected(payload, ("reason", "total_tokens", "elapsed_seconds"))
    if event == "submission_validated":
        return _selected(payload, ("validation", "submission_path"))
    if event in {"run_started", "run_resumed"}:
        config = payload.get("config")
        return {
            "data_fingerprint": _clean_public(payload.get("data_fingerprint")),
            "config": _pick(config, ("provider", "model", "offline", "max_trials", "time_budget")),
        }
    if event == "run_completed":
        report = payload.get("report")
        details = _pick(report, (
            "status", "stop_reason", "task_type", "metric", "best_metric", "selected_trial",
            "llm_calls", "llm_trials", "total_tokens", "elapsed_seconds", "decision_log", "decisions_path",
        ))
        if isinstance(report, dict) and isinstance(report.get("final_fit"), dict):
            final = _pick(report["final_fit"], ("status", "seconds"))
            if final:
                details["final_fit"] = final
        return details
    if event == "trial_started":
        return _selected(payload, ("trial_id", "hypothesis", "is_baseline"))
    if event == "trial_completed":
        result = payload.get("result")
        details = _pick(result, (
            "trial_id", "status", "metric", "metrics", "seconds", "hypothesis",
            "is_baseline", "model_details",
        ))
        if isinstance(result, dict) and result.get("status") != "ok":
            code = _short_error_code(result.get("error"), result.get("status"))
            if code:
                details["error_code"] = code
        return details
    if event == "candidate_selected":
        return _selected(payload, ("trial_id", "metric")) | {"selected": True}
    if event == "candidate_restored":
        return _selected(payload, ("trial_id", "metric")) | {"restored": True}
    if event == "fallback_written":
        return _selected(payload, ("strategy",))
    if event == "llm_request":
        return _selected(payload, ("request_id", "provider", "model", "max_tokens", "input_token_bound"))
    if event == "llm_response":
        response = payload.get("response")
        result = _selected(payload, ("request_id", "charged_tokens"))
        if isinstance(response, dict):
            usage = _pick(response, ("prompt_tokens", "completion_tokens", "total_tokens"))
            if usage:
                result["usage"] = usage
        return result
    if event == "llm_error":
        details = _selected(payload, ("request_id", "charged_tokens"))
        if isinstance(payload.get("code"), str):
            details["error_code"] = _bounded_text(payload["code"], 100)
        return details
    if event in {"llm_protocol_error", "llm_response_consumed", "llm_response_recovered"}:
        return _selected(payload, ("request_id",))
    if event == "context_compacted":
        return _selected(payload, ("reason",))
    if event == "tool_call":
        return _selected(payload, ("call_id", "name"))
    if event == "tool_result":
        return _tool_result_details(payload)
    if event == "final_fit_started":
        return _selected(payload, ("trial_id",))
    if event == "final_fit_completed":
        result = payload.get("result")
        details = _pick(result, ("status", "seconds", "model_details"))
        if isinstance(result, dict) and result.get("status") != "ok":
            code = _short_error_code(result.get("error"), result.get("status"))
            if code:
                details["error_code"] = code
            artifact_dir = result.get("artifact_dir")
            if isinstance(artifact_dir, str) and artifact_dir:
                details["process_ref"] = Path(artifact_dir).name + "/process.json"
        return details
    if event == "run_error":
        return {"error_code": _short_error_code(payload.get("error"), "orchestration_error")}
    if event == "run_interrupted":
        return {}
    # Unknown/legacy events retain their identity, but their unrecognized
    # payload is omitted until a public projection contract is defined.
    return {}


def _tool_result_details(payload: dict) -> dict:
    details = _selected(payload, ("call_id", "name"))
    result = payload.get("result")
    if not isinstance(result, dict):
        return details
    if "ok" in result:
        details["ok"] = bool(result["ok"])
    error = result.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        details["error_code"] = _bounded_text(error["code"], 100)
    value = result.get("value")
    if isinstance(value, dict):
        details["value"] = _pick(value, ("finished", "trial_id", "metrics", "seconds", "selected", "best_metric"))
    return details


def _selected(payload: dict, fields: tuple[str, ...]) -> dict:
    return {key: _clean_public(payload[key]) for key in fields if key in payload and _is_public_key(key)}


def _pick(value: Any, fields: tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        return {}
    return {key: _clean_public(value[key]) for key in fields if key in value and _is_public_key(key)}


def _is_public_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    if normalized == "error_code":
        return True
    return not any(part in normalized for part in _PRIVATE_KEY_PARTS)


def _clean_public(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _bounded_text(value, 6000)
    if isinstance(value, dict):
        result = {}
        for key, child in list(value.items())[:60]:
            if _is_public_key(key):
                result[_bounded_text(str(key), 100)] = _clean_public(child, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        items = [_clean_public(item, depth + 1) for item in value[:40]]
        if len(value) > 40:
            items.append("[ещё элементы опущены]")
        return items
    return _bounded_text(str(value), 300)


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _render_human(record: dict) -> str:
    event = record.get("event", "unknown")
    source = record.get("source", "harness")
    details = record.get("details", {})
    timestamp = record.get("timestamp", "")
    seq = record.get("seq", "?")
    label = _event_text(event, details)
    if event == "trial_completed" and details.get("status") != "ok":
        trial_id = details.get("trial_id", "trial")
        label += (
            f" Код ошибки: {_human_value(details.get('error_code'))}; "
            f"диагностика: events.jsonl #{seq} и {trial_id}/process.json."
        )
    elif event == "final_fit_completed" and details.get("status") != "ok":
        label += (
            f" Код ошибки: {_human_value(details.get('error_code'))}; "
            f"диагностика: events.jsonl #{seq}"
        )
        if details.get("process_ref"):
            label += f" и {details['process_ref']}"
        label += "."
    return f"[{timestamp}] #{seq} [{source}] {label}"


def _event_text(event: str, details: dict) -> str:
    if event == "run_started":
        return "Начат новый запуск исследования."
    if event == "run_resumed":
        return "Возобновлён запуск исследования."
    if event == "data_profiled":
        return f"Профиль данных: {_human_value(details.get('summary'))}."
    if event == "validation_planned":
        return f"План проверки: {_human_value(details.get('summary'))}."
    if event == "research_strategy":
        return f"Стратегия исследования: {_human_value(details.get('strategy'))}."
    if event == "decision_recorded":
        trial = details.get("trial_id", "без trial_id")
        hypothesis = details.get("hypothesis")
        decision = details.get("decision")
        text = f"Решение для {trial}"
        if hypothesis:
            text += f": гипотеза: {_human_value(hypothesis)}"
        fields = (
            ("model_family", "Семейство модели"),
            ("observations", "Наблюдения"),
            ("rationale", "Обоснование"),
            ("alternatives", "Рассмотренные альтернативы"),
            ("expected_effect", "Ожидаемый эффект"),
            ("risks", "Риски"),
            ("previous_result_assessment", "Оценка прошлого результата"),
        )
        for key, label in fields:
            value = details.get(key)
            if value is None:
                value = _get_nested(decision, key)
            if value is None:
                value = _get_nested(hypothesis, key)
            if value is not None:
                text += f"; {label}: {_human_value(value)}"
        if details.get("explanation_provided") is False and not any(
            key in details or key in (decision if isinstance(decision, dict) else {})
            for key in ("rationale", "observations", "expected_effect", "previous_result_assessment")
        ):
            text += "; объяснение не предоставлено"
        return text + "."
    if event == "experiment_assessed":
        metric = details.get("metric")
        metric_name = details.get("metric_name", "метрика")
        result = f"Фактическая валидация {details.get('trial_id', 'trial')}: {metric_name}={_human_value(metric)}"
        if "previous_best" in details:
            result += f", прошлый лучший результат={_human_value(details['previous_best'])}"
        if "delta" in details:
            result += f", изменение={_human_value(details['delta'])}"
        if "maximize" in details:
            result += ", направление=максимизация" if details["maximize"] else ", направление=минимизация"
        result += "; кандидат выбран" if details.get("selected") else "; кандидат не выбран"
        if details.get("reason"):
            result += f". Причина: {_human_value(details['reason'])}"
        chain = _model_chain(details.get("model_details"))
        if chain:
            result += f". Фактическая цепочка модели: {chain}"
        return result + "."
    if event == "research_finished":
        return f"Исследование завершено: {_human_value(details.get('reason', 'причина не указана'))}."
    if event == "research_stopped":
        return (f"Исследование остановлено: {_human_value(details.get('reason', 'причина не указана'))}; "
                f"токены={_human_value(details.get('total_tokens'))}, время={_human_value(details.get('elapsed_seconds'))} с.")
    if event == "submission_validated":
        return f"Проверка submission: {_human_value(details.get('validation'))}; файл={_human_value(details.get('submission_path'))}."
    if event == "trial_started":
        trial = details.get("trial_id", "trial")
        if details.get("is_baseline"):
            return f"Запущен базовый эксперимент {trial}: {_human_value(details.get('hypothesis'))}."
        return f"Запущен эксперимент {trial}: {_human_value(details.get('hypothesis'))}."
    if event == "trial_completed":
        result = f"Эксперимент {details.get('trial_id', 'trial')} завершён со статусом {details.get('status', 'unknown')}"
        if details.get("metric") is not None:
            result += f", метрика={_human_value(details['metric'])}"
        if details.get("seconds") is not None:
            result += f", время={_human_value(details['seconds'])} с"
        return result + "."
    if event == "candidate_selected":
        return f"Выбран лучший кандидат {details.get('trial_id', 'trial')}: метрика={_human_value(details.get('metric'))}."
    if event == "candidate_restored":
        return f"Восстановлен лучший кандидат {details.get('trial_id', 'trial')}: метрика={_human_value(details.get('metric'))}."
    if event == "fallback_written":
        return f"Записан резервный submission; стратегия={_human_value(details.get('strategy'))}."
    if event == "llm_request":
        return f"Отправлен запрос LLM {details.get('request_id', '')}: {_human_value(details.get('provider'))}/{_human_value(details.get('model'))}; содержимое не журналируется."
    if event == "llm_response":
        return f"Получен ответ LLM {details.get('request_id', '')}; списано токенов={_human_value(details.get('charged_tokens'))}."
    if event == "llm_error":
        return f"Ошибка запроса LLM {details.get('request_id', '')}: код={_human_value(details.get('error_code', 'unknown'))}."
    if event == "llm_protocol_error":
        return f"Ошибка формата ответа LLM {details.get('request_id', '')}."
    if event == "llm_response_recovered":
        return f"Восстановлен ожидающий ответ LLM {details.get('request_id', '')}."
    if event == "llm_response_consumed":
        return f"Ответ LLM {details.get('request_id', '')} обработан."
    if event == "context_compacted":
        return f"Сокращён контекст запроса: {_human_value(details.get('reason'))}."
    if event == "tool_call":
        return f"LLM вызвала инструмент {details.get('name', 'unknown')} ({details.get('call_id', '')}); аргументы опущены."
    if event == "tool_result":
        status = "успешно" if details.get("ok") else "с ошибкой"
        code = f", код={details['error_code']}" if details.get("error_code") else ""
        return f"Инструмент {details.get('name', 'unknown')} завершился {status}{code}."
    if event == "final_fit_started":
        return f"Начат финальный fit для {details.get('trial_id', 'лучшего кандидата')}."
    if event == "final_fit_completed":
        result = f"Финальный fit завершён: статус={details.get('status', 'unknown')}, время={_human_value(details.get('seconds'))} с"
        chain = _model_chain(details.get("model_details"))
        if chain:
            result += f"; фактическая цепочка модели: {chain}"
        if details.get("error_code"):
            result += f"; код ошибки={details['error_code']}"
        return result + "."
    if event == "run_completed":
        return (f"Запуск завершён: статус={details.get('status', 'unknown')}, остановка={details.get('stop_reason', 'unknown')}, "
                f"лучшая метрика={_human_value(details.get('best_metric'))}, выбран={_human_value(details.get('selected_trial'))}, "
                f"журнал решений={_human_value(details.get('decisions_path') or details.get('decision_log'))}.")
    if event == "run_interrupted":
        return "Запуск прерван пользователем или системой."
    if event == "run_error":
        return "Ошибка оркестрации; подробности доступны только в каноническом диагностическом журнале."
    return f"Событие {event}."


def _get_nested(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _model_chain(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    chain = value.get("estimator_chain")
    if isinstance(chain, list) and chain:
        return " → ".join(_bounded_text(item, 160) for item in chain[:12])
    model_class = value.get("class")
    return _bounded_text(model_class, 240) if model_class else ""


def _human_value(value: Any) -> str:
    if value is None:
        return "не указано"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return _bounded_text(value, 6000)


def _short_error_code(error: Any, status: Any) -> str | None:
    """Expose only a stable error class/code; raw diagnostics stay canonical."""
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return _bounded_text(error["code"], 100)
    if isinstance(error, str):
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]{0,59})\s*:", error)
        if match:
            return match.group(1)
    if isinstance(status, str) and status and status != "ok":
        return _bounded_text(status, 100)
    return None


def _write_projection(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
