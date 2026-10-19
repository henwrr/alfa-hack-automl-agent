# План

| Шаг | Зависимости | Файлы | Исполнитель / worktree | Проверка | Статус |
|---|---|---|---|---|---|
| Спецификация и baseline | — | docs/tasks/decision-journal | root | pytest baseline | выполнено |
| Проекции журнала и live вывод | спецификация | automl_agent/audit.py, tests/test_audit.py | Luna xhigh / decision-audit | red → green, секреты, resume | интегрировано 1aab4da |
| Метаданные реальной модели | спецификация | automl_agent/worker.py, tests/test_model_metadata.py | Luna xhigh / decision-model-metadata | red → green, wrapper/pipeline | интегрировано 361c694; уточнения root |
| Стратегия и объяснения LLM | спецификация | strategy.py, agent.py, contracts.py, prompts.py, tests/test_decisions.py | root | red → green интеграция | реализовано, финальные проверки |
| Интеграция и документация | все компоненты | README.md, docs/RESEARCH_STRATEGY.md, task docs | root | полный suite, Ollama, ZIP | запланировано |

## Базовый коммит
f5feddc. Baseline: 65 passed, 17.63s (exit 0).

## Контракты интеграции
AuditJournal(path, secrets=(), echo=True) наследует EventJournal. Основной агент заменяет только конструктор журнала. Производные agent.log и decisions.jsonl в той же директории. Записи с исходными seq/timestamp/event, source и details. Объяснения только структурированные и публичные. События: data_profiled(summary), validation_planned(summary), research_strategy(strategy), decision_recorded(trial_id, source, hypothesis, decision, explanation_provided), experiment_assessed(trial_id, metric_name, metric, previous_best, maximize, selected, delta, reason, model_details), research_finished(reason), research_stopped(reason, total_tokens, elapsed_seconds), submission_validated(validation, submission_path). Существующие события сохраняются.
Worker добавляет estimator_details в worker_result.json: {class: fqcn, estimator_chain: [fqcn,...], pipeline_steps: [{name, class},...]}. Глубина/число элементов ограничены, без параметров и содержимого данных. Необязательное поле; старые файлы валидны.
Decision schema: model_family(str), observations(list[str]), rationale(str), alternatives(list[str]), expected_effect(str), risks(list[str]), previous_result_assessment(str); всё необязательно для старых вызовов. ExperimentResult расширяется полями decision и model_details с пустыми defaults.

## Интеграция
Исполнители коммитят только свои файлы в отдельных worktree; root проверяет diff и переносит коммиты. Общие проверки запускаются после сведения.
