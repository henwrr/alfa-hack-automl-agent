# Независимое ревью decision journal

Проверен diff `f5feddc..82fe1cc` в изолированном worktree. Изменения кода не вносились. Полный suite, по отчёту интеграции, проходит: 86 passed.

## Findings

### [P1] API key попадает в snapshots и отчёты через гипотезу LLM

`ResearchRun.experiment` создаёт `ExperimentResult` с исходной `hypothesis`, затем получает redacted событие, но заменяет только `result.decision`; гипотеза остаётся исходной (`automl_agent/agent.py:260-266`). Позже результат сериализуется в `state.json` и `experiments.json` (`:150-159`), а также в `report.json` и `report.md` (`:494-509`).

Воспроизведено коротким offline test-run с fake provider: когда `hypothesis` и `decision.rationale` содержали значение `OPENAI_API_KEY`, секрет отсутствовал в `events.jsonl`, `agent.log` и `decisions.jsonl`, но присутствовал в `state.json`, `experiments.json`, `report.json` и `report.md`. Событие журнала возвращает redacted гипотезу, поэтому присваивание `result.hypothesis = recorded["payload"]["hypothesis"]` до последующих записей закрыло бы проверенный путь утечки. Сходную проверку стоит применить к другим свободным строкам результата перед записью snapshots.

### [P2] Стратегия объявляет `weighted_f1` метрикой минимизации

В `automl_agent/strategy.py:62` максимум выбирается по явному списку, где нет `weighted_f1`. Однако `data.py` разрешает эту метрику для классификации и задаёт `maximize=True` для всех метрик кроме `rmse`, `mae`, `log_loss`. Для задачи с `weighted_f1` сохранённый shortlist/`agent.log` поэтому утверждает обратное направление, чем фактическое правило выбора и `experiment_assessed`.

Проверено прямым вызовом `plan_research`: он возвращает `weighted_f1: минимум на одном split`.

### [P2] Resume не восстанавливает host assessment после сбоя между двумя событиями

`experiment` сохраняет `trial_completed` и следующим отдельным append сохраняет `experiment_assessed` (`automl_agent/agent.py:300-301`). При остановке процесса в этом промежутке результат уже восстановится из canonical journal, но `ResearchRun._recover_calls` восстанавливает только отсутствующий `trial_completed` (`:134-148`); он не проверяет и не дополняет отсутствующие assessments. В результате завершённый trial остаётся без обещанного проверяемого host-объяснения в `agent.log` и `decisions.jsonl` после `--resume`.

## Проверенные пункты без findings

- Публичный decision с настроенным API key redacts правильно в `events.jsonl` и двух audit-проекциях; сырой prompt/code отсутствует в `agent.log` и `decisions.jsonl`.
- Оценка host отдельно показывает измеренную метрику и фактическую estimator chain; baseline shortlist представлен как стартовая рекомендация, а fallback явно указывает на ExtraTrees и его причину.
- Вызов `tail -F` документирован для атомарно заменяемого `agent.log`.
- Повторный `--resume` восстанавливает повреждённые проекции из canonical events без дублирования строк.

## Исправления root
Все три findings покрыты regression-тестами в tests/test_decisions.py. Результаты и snapshots проходят canonical redaction, направление стратегии берётся из TaskSpec.maximize, а отсутствующий assessment восстанавливается с recovered=true и номером исходного completion. Проверка исправлений: 19 целевых тестов passed; полный повторный suite: 89 passed, exit 0 (27.35s).
