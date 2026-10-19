# Независимое ревью decision journal

Первичный review diff `f5feddc..82fe1cc` выполнен в отдельном worktree. Повторно проверен фиксирующий commit `edba428224183ff5c066e1eb2ee61aa584fc4bf4`; изменения кода не вносились.

## Исправленные findings

- **API key через гипотезу LLM — исправлено.** `experiment` теперь берёт гипотезу из redacted события, redacts `ExperimentResult` перед записью результата, snapshots и отчёта; `checkpoint`, `process.json`, `report.json` и `report.md` также используют политику `EventJournal.redact`. Добавленный тест проверяет отсутствие ключа в `events.jsonl`, audit-проекциях, state, experiments и обоих отчётах. Ранее это было воспроизведено fake provider.
- **Направление `weighted_f1` — исправлено.** `plan_research` берёт направление из `TaskSpec.maximize`, согласованного с host evaluator.
- **Потерянная оценка после сбоя — исправлено.** `_recover_assessments` восстанавливает отсутствующий host assessment из сохранённых событий в порядке завершения, добавляет `recovered` и `completion_seq`, не запускает обучение или LLM. Regression test удаляет assessment, запрещает повторный `experiment`, возобновляет run и проверяет однократное восстановление.

Проверка в этом worktree: `/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest tests/test_decisions.py -k 'key_echo or task_contract or recovers_assessment' -q --tb=short` — **3 passed, 11 deselected**. Root сообщил полный результат: **89 passed**.

## Остаточный finding

### [P2] Resume завершённого pre-fix run возвращает и сохраняет прежний незаредактированный отчёт

В `ResearchRun.execute` ранний путь для неактивного запуска с `report.json` возвращает `_read_json(report.json)` без `journal.redact`, перезаписи отчёта или checkpoint. `ResearchRun.__init__` redacts восстановленные `ExperimentResult` в памяти, но оставляет исходные `state.json` и отчёты на диске. Для completed run, созданного до `edba428` и содержащего ранее утёкший ключ, `--resume` поэтому всё ещё возвращает секрет и оставляет его в `state.json`, `report.json` и `report.md`.

Воспроизведено в этом worktree: создан завершённый offline run, в старые snapshots и отчёты добавлена строка-маркер, затем выполнен no-op `--resume`; маркер сохранился в возвращённом report и всех трёх файлах. Новые запуски на `edba428` проверенным тестом покрыты; остаток касается миграции/очистки уже существующих завершённых директорий.

## Другие проверенные пункты

- Public decision values редактируются при записи canonical events и при выводе в audit-проекции; raw prompts/code не попадают в `agent.log` и `decisions.jsonl`.
- В host assessment отображаются измеренная метрика и фактическая estimator chain; shortlist не объявляет победителя.
- `tail -F` указан для атомарно заменяемого `agent.log`.
- Исправление assessment recovery не повторяет завершённое обучение и не дублирует восстановленные записи при следующем resume.
