# Независимое ревью decision journal

Первичный review diff `f5feddc..82fe1cc` выполнен в отдельном worktree. Повторно проверены фиксы `edba428224183ff5c066e1eb2ee61aa584fc4bf4` и `b98827d478bb505c72d87b1b5a6e093eaf16dac0`; изменений кода не вносилось.

## Findings и исправления

- **API key через гипотезу LLM — исправлено в `edba428`.** `experiment` берёт гипотезу из redacted события и redacts результат перед записью результата, snapshots и отчётов; checkpoint и persisted process/report файлы используют политику `EventJournal.redact`.
- **Направление `weighted_f1` — исправлено в `edba428`.** `plan_research` берёт его из `TaskSpec.maximize`, согласованного с host evaluator.
- **Потерянная оценка после сбоя — исправлено в `edba428`.** `_recover_assessments` восстанавливает отсутствующий host assessment из событий, добавляет `recovered` и `completion_seq`, не запускает обучение или LLM.
- **Старые незаредактированные reports при completed `--resume` — исправлено в `b98827d`.** Завершённый no-op resume валидирует submission, redacts и атомарно переписывает существующие `state.json`, `experiments.json`, `report.json`, очищает `report.md`, добавляет отсутствующие пути к audit-проекциям и возвращает очищенный report. Добавленный regression test симулирует старые файлы с ключом и запрещает повторное обучение.

Проверка в этом worktree: `/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest tests/test_decisions.py -k 'completed_legacy or key_echo or task_contract or recovers_assessment' -q --tb=short` — **4 passed, 11 deselected**. Root сообщил полный результат: **90 passed**.

## Другие проверенные пункты

- Public decision values редактируются в canonical events и audit-проекциях; raw prompts/code не попадают в `agent.log` и `decisions.jsonl`.
- Host assessment показывает измеренную метрику и фактическую estimator chain; shortlist не объявляет победителя.
- `tail -F` указан для атомарно заменяемого `agent.log`.
- Assessment recovery не повторяет завершённое обучение и не дублирует восстановленные записи при следующем resume.

После проверки `b98827d` открытых findings P1/P2 не осталось.
