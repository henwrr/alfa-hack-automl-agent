# Проверки

| Критерий | Команда / сценарий | Baseline | Red | Green |
|---|---|---|---|---|
| Старые | .venv/bin/python -m pytest -q --tb=short | 90 passed | — | 160 passed |
| P1 | tests/test_families.py | — | см. logs и validation | 14 passed |
| P2,P3,P6 | tests/test_scheduler.py | — | 10 failures | 11 passed |
| P4,P5 | tests/test_budgeting.py | — | см. logs | 34 passed |
| P1–P7 | tests/test_parallel_agent.py | — | missing batch/default/tie | 11 passed |
| P8 | Live Ollama на обоих публичных CSV | — | — | CSV/refit/reload/overlap/resume ok |
| P9 | packaging, review, remote SHA | — | — | review: no P1/P2; ZIP22files; remote SHA — в финальном ответе |

Доказательства: статусы, измеренные интервалы процессов, n_jobs, fingerprint split, CSV validator, модель reload/predict. API организаторов и Docker остаются вне доступного окружения. Логи команд в logs/.

## Интеграция
Первый red: реальные failures нового batch tool, tie selection и defaultconfig; два offline-fixture сначала требовали исправления настройки до проверки поведения. Затем интеграция выявила отсутствие model_family в декларативном decision генератора; добавлены публичные объяснения с явным семейством. Старые тесты одиночного baseline явно используют --family-trials1. Для ограниченного token budget введён компактный systemprompt с сохранением contracts и последней tool-группы. Итог: 160 passed, 1 ConvergenceWarning на маленьком линейном fixture, 44.43s, exit0.

## Public validation
См. docs/VALIDATION.md и docs/validation/parallel-public-2026-10-19.json. Скрипт verify_public_artifacts.py воспроизводит проверки сохранённых артефактов; исходные CSV и runs доступны локально. Регрессия degraded из-за ошибок четырёх LLM-trials, но оба итоговых CSV и refit валидны. Runtime не менялся после полного suite.
