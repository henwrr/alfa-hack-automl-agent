# Проверки

| Критерий | Команда / сценарий | Baseline | Red | Green |
|---|---|---|---|---|
| Старые | .venv/bin/python -m pytest -q --tb=short | 90 passed | — | ожидается |
| P1 | tests/test_families.py | — | ожидается | ожидается |
| P2,P3,P6 | tests/test_scheduler.py | — | ожидается | ожидается |
| P4,P5 | tests/test_budgeting.py | — | ожидается | ожидается |
| P1–P7 | tests/test_parallel_agent.py | — | ожидается | ожидается |
| P8 | Live Ollama на обоих публичных CSV | — | — | ожидается |
| P9 | packaging, review, remote SHA | — | — | ожидается |

Доказательства: статусы, измеренные интервалы процессов, n_jobs, fingerprint split, CSV validator, модель reload/predict. API организаторов и Docker остаются вне доступного окружения. Логи команд в logs/.
