# Проверки

Итоговая команда `.venv/bin/python -m pytest -q --tb=short`: **65 passed in 19.38s, exit 0**. Transcript: logs/20261019T193139.575627Z-verify.json. Подробная приёмка: [../../VALIDATION.md](../../VALIDATION.md), [../../validation/2026-10-19.json](../../validation/2026-10-19.json).

| Критерий | Проверка | Результат |
|---|---|---|
| AC1–2 | config/data tests, acceptance, оба public runs | passed, без правок под public target |
| AC3–5 | fake OpenAI tools, live Ollama, host score/selection | passed; слабые кандидаты отвергнуты, ошибки кода исправлялись моделью |
| AC4 | runtime/provider tests, hung-fit integration | passed: timeout, потомки, output/RSS caps, API failures |
| AC6 | evaluation tests + повторный public validator | passed: колонки/строки/ID/классы/числа/атомарная публикация |
| AC7–8 | journal/agent tests + независимое review | passed: corruption, redaction, fingerprint, lock, usage, durable response, best artifact |
| AC9 | packaging tests и запуск распакованного ZIP с другой cwd | passed |
| AC10 | полная suite + два public runs + новая схема через Ollama | passed; границы API/Docker/soak описаны отдельно |

## Red → green

- Исходных тестов не было: unittest discovery exit 0, **0 tests**, logs/20261019T190623.330233Z-baseline.json. Это не доказательство работоспособности.
- Config/tools: 16 падений NotImplementedError → 16 passed; logs/20261019T191320.303246Z-red.json и 20261019T191414.036872Z-green.json.
- Agent loop: 2 падения NotImplementedError → agent/acceptance green; logs/20261019T191447.157191Z-red.json и 20261019T191817.667245Z-green.json.
- Concurrent resume: воспроизведена запись в чужой активный журнал → green после переноса lock до recovery; logs/20261019T192137.649773Z-red.json и 20261019T192159.612404Z-green.json.
- Crash review: 3 падения (повтор API, fallback вместо winner, tool effect после cap) → green; logs/20261019T192541.632105Z-red.json и 20261019T192639.542846Z-green.json. Независимая перепроверка подтвердила закрытие findings.
- ML/runtime workers сохранили собственные transcripts в docs/research/. Поздние regression fixes вошли в общую suite.
- Acceptance в исходном delivery worktree ожидал отсутствующий CLI: это ограничение интеграции, не поведенческий red. В сведённой копии настоящий acceptance прошёл.

## Реальные артефакты

- Регрессия: runs/ollama-public/regression/report.json; 22 449 строк; RMSE 46.061552300232094; 2 успешных LLM-кандидата; final fit ok; 373.674 s.
- Классификация: runs/ollama-public/classification/report.json; 41 166 строк; Macro F1 0.5650187807888305; 1 успешный LLM-кандидат после двух исключений; final fit ok; 617.321 s.
- Итоговая версия на новой схеме: runs/ollama-current/research/report.json; 30 строк; 1 успешный LLM-кандидат; 44.340 s.
- Все три model.joblib загружены отдельно от worker; predict совпал с CSV.

## Границы

Метрики локальные, не leaderboard. Qwen API и Docker не проверены. Трёхчасовой soak не проводился; failure paths проверены целевыми тестами. OS sandbox требует внешнего окружения. В ZIP нет train/test, ключей, .venv, тестов и research logs.
