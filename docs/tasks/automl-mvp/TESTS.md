# Проверки

| Критерий | Тест / команда | Baseline | Red: причина | Green |
|---|---|---|---|---|
| AC1–2 | tests/test_data.py + CLI acceptance | продуктового кода нет | исполняемые заглушки/отсутствующее поведение; не import failure | ожидается |
| AC3–5 | fake OpenAI tool calls → worker → host score | нет | recovery и выбор победителя | ожидается |
| AC4 | tests/test_runtime.py, test_providers.py | нет | зависание/output overflow/HTTP-ошибки | ожидается |
| AC6 | tests/test_evaluation.py | нет | неверные колонки/строки/ID/метки/NaN/inf | ожидается |
| AC7–8 | tests/test_journal.py, acceptance resume | нет | оборванный хвост/середина JSONL/secrets/changed input | ожидается |
| AC9 | tests/test_packaging.py, unpack ZIP smoke | нет | неполная упаковка | ожидается |
| AC10 | полная suite + реальные public runs Ollama | нет | не применяется к реальным прогонам | ожидается |

## Что проверяется вручную

Интеграционная проверка 19.10.2026: `.venv/bin/python -m pytest -q --tb=short` — 52 passed in 10.87s, exit 0. Отдельный цикл agent/acceptance: 3 passed in 3.34s, лог logs/20261019T191817.667245Z-green.json. Модельный протокол проверен реальным HTTP server: ошибочный Python → tool error → исправленный кандидат → finish, включая передачу парных tool-result в следующий model request. Unit suite не обращается к Ollama.

Ведущий обнаружил при review и отправил на исправление: отдельная загрузка динамического сохранённого estimator, binary F1 и сохранение ID с ведущими нулями. Финальный общий прогон будет повторён после этих изменений.

Полный размер и точный порядок public submission, реальное участие LLM, список гипотез/метрик/кода, отсутствие секретов в архиве, фактический git diff и чистота ветки. Визуальная проверка PDF страниц 18–19 выполнена для формул нормировки.

## Ограничения окружения

Исходный baseline discovery: `python3 -m unittest discover -v`, exit 0, **0 тестов**; это отсутствие существующего покрытия, а не доказательство работоспособности MVP. Лог сохранён в logs/20261019T190623.330233Z-baseline.json.

Python 3.12.13 устанавливается через uv в локальную .venv. Qwen API отсутствует; Ollama 0.40.4 доступна. Native qwen3.5:9b JSON smoke выполнен: 6.71 сек, 57 input/56 output tokens, корректный JSON. Полноценный исследовательский loop этим smoke ещё не проверен.

Автоматические протоколы находятся в logs/. Код выхода сам по себе не доказывает выполнение критерия.
