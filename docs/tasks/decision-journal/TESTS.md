# Проверки

| Критерий | Тест / команда | Baseline | Red: причина | Green |
|---|---|---|---|---|
| Все прежние | .venv/bin/python -m pytest -q --tb=short | 65 passed, exit 0 | — | ожидается |
| D1,D6 | tests/test_audit.py | — | ожидается | ожидается |
| D2,D3,D4,D7 | tests/test_decisions.py | — | ожидается | ожидается |
| D5 | tests/test_model_metadata.py | — | ожидается | ожидается |
| D8 | живой Ollama + package_solution.py | — | — | ожидается |

## Что проверяется вручную
Читаемость журнала, отсутствие raw prompts и code в проекциях, связь гипотезы и фактической модели, достоверность вывода host, resume без дублей.

## Ограничения окружения
Ollama qwen3.5:9b доступна локально. API организаторов и Docker недоступны. Протоколы команд в logs/.
