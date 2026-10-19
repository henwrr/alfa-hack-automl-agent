# Проверки

| Критерий | Тест / команда | Baseline | Red: причина | Green |
|---|---|---|---|---|
| Все прежние | .venv/bin/python -m pytest -q --tb=short | 65 passed, exit 0 | — | 90 passed, exit 0, 26.39s |
| D1,D6 | tests/test_audit.py | — | 5 genuine failures, см. AUDIT_VALIDATION.md | 5 passed + integration |
| D2,D3,D4,D7 | tests/test_decisions.py | — | 9 failed: нет проекций; decision отклонён; старый trial не содержит поля; стратегия и оценка ещё не реализованы (exit 1, без import errors) | 15 passed в итоговом suite |
| D5 | tests/test_model_metadata.py | — | отсутствие metadata после успешного fit | 5 passed; см. METADATA_VALIDATION.md |
| D8 | живой Ollama + package_solution.py | — | — | completed, 30 строк, Macro F1 0.83625 → 0.86741; модель загружена; resume без новых событий; ZIP 19 файлов |

## Что проверяется вручную
Читаемость журнала, отсутствие raw prompts и code в проекциях, связь гипотезы и фактической модели, достоверность вывода host, resume без дублей.

## Ограничения окружения
Ollama qwen3.5:9b доступна локально. API организаторов и Docker недоступны. Протоколы команд в logs/.

## Найденное при интеграции
Первый полный green-прогон: 1 failed / 84 passed — при compaction пропадала последняя assistant/tool группа, fake API не получал ошибку для исправления. Исправлено реальным сокращением auxiliary context с сохранением последней группы. Второй прогон выявил запрет AST на имя класса исключения в baseline и необходимость завершать metadata на ансамбле, а не отдельных деревьях. Baseline теперь сохраняет текст причины без запрещённой интроспекции; metadata предпочитает fitted-ссылки и не раскрывает отдельные деревья. Итоговый полный прогон: 86 passed, exit 0.

## Исправления независимого ревью
Три новых regression-теста сначала упали по указанным причинам (exit 1): секрет в гипотезе попадал в отчёт; weighted_f1 объявлялся минимизируемым; отсутствующая host-оценка не восстанавливалась. Исправлены единая redaction для результатов/snapshots/отчётов, направление из TaskSpec.maximize и детерминированное восстановление assessment из canonical результата без нового обучения. 19 целевых тестов прошли, exit 0.

Финальный regression для старых завершённых запусков тоже сначала упал (секрет оставался в ранее записанных snapshot/report), затем исправлен completed-resume fast path. Полный итог: 90 passed, 26.39s, exit 0. Независимая перепроверка четырёх исправлений: 4 passed, 11 deselected; открытых P1/P2 из review нет.
