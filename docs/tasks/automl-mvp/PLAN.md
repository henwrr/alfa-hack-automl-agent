# План

| Шаг | Зависимости | Файлы | Исполнитель / worktree | Проверка | Статус |
|---|---|---|---|---|---|
| 1. Кейс и контракт | исходные материалы | CASE_SUMMARY.md | case_research / automl-case-research | все страницы/формулы/CSV сверены | завершено, проверено ведущим |
| 2. Архитектура Harness | upstream master pin | DEEPSEEK_HARNESS_SUMMARY.md | harness_architecture / automl-harness-architecture | ссылки/ключевые исходники | завершено, проверено ведущим |
| 3. Надёжность Harness | upstream master pin | docs/research/DEEPSEEK_HARNESS_RELIABILITY.md | harness_reliability / automl-harness-reliability | ссылки/семантика retries/timeout/checkpoint | завершено, проверено ведущим |
| 4. Требования и интерфейсы | 1–3, ответы пользователя | SPEC.md, CONTRACTS.md, contracts.py | ведущий | сверка AC, отсутствие dataset-specific contracts | готово |
| 5. Данные/worker/evaluator | 4 | data.py, modeling.py, worker.py, evaluation.py; собственные тесты | ml_engine / отдельный worktree | red→green unit/integration tests на новых схемах/категориях/редких классах/NaN/инвалидах CSV | запланировано |
| 6. Runtime/providers/journal | 4, параллельно 5 | runtime.py, providers.py, journal.py; собственные тесты | harness_runtime / отдельный worktree | red→green: transient/permanent HTTP, timeout/потомки/output cap, corruption/recovery/secrets | запланировано |
| 7. Упаковка и acceptance | 4, параллельно 5–6 | scripts, requirements-mvp/dev, README, Dockerfile, .env.example, tests/test_packaging.py, tests/test_acceptance.py | delivery / отдельный worktree | unpack ZIP+CLI, fake OpenAI tools/recovery без Ollama | запланировано |
| 8. Agent loop и CLI | 4, интеграция 5–7 | solution.py, config.py, prompts.py, tools.py, agent.py | ведущий | ранний CSV, LLM-решения/код, host scoring, budgets, checkpoint/resume/report | запланировано |
| 9. Интеграция и ревью | 5–8 | общая ветка, тесты и исправления | ведущий | полный pytest, diff --check, запуск из другой cwd/ZIP | запланировано |
| 10. Реальные публичные прогоны | 9, Ollama | runs/ артефакты, TESTS.md | ведущий | оба полных CSV, >=1 LLM-кандидат, валидные submission, записанные метрики | запланировано |
| 11. Завершение | 10 | README, STATE/TESTS, dist/solution.zip | ведущий | чистая aprelskiy_branch; архив без секретов/данных, инструкция запуска | запланировано |

## Базовый коммит

Исходный HEAD e520b9e, ветка aprelskiy_branch. 6157aca фиксирует исходные baseline/requirements, описания/словари данных, согласованные требования и baseline test transcript (исходных тестов 0). Датасеты и PDF сохраняются локально и исключены из Git/ZIP. Исследовательские commits интегрированы cherry-pick после чтения файлов и выборочной сверки upstream-кода.

## Интеграция

Общие dataclasses и точные Python/file contracts — в CONTRACTS.md. Ведущий владеет contracts/agent/CLI и task docs; исполнители не меняют эти файлы. Работники создают тесты поведения до кода, проверяют реальное падение из-за отсутствующего поведения, затем реализуют; возвращают коммиты и transcript. Ошибка импорта/нет зависимости не принимается за red. Перенос только их файлов, после review и повторной проверки в сведённой копии. Ни один работник не пишет в основную копию.

В обычном режиме агент вызывает LLM через нормализованный tools protocol. Сначала надёжный baseline и атомарный submission, затем LLM-кандидаты с build_model(context); host вычисляет метрику, фиксирует победителя и публикует только прошедшие validation предсказания. Финальный fit использует весь train и сохраняет рабочий ранний ответ при неудаче. Установки/скачивания во время работы нет.

Один общий monotonic wall deadline включает API retries, cleanup и final fit; финальная доля бюджета резервируется. По умолчанию один training child, <=8 потоков. Journal сохраняет точный request и call intent до эффекта, результат после, не пишет ключи. Числовые метрики и provenance остаются в structured ledger независимо от сокращения model context. Возобновление проверяет fingerprint входов и конфигурацию split; незавершённые calls получают явный interrupted outcome.

Qwen API и Docker проверяются при фактической доступности. Пока acceptance использует локальный fake OpenAI server и реальные Ollama прогоны. Ограничение по скрытым задачам: MVP поддерживает supervised tabular CSV; расширение на неизвестные форматы требует новых evaluator/input contracts, поскольку материалов по ним нет.
