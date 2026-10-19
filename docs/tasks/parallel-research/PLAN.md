# План

| Шаг | Файлы | Исполнитель | Проверка | Статус |
|---|---|---|---|---|
| Baseline и контракты | task docs | root | pytest baseline | 90 passed |
| Генераторы разных семейств | families.py, test_families.py | Luna xhigh / parallel-families | red/green, реальные fit/predict | запланировано |
| Параллельные процессы и отмена | scheduler.py, runtime.py, test_scheduler.py | Luna xhigh / parallel-scheduler | overlap, caps, cancellation, deadline | запланировано |
| Адаптивный reserve | budgeting.py, test_budgeting.py | Luna xhigh / parallel-budget | red/green, sizing, resume pure plan | запланировано |
| Интеграция host | agent.py, config.py, contracts.py, prompts.py, audit.py, tests/test_parallel_agent.py | root | red/green, LLM batch/recovery | запланировано |
| Проверка/доставка | README, strategy docs, validation evidence, ZIP | root + review | все tests, оба public, push | запланировано |

## Базовый коммит
 dedbb05. Рабочая копия чистая; предыдущие 10 локальных коммитов ещё не push. Пользователь теперь явно разрешил push всей aprelskiy_branch.

## Контракты
См. CONTRACTS.md. Координатор — единственный писатель событий, state и submission. Потоки запускают только subprocess; worker никогда не выбирает победителя.
