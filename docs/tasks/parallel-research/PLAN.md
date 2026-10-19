# План

| Шаг | Файлы | Исполнитель | Проверка | Статус |
|---|---|---|---|---|
| Baseline и контракты | task docs | root | pytest baseline | 90 passed |
| Генераторы разных семейств | families.py, test_families.py | Luna xhigh / parallel-families | red/green, реальные fit/predict | готово |
| Параллельные процессы и отмена | scheduler.py, runtime.py, test_scheduler.py | Luna xhigh / parallel-scheduler | overlap, caps, cancellation, deadline | готово |
| Адаптивный reserve | budgeting.py, test_budgeting.py | Luna xhigh / parallel-budget | red/green, sizing, resume pure plan | готово |
| Интеграция host | agent.py, config.py, contracts.py, prompts.py, audit.py, tests/test_parallel_agent.py | root | red/green, LLM batch/recovery | готово |
| Проверка/доставка | README, strategy docs, validation evidence, ZIP | root + review | все tests, оба public, push | готово; remote SHA — в финальном ответе |

## Базовый коммит
 dedbb05. Рабочая копия чистая; базовый коммит уже находится в origin/aprelskiy_branch. Пользователь явно разрешил push новых изменений всей aprelskiy_branch.

## Контракты
См. CONTRACTS.md. Координатор — единственный писатель событий, state и submission. Потоки запускают только subprocess; worker никогда не выбирает победителя.
