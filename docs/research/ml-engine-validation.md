# ML engine validation transcript

Все команды выполнялись в выделенном worktree с Python 3.12 из общей `.venv`; тестовые таблицы генерируются самими тестами.

## Red до реализации

Команда:

```text
/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest -q tests/test_data.py tests/test_worker.py tests/test_evaluation.py
```

Фактический результат до реализации интерфейсов:

```text
FFFFFF                                                                   [100%]
6 failed in 12.25s
exit_code=1
```

Причина всех шести падений — `NotImplementedError` в импортируемых функциях `discover_task`, `baseline_code`, `score_trial` и `publish_submission`. Тестовые модули импортировались успешно, зависимости были доступны; это red по отсутствующему поведению, а не ошибка окружения.

## Green после реализации

Команда после реализации и исправлений:

```text
/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest -q tests/test_data.py tests/test_worker.py tests/test_evaluation.py
```

Фактический вывод итогового запуска:

```text
..........                                                               [100%]
10 passed in 1.62s
exit_code=0
```

Также выполнена read-only проверка discovery на локальных описаниях и исходных CSV основной копии (эти файлы не используются тестами):

```text
Регрессия regression rmse site_eui 62 ['id'] ['State_Factor', 'building_class', 'facility_type'] ['id', 'site_eui']
Многоклассовая классификация classification macro_f1 attack_cat 42 ['id'] ['proto', 'service', 'state'] ['id', 'attack_cat']
exit_code=0
```
