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

## Red по замечаниям review

После добавления четырёх регрессионных сценариев, до исправления кода, та же команда завершилась так:

```text
...FF......FF                                                            [100%]
4 failed, 9 passed in 2.18s
exit_code=1
```

Падения подтвердили дефекты поведения: новый процесс не загружал `model.joblib` (`ModuleNotFoundError` для динамического `automl_candidate_<pid>`); `f1` на бинарном примере совпадал с macro F1 вместо F1 положительного класса; публикация превращала ID `001` в `1`; worker принимал train feature CSV без обязательной колонки `kind`.

## Green после исправлений review

Команда:

```text
/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest -q tests/test_data.py tests/test_worker.py tests/test_evaluation.py
```

Фактический вывод:

```text
.............                                                            [100%]
14 passed in 2.37s
exit_code=0
```

Регрессионный тест загружает `model.joblib` через `joblib.load` в новом Python-процессе и проверяет `predict`; бинарный F1 тест различает `f1=0` положительного класса и `macro_f1=5/11`. Для многоклассового `f1` evaluator теперь сообщает, что метрика поддержана только для бинарной задачи, и предлагает `macro_f1`.
