# Валидация метаданных обученной модели

Проверка сделана в worktree `decision-model-metadata` интерпретатором
`/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python`.

## Red → green

Сначала добавлены четыре сквозных теста. На исходном worker все четыре успешно
обучили модель, записали прогнозы и дошли до чтения `worker_result.json`, затем
упали с `KeyError: 'estimator_details'`. Это подтверждает отсутствие требуемого
поведения, а не проблему импорта или запуска.

После реализации:

```text
$ /Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest -q tests/test_model_metadata.py --tb=short
.....                                                                    [100%]
5 passed in 4.57s

$ /Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest -q tests/test_worker.py --tb=short
....                                                                     [100%]
4 passed in 3.05s
```

Тесты проверяют backend CatBoost внутри `_AdaptiveModel`, реальный конечный
предиктор и шаги sklearn Pipeline в режимах `evaluate` и `finalize`, wrapper с
падающими свойствами и циклической ссылкой, сохранение модели, а также лимиты
глубины, числа estimator-объектов и шагов pipeline.

Подробные red/green результаты сохранены в
`docs/tasks/decision-journal/logs/20261019T200615Z-decision-model-metadata.log`.

## Ограничения

Метаданные отражают только известные связи wrapper-ов и sklearn Pipeline. Поиск
останавливается после 6 уровней, 32 объектов и 32 шагов; списки или словари
wrapper-ов ограничены 16 дочерними объектами. Необычные wrapper-ы, которые
хранят fitted estimator в иных полях, могут отобразить только собственный
класс. В metadata не включаются параметры, repr, имена колонок, признаки,
наблюдения или значения целевой переменной. Ошибки при обходе дают частичный
результат и не прерывают обучение, предсказание или запись результата.
