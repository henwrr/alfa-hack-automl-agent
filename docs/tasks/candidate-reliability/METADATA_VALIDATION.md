# R4: фактические параметры модели

Metadata дополняется только для точных типов CatBoostRegressor/Classifier,
ExtraTreesRegressor/Classifier, LogisticRegression и LinearRegression. Для
CatBoost источник — captured native `get_all_params()` после fit; если API не
отдаёт заданный `thread_count`, он читается из точного native `_init_params`.
`parameters_source` указывает этот источник. Для sklearn используется
whitelist полей конструкторов из `__dict__`, без `get_params()`. Для неизвестных
классов возвращается пустой `parameters` и источник `unsupported_estimator`.

CatBoost whitelist содержит iterations, learning_rate, depth, l2_leaf_reg,
border_count, auto_class_weights и class_weights. Значения auto learning rate и
рассчитанные веса после fit совпадают с native API. Явно настроенный
thread_count добавляется из `_init_params`, если API его не сообщает; значение
по умолчанию `None` не синтезируется. Число полей, длины строк и списков
ограничены; значения вне JSON scalar subset пропускаются. Из mapping
`class_weight` сохраняются числовые веса, а ключи классов (значения target)
опускаются. Отсутствующие поля native API не добавляются как `null`.
Вложенные и циклические containers отклоняются. Metadata extraction best-effort
и не вызывает пользовательские getters, `get_params()` или `repr()`.

## RED/GREEN protocol

- Baseline: `tests/test_model_metadata.py tests/test_worker.py` — 9 passed.
- RED: новая `tests/test_realized_parameters.py` на прежнем worker — 4 failed
  из-за отсутствующих `parameters` / `parameters_source` в fitted metadata и
  worker JSON. Ошибки импорта и зависимостей отсутствовали.
- GREEN: `tests/test_realized_parameters.py tests/test_model_metadata.py
  tests/test_worker.py` — 14 passed.
- Review RED: случаи с глубиной/cycle, LinearRegression, отсутствующими CatBoost
  полями и thread_count — 4 поведенческих отказа на предыдущей версии.
- Review GREEN: та же команда после исправлений — 17 passed.

Новые проверки включают fitted CatBoostClassifier с автоматическим learning
rate и `auto_class_weights="Balanced"`, ExtraTrees, LogisticRegression и
LinearRegression constructor parameters без вызова sklearn `get_params`,
hostile custom
getters/`repr`, а также subprocess worker JSON и joblib reload fitted CatBoost.
Актуальные коды выхода и сводки находятся в `logs/metadata-*.json`.
