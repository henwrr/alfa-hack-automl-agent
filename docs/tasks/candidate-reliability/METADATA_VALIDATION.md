# R4: фактические параметры модели

Metadata дополняется только для точных типов CatBoostRegressor/Classifier,
ExtraTreesRegressor/Classifier и LogisticRegression. Для CatBoost источник —
captured native `get_all_params()` после fit; для sklearn — whitelist полей
конструкторов из `__dict__`, без `get_params()`. Для неизвестных классов
возвращается пустой `parameters` и источник `unsupported_estimator`.

CatBoost whitelist содержит iterations, learning_rate, depth, l2_leaf_reg,
border_count, auto_class_weights и class_weights. Значения auto learning rate и
рассчитанные веса после fit совпадают с native API. Число полей, длины строк и
списков ограничены; значения вне JSON scalar subset пропускаются. Из mapping
`class_weight` сохраняются числовые веса, а ключи классов (значения target)
опускаются. Metadata extraction best-effort и не вызывает пользовательские
getters, `get_params()` или `repr()`.

## RED/GREEN protocol

- Baseline: `tests/test_model_metadata.py tests/test_worker.py` — 9 passed.
- RED: новая `tests/test_realized_parameters.py` на прежнем worker — 4 failed
  из-за отсутствующих `parameters` / `parameters_source` в fitted metadata и
  worker JSON. Ошибки импорта и зависимостей отсутствовали.
- GREEN: `tests/test_realized_parameters.py tests/test_model_metadata.py
  tests/test_worker.py` — 14 passed.

Новые проверки включают fitted CatBoostClassifier с автоматическим learning
rate и `auto_class_weights="Balanced"`, ExtraTrees и LogisticRegression
constructor parameters без вызова sklearn `get_params`, hostile custom
getters/`repr`, а также subprocess worker JSON и joblib reload fitted CatBoost.
Актуальные коды выхода и сводки находятся в `logs/metadata-*.json`.
