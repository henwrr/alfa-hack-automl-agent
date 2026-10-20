# Независимый review CV-интеграции

Read-only review текущих файлов в main checkout
`/Users/egoraprelsky/Desktop/coding/alfa_hackathon`: `validation.py`,
`cv_worker.py`, `prediction.py`, `agent.py`, `config.py`, `budgeting.py`,
`worker.py`, а также `training.py` и соответствующие focused tests. Main checkout
не редактировал. Публичные данные, Ollama и полный suite не запускал.

В обычном CV-пути утечки outer validation labels не обнаружено. `prepare_cv_split`
строит план и `validation_target` на host стороне; manifest CV передаёт только
train target, train/test features и план индексов. Каждый fold вызывает
`fit_model` только с `X.iloc[train]` и `y[train]`; `fit_model` не принимает
outer-validation target. Для future-only time features `make_validation_plan`
использует только forward groups; `_inner_time_split` отделяет последний period
из локального fold-train для early stopping. Внутренние и внешние индексы
проверяются на пересечение. Самая ранняя временная группа явно остаётся без OOF,
и coverage записывается в результат.

OOF порядок в текущей реализации согласован: host сортирует `validation_indices`
и строит `validation_target` из `train.iloc[eligible]`; worker записывает
`oof[eligible]` в том же порядке. `oof_fold_ids` использует те же индексы.
Вероятности каждого fold выравниваются по `model.classes_` в глобальный
`TaskSpec.classes` axis; отсутствующие в fold классы получают нулевую колонку.
Host scorer затем переставляет колонки по своим task labels для ROC AUC/log loss.

Во время review были найдены и затем исправлены root:

- Final worker раньше игнорировал переданные `fit_options` и
  `iteration_override`; теперь `worker._run_manifest` использует
  `training.fit_model`, когда есть fit options или override. Tiny CatBoost
  finalize check прошёл: 7 фактических деревьев, `balanced` веса, три test
  predictions и сериализованная модель.
- `fit_model` раньше безусловно требовал `sklearn.clone`, хотя старый CLI
  принимал любой custom estimator с `.fit/.predict`. Теперь non-native clone
  failure оставляет свежий объект, полученный из fold builder. Отдельный tiny
  `PlainEstimator` check прошёл, включая игнорирование `iteration_override`.
- Reserve сначала рассчитывался от полного CV process seconds. Теперь
  `plan_finalization` предпочитает `training.estimated_refit_seconds`, а legacy
  path сохраняет `best.seconds`. Tiny policy check дал 30 секунд при CV wall
  time 900 секунд и fold refit estimate 20 секунд с safety factor 1.5.

Остаются две рекомендации для независимой валидации artifacts. Сейчас
`worker_result.json` содержит `split_fingerprint`, а `oof_indices.npy` содержит
фактические индексы, но host scorer проверяет только длину
`valid_predictions.csv` относительно `validation_target`. Текущий writer и plan
совпадают по порядку, но host-side проверка fingerprint/индексов защитит от
same-length reorder или accidental stale/mismatched plan. Также
`aligned_probabilities` отклоняет не конечные/отрицательные значения, но
принимает значения выше 1 и ненормированные строки, после чего нормализует их.
Для `predict_proba` было бы безопаснее проверять диапазон и сумму строк с
небольшим допуском до выравнивания. Оба пункта — robustness gaps; на текущих
native fixtures неверных axes или значений не наблюдал.

Resume settings fingerprint теперь включает CV folds/strategy/time column для
state версии 2; state версии 1 остаётся на legacy holdout fingerprint. Config
восстанавливает сохранённые validation options, если пользователь не переопределил
их. Default CLI не ограничен шестью trials: `max_trials` и `max_steps` по умолчанию
`None`, запуск ограничивают время и token budget; warm-up сравнивает четыре
семейства.

Проверки на main checkout с ограничением BLAS/OpenMP до одного потока:

- `pytest tests/test_cv_worker.py tests/test_validation_plan.py tests/test_config.py -q` — **18 passed**.
- `pytest tests/test_worker.py tests/test_agent.py -q` — **11 passed**.
- `pytest tests/test_training.py tests/test_budgeting.py -q` — **52 passed**.
- Отдельные маленькие проверки custom `.fit/.predict`, final CatBoost
  override/weights/serialization и выбора CV refit estimate прошли.

Новый `tests/test_research_cv_agent.py` прочитан, но не запускался параллельно с
root review; он покрывает два recipe edits, full-CV OOF и selected-config refit.

Root: оба дополнительных замечания закрыты. Host проверяет fingerprint, OOF indices и fold IDs; probability alignment отвергает значения вне [0,1], неединичную сумму и неверную ось классов. Добавлены поведенческие RED/GREEN тесты.
