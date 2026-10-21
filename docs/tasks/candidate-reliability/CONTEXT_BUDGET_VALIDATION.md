# Проверка лимита recipe-контекста

## Что изменено

`build_messages` для recipe mode собирает отдельный компактный payload и поэтапно сокращает его до лимита `max(12000, context_chars // 2)` при `compact=True` (иначе до `max(12000, context_chars)`). В compact режиме сохраняются задача и ограничения, точные target/class counts, план и покрытие валидации, temporal/adversarial diagnostics, лучшая проверенная recipe с фактическими параметрами и trial ID, а также короткие полезные per-class/segment/feature-importance сведения для лучшей и последней попыток. История старых trials оставляет краткие hypothesis/status/metric. Семейное сравнение исключает подробные fold timings. Ошибка сохраняется только для последней неуспешной попытки. Входные объекты не мутируются; code mode идёт прежним путём.

Каталог остаётся основан на `recipe_catalog`: сохранены поддерживаемые диапазоны, enum options и defaults, удалены пояснения про host execution и запрещённые поля, а явно недоступные семейства исключаются. Для измеренных семейств добавляются фактические параметры; выбранная recipe и параметры также всегда доступны в `best_experiment`. Prompt теперь требует продолжать измеряемые эксперименты до временного резерва и останавливаться, когда резерв достигнут или нового эксперимента уже не помещается.

## Проверки

Сначала новые тесты дали RED: обе синтетические сборки выходили за лимит (184,675 символов при 48,000 и 181,079 при целевых 12,000), а отдельный `latest_experiment` отсутствовал. После реализации:

- `tests/test_context_budget.py` и `tests/test_recipe_agent.py`: **15 passed**.
- Синтетический контекст с 30 trial, 200 features и 200 диагностическими строками: 26,530/48,000 символов в обычном режиме и 11,676/12,000 в compact-режиме с более жёсткой тестовой целью. Тесты проверяют точные class counts, план/coverage, adversarial ROC AUC, лучшую recipe и native parameters, per-class/feature-importance diagnostics, последнюю ошибку, отсутствие устаревшей ошибки после успеха, отсутствие fold timings, отсутствие мутации и неизменность code mode.
- Сохранённый live regression профиль `runs/research-cv-public-20261020/regression` обработан локальным dry-run без LLM и fit. Для `context_chars=48,000`: baseline из `ec9a23c` — 46,205 символов (`compact=False`) и 42,321 (`compact=True`); новая сборка — 26,794 и 21,364 соответственно. В compact-варианте остались `trial_0006`, его ExtraTrees recipe и фактические параметры, forward-group validation plan и best diagnostics.

Live dry-run использовал сохранённые `task.json`, `profile.json`, `experiments.json` и `state.json`; файлы не изменялись. Новые публичные fits и Ollama не запускались.

Root review: матрица ошибок, все10 per-class строк, actual numeric/category drift и nested without_temporal AUC теперь сохраняются. История ограничена последними12+best trials; при переполнении удаляются вторичные сравнения, затем defaults каталога. Exact best recipe/native params, target counts и validation plan остаются; патологически большой обязательный payload даёт явную ошибку. Дополнительные tests real schema +1000 trials:18 passed вместе с recipe_agent. Интеграция с bounded postprocessing:41 passed (16.03s).
