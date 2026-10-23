# Проверка универсального агента

Исходные прогоны: 21.10.2026; финальная проверка и исправления: 23.10.2026. Python 3.12.13, macOS arm64. Все 13 runtime-пакетов установлены ровно в версиях из requirements-mvp.txt; этот список побайтно по версиям соответствует подмножеству исходного requirements.txt организаторов. Ключи API не использовались.

## Автоматические проверки

```bash
.venv/bin/python -m pytest -q
.venv/bin/python universal_agent/scripts/package_solution.py
.venv/bin/python -m pytest -q universal_agent/tests/test_packaging.py
```

Исходный baseline: 378 passed. Целевые проверки последних исправлений: 49 passed, exit0. Упаковочные проверки: 3 passed, exit0; включая извлечение настоящего ZIP и запуск из постороннего рабочего каталога. **Полный финальный прогон: 416 passed, 2 warnings, exit0 за89.23с.** Два ConvergenceWarning относятся к намеренно ограниченной числом итераций sparse Linear-модели; они не пропущенные тесты и не ошибки.

Покрытие: определение target/metric/IDs, утечки и OOF alignment, temporal/group folds, fit-only feature statistics и vocabulary, malformed/nonnumeric DSL, сериализация, weights/early stopping, реальные backend/parameters, resource/deadline limits, ошибки LLM и recovery, resume, class axis/positive class, binary/multiclass probabilities, мета-постобработка, атомарный CSV, упаковка.

При финальной проверке дополнительно воспроизведены и исправлены: пропуск класса в probability artifacts; coercion смешанных меток; поздняя ошибка конфликтующей sample schema; неверная ось текстовых классов в meta LogLoss; binary F1 без учёта явного positive class; отсутствие train_rows у holdout refit и потеря источника thread_count при слиянии metadata. Тесты сначала падали на конкретном отсутствующем поведении, затем прошли после исправления. Существующие assertions не ослаблялись. Один старый parser fixture изменён на бинарную цель для корректной single-column LogLoss-схемы; отдельный тест требует отклонять неподдержанную multiclass label-only схему.

## Полные публичные данные с реальной Ollama

Модель qwen3.5:9b через native Ollama. Исходные task.md и все строки train/test использованы во временных копиях; прежние submission пользователя не изменены. Имеющиеся submission не использовались как обучающие labels. Данные и модели не входят в Git.

| Проверка | Протокол | Измеренный результат | Итог |
|---|---|---|---|
| Регрессия | 3 forward folds; budget300с, trial100с; 4 trials; 2 LLM calls/2 measured LLM trials | OOF RMSE 50.3250519513 | completed; 102.031с; full refit ok; 22449 строк |
| Классификация, исследовательский прогон | 3 requested group folds; budget420с, trial60с; 5 trials, из них 2 LLM; 4 LLM calls | OOF Macro F1 0.6251988417 | completed; 345.973с; full refit ok; 41166 строк |

Регрессия использует наблюдаемое future-only распределение Year_Factor; его имя в runtime не зашито. Классификация группирует одинаковые строки признаков; этот протокол строже прежней случайной CV и не сопоставим напрямую с числами из README исходных агентов.

В классификации CatBoost и ExtraTrees достигли заданного короткого trial timeout. LightGBM сохранил рабочий checkpoint, а LLM-гипотеза sqrt_balanced повысила Macro F1 с 0.5525937088 до 0.6251988417. Параллельное назначение в публичных прогонах не превышало 4 CPU (2×2). Финальные модели обучились успешно.

Постобработка в последнем классификационном прогоне отклонена: meta assessment0.6198765267 хуже0.6251988417, хотя calibration score0.6314748485 выглядел лучше. Сохранена одиночная модель. Это показывает проверку гипотезы, а не гарантирует качество скрытого теста.

Первый короткий классификационный прогон с подробным decision prompt завершился degraded: LLM возвращала ошибочные/повторные изменения, модель и валидный CSV сохранились. После упрощения prompt повторный запуск дал два реально обученных LLM-кандидата. Все форматные ошибки сохранены в локальном журнале; их не выдают за успешные эксперименты.

Независимый verify_run на регрессии и итоговой классификации: exit0. Повторно вычислены метрики всех успешных trials, проверены source checksums, фактические weights/iterations/ресурсы, модель перезагружена и её прогноз совпал с submission. Компактные результаты без данных, ключей и абсолютных временных путей: [VALIDATION_RESULTS.json](VALIDATION_RESULTS.json).

## Вероятности и ZIP

Дополнительные end-to-end offline задачи: binary ROC-AUC с явной положительной строковой меткой и ведущими нулями ID; multiclass LogLoss с нестандартным порядком prob_<class> столбцов. Оба режима сохраняют конечные нормированные вероятности, проходят final fit и независимую проверку повторно загруженной модели. Это контрактные synthetic-проверки, их метрики не оценивают качество хакатонного решения.

ZIP содержит 34 runtime-файла. В корне solution.py, automl_agent, README, .env.example и requirements-mvp.txt. Датасетов, моделей, тестов, local runs, настоящих ключей и служебных журналов в ZIP нет. README внутри ZIP содержит отдельную команду запуска из корня архива и ссылки на опубликованную документацию. Собранный ZIP остаётся локальным build artifact, в Git хранится сборщик.

## Ограничения доказательств

Полный трёхчасовой soak и платформа с выданным Qwen endpoint здесь не запускались. Docker CLI в среде отсутствует, поэтому сборка Docker не подтверждена; локальный ZIP/процессы и pinned runtime проверены. Hidden-test labels/private tasks недоступны: public/private leaderboard score не вычислялся. OOF после адаптивного выбора и meta assessment не являются независимой nested CV оценкой. RSS sampling дополняет системные ограничения контейнера. Локальная Ollama использует ресурсы отдельной службы, которые не входят в RSS worker.
