# Происхождение решений universal_agent

Пути источников ниже относятся к локальным snapshots all_agents. Их точное содержимое зафиксировано SHA256 в SOURCE_INVENTORY.json. Эти snapshots были переименованы при сборе; несовпадение ожидаемого имени Python-пакета с каталогом не приписывается автору алгоритма. Новый пакет лежит в universal_agent/automl_agent и не зависит от наличия all_agents.

Это интеграция проверяемого ядра и отобранных идей, а не конкатенация четырёх entrypoints. У Oleg/Rakovskiy побайтно совпадают diagnostics.py, features.py, llm.py, metrics.py, schemas.py, validation.py и __init__.py; models.py, researcher.py, runner.py и task.py различаются; совместное происхождение указано явно, без выдуманного единоличного авторства.

| Идея / преимущество | Конкретный источник | Реализация в universal_agent | Адаптация и проверка |
|---|---|---|---|
| Host сам оценивает эксперименты, фиксирует метрику и лидера | aprelskiy/aprelskiy_agent/agent.py, evaluation.py; также ruzik/agent_ruzik/runtime.py | automl_agent/agent.py, evaluation.py | Исходное ядро; единый scorer, finite checks, immutable split. Наследованные integration tests |
| CV/OOF, fingerprints, проверка оси индексов и folds | aprelskiy/aprelskiy_agent/validation.py, cv_worker.py, evaluation.py | одноимённые модули | Сохранено; авто группировка точных дубликатов добавлена отдельно |
| Temporal validation при future-only test и понятной временной семантике | aprelskiy/aprelskiy_agent/data_diagnostics.py, validation.py; oleg и rakovskiy/validation.py | data_diagnostics.py, validation.py | Forward CV без будущих периодов в train; earliest period не имеет OOF |
| Inner early stopping и refit внешнего fold | aprelskiy/aprelskiy_agent/training.py | training.py | Только outer-train; final iteration override, реальные metadata, XGBoost label decoding |
| Каталог валидированных recipes и самостоятельный код кандидатов | aprelskiy/aprelskiy_agent/recipes.py, families.py; oleg/agent_oleg/models.py:valid_params | recipes.py, families.py | Исходный allowlist усилен DSL; неподдержанные и host-controlled параметры отклоняются до fit |
| Декларативный язык новых признаков вместо переписывания модели | oleg/agent_oleg/features.py:OPS_DOC,apply_ops; rakovskiy/agent_rakovskiy/features.py (общий код) | новый features.py, recipes.py, agent.py, prompts.py | Заимствованы операции и идея; реализация переписана как sklearn transformer. Исключены custom exec, test statistics, target keys, глобальный TE. До 16 ops / 128 новых колонок |
| Частоты категорий, групповые агрегаты, bins | те же features.py | features.py:FeatureTransformer | В отличие от исходного train+test transform, fit только внутри fold; unknown category -> 0 frequency / NaN group aggregate; проверено изменение состава transform batch |
| Признаки текста: длина, слова и частоты слов | oleg/agent_oleg/task.py:_expand_text | features.py:text_stats | Явная LLM-гипотеза в DSL; top vocabulary учится на fit, фиксированная ширина, ограниченная длина/объём. Исходная train+test vocabulary не переносится |
| Одинаковые строки признаков в одной группе валидации | ruzik/agent_ruzik/data.py:make_split | validation.py:make_validation_plan | GroupKFold/StratifiedGroupKFold вместо одного GroupShuffleSplit. Исключены IDs/target; редкие группы не вызывают исходную ошибку «разбиение потеряло классы»; невозможно независимый CV для единственной группы -> явная ошибка с ранним fallback |
| Немедленный резервный submission и checkpoint каждого лидера | ruzik/agent_ruzik/runtime.py:run; oleg/rakovskiy runner.py:safety prediction; aprelskiy/agent.py | agent.py:_emergency_submission,_commit_trial; evaluation.py | Fallback перенесён ДО profile/diagnostics/split; атомарная публикация и сохранение предыдущего файла при сбое final |
| Контроль времени и очистка process tree | ruzik/agent_ruzik/runtime.py:run_worker,stop_process; aprelskiy/runtime.py,scheduler.py | runtime.py,scheduler.py; новый общий CLI timer в solution.py | Отдельная process group, RSS и output caps, <=8 потоков, <=28 GB worker caps. CLI timer включает host work, не только fit |
| Резерв финального fit по измеренному времени | aprelskiy/budgeting.py, agent.py; ruzik/runtime.py | budgeting.py, agent.py | Сохраняется измеренная оценка/запас; final timeout ограничен остатком, оценка не объявляется гарантией |
| Воспроизводимый случайный поиск по ограниченным сеткам | rakovskiy/agent_rakovskiy/researcher.py:_tuning_moves; также общий код oleg/researcher.py | новый search.py; agent.py:bounded_search | Небольшие host recipes, seed зависит от шага, dedup и тот же CV. Автоматически <=4 trials при сбое LLM или отсутствии успешных LLM-кандидатов, либо --search-trials; origin=search не выдаётся за LLM |
| Разнообразные ансамбли и separate meta assessment | aprelskiy/postprocessing.py, ensemble.py, final_ensemble.py; oleg/rakovskiy runner.py:Caruana | postprocessing.py, ensemble.py, final_ensemble.py | Взята более защищённая OOF/cross-fitted реализация ядра; маленькие смеси, bounded meta worker, явная оговорка adaptive OOF bias. Без безусловного seed bagging |
| Вероятности и class axis для новых задач | oleg/rakovskiy task.py:discover, runner.py:save_submission; ruzik/data.py:write_submission | contracts.py,data.py,evaluation.py,worker.py,cv_worker.py,final_ensemble.py | Однозначный sample mapping; бинарная вероятность положительного класса; class probabilities вместо ошибочных labels для AUC/LogLoss; единый validator |
| Более широкий набор метрик | oleg/rakovskiy metrics.py; найденная ошибка aprelskiy parser balanced accuracy | data.py,evaluation.py,config.py,validation.py,postprocessing.py | Добавлены MSE/MAPE/RMSLE/Balanced Accuracy; MAPE использует sklearn convention вместо произвольного 1e-9 |
| OpenAI-compatible и native Ollama, нормализация tool calls | aprelskiy/providers.py; oleg/rakovskiy llm.py | providers.py,config.py | Основной протокол aprelskiy; локальный default Ollama без ключа, baseline env сохраняет platform API |
| Устойчивый журнал, redaction, resume оплаченных ответов | aprelskiy/journal.py,audit.py,agent.py | те же модули | Run lock, fsync/atomic snapshots, fingerprint, interrupted calls не повторяются молча |
| Проверка схемы, строк, ID, finite predictions, атомарная замена | aprelskiy/evaluation.py; ruzik/data.py:write_submission | evaluation.py | ID как строки, exact order, вероятность/классы, round-trip. При ошибке кандидат не затирает прежний файл |
| Детерминированный ZIP allowlist и независимая проверка запуска | aprelskiy/scripts/package_solution.py,verify_run.py и tests | scripts/,tests/ | Исправлен layout и доставлены все обязательные runtime-файлы; ZIP не включает исходные данные, тесты, secrets, models/runs |
| Точный binary F1 threshold с учётом ties и positive class | Обновлённый ruzik, commit 85f4988ba11c70ebe2242c8722b32f2d4bea5c7a: ruzik_agent/decisions.py:19–35 | postprocessing.py:_tune_binary_threshold,binary_threshold_labels; ensemble.py, agent.py, final_ensemble.py, prediction.py, audit.py | Алгоритмическая идея заимствована; настройка перенесена в существующий cross-fitted/forward meta assessment. Отдельное правило меток, неизменные вероятности, проверки оси классов, сериализации и отказа при отсутствии независимого выигрыша |
| Календарные epoch-days/month/weekday | Тот же обновлённый ruzik: ruzik_agent/models.py:98–116 | features.py:date_parts; экспортированный recipe source | Переписано как явная ограниченная операция без эвристики имени; ISO8601/UTC, fit-only проверка >95%, исходная колонка сохранена, защита границ timestamp и numeric-as-nanoseconds |

Подробный [повторный аудит Ruzik 85f4988](reviews/ruzik_update_85f4988.md) отделён от первоначального snapshot и фиксирует полезные, уже покрытые и отложенные идеи новой ветки.

## Что намеренно исключено и почему

- Поиск entity key на target всего train перед CV из oleg/rakovskiy: выбор признака уже использует validation labels. Даже OOF encoding после этого не отменяет selection leakage.
- Их train+test агрегаты и vocabulary: это transductive protocol, который затрудняет перенос оценки. Здесь никакие learned feature statistics не обучаются на validation/test.
- Их глобальная обрезка regression prediction диапазоном train target: запрещает полезную экстраполяцию. Для RMSLE используется только математически необходимый clip0 при оценке, а не ограничение train min/max для всех задач.
- Синхронные fit без hard timeout: контролируются workers и общим CLI deadline.
- Слепое добавление трёх seed к финалу: может съесть резерв без проверенного выигрыша. Используется bounded OOF ensemble; отдельный seed bagging не включён.
- Автоматический target encoding/lag по подобранному ключу: ошибка temporal protocol или supervised selection легко завышает score. В новом Ruzik 85f4988 ключ выбирается без y и история strictly past-only; это существенное исправление. Перенос отложен до отдельной проверки вложенного отбора/временных границ и внешнего качества, а не объявлен заведомой утечкой обновлённой реализации.
- Свободный custom pandas exec в декларативном режиме: заменён ограниченными операциями. Полный code mode остаётся отдельным явным режимом ядра с preflight/supervisor, не называется OS sandbox.
- Все доступные AutoML framework сразу: не добавляют доказанного выигрыша и увеличивают импорт/ресурсы. Runtime использует только необходимое подмножество исходного requirements.txt.

## Код ядра без существенного изменения идеи

Файлы audit.py, journal.py, budgeting.py, runtime.py, scheduler.py, training.py, families.py, modeling.py, strategy.py, diagnostics.py, data_diagnostics.py и providers.py взяты из aprelskiy/aprelskiy_agent. Остальные модули ядра перечислены в таблице и расширены под best-of требования. 378 исходных тестов сохранены, новые сценарии добавлены отдельно; исходный временной тест обновлён только потому, что после отключения temporal semantics теперь ожидается duplicate-group CV.

Объём переноса не является доказательством превосходства по hidden score. Итоговые измерения и непроверенные сценарии перечислены в VALIDATION.md.

## Дополнительные изменения после живой проверки

На реальной Ollama короткий классификационный прогон тратил ответы на необязательные decision metadata и возвращал декларации без реального изменения recipe. Промпт упрощён: мотивировка остаётся в hypothesis, исполняемые поля обязательны по смыслу, подробный decision необязателен. Следующий живой прогон сформировал корректные веса и параметры. Это изменение интеграции, а не идея, приписанная одному из исходных авторов.
