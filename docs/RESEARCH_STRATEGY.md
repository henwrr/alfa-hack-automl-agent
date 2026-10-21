# Стратегия исследования

## Нормальный режим запуска

Команда без флагов исследования использует 10800 секунд общего wall-clock бюджета, два параллельных trial по четыре CPU-потока, до 1800 секунд на один candidate, максимум 2 млн суммарных токенов как safety cap. max_trials и max_steps без фиксированных значений: исследование заканчивается по времени/резерву, token cap, явному finish_research или системной остановке. Любые заданные max лимиты становятся верхними границами. Суммарное назначение потоков ограничено восемью.

При наличии библиотек стартовое сравнение включает CatBoost (до 3000 итераций, learning_rate 0.05, depth 6), LightGBM (до 5000, learning_rate 0.05, num_leaves 31), ExtraTrees и XGBoost (до 3000, learning_rate 0.05, max_depth 6). Бустеры используют inner early stopping до 100 rounds. У каждого семейства свой рецепт; недоступные пакеты не запускаются. Каталог и фактически установленные версии лежат в профиле запуска.

Ограниченный startup shortlist, каждое CV исследование и финальный refit расходуют общий бюджет. Начальные проверки не получают отдельную процентную квоту. Если time budget завершится раньше, чем все семейства и гипотезы успеют выполниться, отчёт должен показать фактическую остановку.

## Диагностика и выбор CV

До обучения host сохраняет train/test profile и dataset_diagnostics.json. В нём находятся пропуски, распределения категорий, численные сдвиги, temporal candidates и adversarial train/test ROC-AUC. AUC — вспомогательная диагностика того, насколько признаки различают train и test; это не target metric и не причина автоматически отбрасывать данные.

По умолчанию используется пять OOF folds. Для iid данных классификация стратифицируется, если редкие классы позволяют; иначе строится KFold. При strategy=auto temporal CV включается только когда найденный по семантике имени числовой или datetime признак имеет минимум три train-периода, а все test-значения строго позже максимума train. Например, при пяти train-периодах формируются четыре forward-group folds по последним четырём периодам; самый ранний период не получает OOF-предсказания. Для регрессии это проверяет перенос на будущий период и обычно сложнее случайного holdout.

План CV записывается в cv_plan.json и разделяется всеми кандидатами: folds, стратегия, coverage, warnings и fingerprint должны совпадать. outer fold отделяет train от OOF validation; targets валидации остаются у host scorer. --cv-folds 1 переключает режим на прежний holdout и не создаёт полного OOF протокола. Это полезно для короткого smoke, но его метрики не сравнивают напрямую с CV, особенно forward temporal.

## Обучение кандидатов

Каждый candidate оценивается на всех folds общего плана. Для CatBoost, LightGBM и XGBoost host выполняет внутреннее early stopping только на части outer-train, передавая реальный eval_set в native fit. Обычный inner validation — отдельное случайное/стратифицированное train-only разбиение; для temporal fold inner split использует более ранние периоды и последний период outer-train. При недостатке строк или редком классе stopping может быть пропущен с явной причиной. Outer validation labels не используются в fit или выборе числа итераций.

На каждом outer fold early stopping выбирает число деревьев, затем свежая модель refit-ится на всём outer-train с этим количеством. ExtraTrees не использует early stopping. При финальном refit на полном train host задаёт iteration override: медиану fold значений для IID или значение последнего forward fold для temporal. Применённые параметры и время, inner размеры, best iteration и статус сохраняются в fold_*.json и worker_result.json.

Structured recipes поддерживают host-controlled sample weights. Для classification balanced/sqrt-balanced веса вычисляются по outer-train строкам и передаются в fit как sample_weight; constructor class weights и отдельные веса нельзя включать одновременно. Веса не применяются молча. Test rows и targets не участвуют в обучении.

Свободный code mode проходит до полной CV train-only preflight: на небольшом наборе из train части первого fold запускаются fit/predict и serialization/reload. Ошибка блокирует полный CV trial. Preflight не получает outer validation или test targets. Structured recipe mode собирается host из валидированной схемы и обходится без произвольного preflight.

## Гипотезы LLM

По умолчанию активен validated recipe mode и инструмент run_model_experiments. Каждая гипотеза указывает успешный измеренный base_trial_id и только изменения parameters, preprocessing и/или fit_options. Host наследует recipe и фактические допустимые defaults, например автоматически выбранный learning_rate CatBoost, проверяет типы/ranges/сочетания весов, строит полный код модели и отвергает уже измеренную конфигурацию до дорогого fit. Формат сохраняет проверенный preprocessing и штатную оценку кандидата.

Параметр experiment-mode=code включает прежние run_experiment/run_experiments и произвольный Python candidate. Кандидат обязан вернуть unfitted estimator с predict; host проверяет код и запускает preflight, затем полную CV. Code mode остаётся расширенным интерфейсом с более широким риском ошибок. Worker выполняет каждый trial в отдельном процессе; host назначает IDs, сравнивает метрики и выбирает победителя.

В каждой гипотезе публично фиксируются наблюдения, семейство, причина, ожидаемый эффект, альтернативы, риски и оценка прошлого результата. Ожидания не считать измерениями. Успешный fit подтверждает выполнение кода, но не пользу гипотезы.

## OOF, выбор кандидата и postprocessing

Host пересчитывает метрику по OOF predictions и хранит per-class, confusion, regression segments, feature importance и fold-level summaries. Feature importance описывает association модели, не причинный эффект. Основным критерием остаётся метрика из task.md; тай-брейк и фактические параметры записываются в решениях и отчёте.

После trials host может собрать ансамбль максимум из трёх совместимых кандидатов, предпочитая разные семейства. Для Macro F1 дополнительно подбираются class multipliers по вероятностям. В postprocessing.json отдельно указаны исходный baseline_score, assessed_score по cross-fitted/forward meta folds и calibrated_oof_score правила, обученного на OOF. При принятии план/weights/multipliers используются в итоговом refit.

Разделение meta-training и assessment rows уменьшает непосредственную подгонку к одной и той же мета-части, но nested CV не реализована: базовые модели и их ранжирование уже адаптивно выбраны по этим OOF результатам. Поэтому assessed_score не является независимым тестом, а calibrated_oof_score нельзя выдавать за независимую оценку. Temporal оценка дополнительно отражает перенос между периодами. Логи и report сохраняют предупреждения и причину принятия/отказа.

## Бюджет и ограничения

На старте host резервирует 15% времени для финального обучения и записи результата. После успешных CV trials reserve пересчитывается по измеренным refit estimates; при возможном блендинге учитывается стоимость до трёх моделей. Это оценка, а не гарантия: fit может идти нелинейно или превышать ожидание. Исследование останавливается перед резервом, финальный refit получает оставшееся время. Candidate trial ограничен --trial-timeout; он не задаёт timeout финальному refit.

По умолчанию scheduler допускает два процесса с четырьмя потоками каждый. Настройки n_jobs × parallel_trials не должны превышать восемь; RSS лимиты worker контролируются отдельно. Supervisor прекращает группу при deadline или системной ошибке и не запускает заново trial с неизвестным исходом при resume.

OOF selection остаётся исследовательской процедурой с риском переобучиться на одну задачу. Значение status=completed означает, что успешно выполнен хотя бы один LLM-кандидат, а не то, что его метрика улучшилась или что скрытый test подтверждён. Сравнения прежнего holdout нельзя считать прямым benchmark temporal CV. Краткие результаты публичной проверки и ограничения находятся в [VALIDATION.md](VALIDATION.md).

## Артефакты и решения

- cv_plan.json — общий OOF split/fold план, coverage, warnings и fingerprint.
- dataset_diagnostics.json — bounded train/test drift и adversarial CV.
- trial_XXXX/recipe.json и code.py — измеренный рецепт/hash и полный standalone код.
- trial_XXXX/preflight.json — результат train-only preflight для code mode.
- trial_XXXX/fold_*.json — fold metrics, actual training options, early stopping, best/final iterations.
- trial_XXXX/valid_predictions.csv, oof_indices.npy, oof_fold_ids.npy — OOF outputs и индексы; вероятности при поддержке лежат в valid_probabilities.npy/test_probabilities.npy.
- postprocessing.json — выбор blend/calibration и его оценки.
- experiments.json, report.json, report.md — результаты trials, diagnostics, победитель, postprocessing и final_fit.
- final_*/model.joblib — сериализованная финальная модель или ensemble; submission.csv сохраняется рядом с task.md.

agent.log даёт человекочитаемую хронологию, decisions.jsonl — её структурированные публичные проекции, events.jsonl — durable источник событий. Детали LLM messages/raw code не предназначены для публичных объяснений. Для просмотра во время запуска используйте tail -F к agent.log.
