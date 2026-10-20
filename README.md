# AutoML MVP

CLI-агент исследует supervised tabular CSV и сохраняет submission рядом с task.md. После настройки провайдера обычный запуск без исследовательских флагов использует до трёх часов, CV, сравнение моделей и LLM-гипотезы:

~~~bash
.venv/bin/python solution.py task.md
~~~

Поддерживаются задачи регрессии и классификации с одной целевой колонкой. Качество заранее неизвестно: завершение задачи и успешный fit сами по себе не означают улучшения метрики.

## Установка

Требуются Python 3.12 и [uv](https://docs.astral.sh/uv/):

~~~bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-mvp.txt
~~~

Для тестов:

~~~bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q
~~~

requirements-mvp.txt — runtime-подмножество закреплённых версий из организаторского requirements.txt. ZIP содержит только это runtime-подмножество.

CPU wheels для CatBoost, LightGBM и XGBoost могут требовать системную OpenMP runtime-библиотеку. Linux Docker image устанавливает libgomp1. При ошибке загрузки libomp/libgomp сверяйтесь с официальными инструкциями [LightGBM](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/install.html) и [CatBoost](https://catboost.ai/docs/en/concepts/python-installation), учитывая свою ОС и Python wheel. На локальной macOS-среде запуск использовал уже имеющийся sklearn/.dylibs/libomp.dylib через process-local rpath; Homebrew там не установлен. Это особенность той среды, а не новая зависимость проекта. Глобальные пути библиотек проект не меняет.

## Провайдер и запуск

Для OpenAI-compatible endpoint задайте переменные окружения:

~~~bash
export AUTOML_PROVIDER=openai
export OPENAI_API_KEY='ваш ключ'
export OPENAI_MODEL='имя модели'
export OPENAI_BASE_URL='https://адрес-endpoint/v1'
.venv/bin/python solution.py task.md --run-dir runs/my-run
~~~

Для локальной Ollama:

~~~bash
export AUTOML_PROVIDER=ollama
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=qwen3.5:9b
export OPENAI_BASE_URL=http://127.0.0.1:11434
.venv/bin/python solution.py task.md --run-dir runs/qwen-run
~~~

Ollama и модель должны быть установлены и запущены отдельно. Native endpoint использует адрес без /v1. Переменные и пример находятся в [.env.example](.env.example), но приложение не загружает файл .env автоматически: экспортируйте значения из shell или используйте менеджер секретов. Не помещайте ключи в task.md, CLI-аргументы или код.

После запуска можно следить за решениями:

~~~bash
tail -F runs/qwen-run/agent.log
~~~

Путь к данным задаётся в task.md. Параметры задачи, провайдера и бюджета сохраняются в run-dir.

## Режим исследования и лимиты

Без исследовательских флагов действует CV-протокол. Основные параметры CLI:

| Флаг | Значение по умолчанию | Назначение |
|---|---:|---|
| --time-budget | 10800 секунд | Общий wall-clock бюджет; максимум 10800 |
| --trial-timeout | 1800 секунд | Максимальное время одного candidate trial |
| --llm-timeout | 180 секунд | Таймаут одного запроса LLM |
| --max-trials | без фиксированного предела | Необязательный лимит всех trials, включая стартовые и неудачные |
| --max-steps | без фиксированного предела | Необязательный лимит шагов исследователя |
| --max-tokens | 2000000 | Общая страховочная граница входных и выходных токенов |
| --cv-folds | 5 | Число OOF folds; значение 1 включает прежний holdout-режим |
| --validation-strategy | auto | IID CV или forward-group CV по подтверждённому future-only периоду |
| --family-trials | 4 | Максимальное число начальных семейств |
| --parallel-trials | 2 | До двух процессов одновременно |
| --n-jobs | 4 | CPU-потока на процесс; общее назначение не выше восьми |
| --experiment-mode | recipe | Валидированные рецепты; code включает свободный Python |
| --time-column | автоматически | Необязательное явное указание временного признака |
| --run-dir | .automl/timestamp-id рядом с task.md | Переопределить каталог артефактов; с --resume продолжить его |

--max-trials и --max-steps можно указать, чтобы остановить исследование раньше. Обычно исследование ограничивают общее время и резерв на финальный fit. Начальный резерв составляет 15% общего времени и пересчитывается по измеренным fold-fit/refit данным; это оценка, а не гарантия успеть переобучить модель. Исследовательский timeout не ограничивает финальный fit.

Для короткого legacy-smoke указывайте holdout и timeout явно, чтобы случайно не запускать несколько полных CV fits:

~~~bash
.venv/bin/python solution.py task.md --offline --cv-folds 1 --trial-timeout 180 \
  --time-budget 300 --family-trials 1 --max-trials 1 --n-jobs 1 \
  --run-dir runs/smoke-holdout
~~~

cv-folds=1 использует прежнее разбиение и не создаёт OOF артефакты. Его метрики не сравнивайте с CV, особенно с forward temporal CV.

## CV, кандидаты и исследовательские инструменты

Сначала host строит bounded train/test диагностику и сохраняет dataset_diagnostics.json. По умолчанию для iid данных используются пять folds: стратифицированные для классификации, когда возможно, иначе KFold. При validation-strategy=auto временная схема выбирается, когда семантика названия признака поддерживает время, в test все значения позже максимума train и есть минимум три train-периода. Для пяти train-периодов это даёт четыре forward-group folds; первый период не получает OOF предсказаний. Эта проверка опирается на наблюдаемое распределение, а не на фиксированное имя датасета. Явный --time-column задаёт временной столбец; validation-strategy=iid отключает автоматический temporal выбор.

Стартовый shortlist включает до четырёх CPU-семейств при доступных пакетах: CatBoost до 3000 итераций, LightGBM до 5000, ExtraTrees и XGBoost до 3000. Начальные boosting-рецепты запрашивают 100 rounds early stopping на отдельном inner train-only eval_set; ExtraTrees обучается без early stopping. Выбранные итерации сохраняются по folds и используются для final iteration override при refit. Внешние OOF labels не используются для inner stopping. Реальные параметры и время обучения записываются в результаты, поэтому заявленное LLM семейство само по себе не считается доказательством backend.

По умолчанию LLM использует tool run_model_experiments. Каждая гипотеза указывает base_trial_id успешного измеренного recipe trial и только изменённые поля parameters, preprocessing и/или fit_options. Host наследует измеренные defaults, проверяет параметры и веса, строит полный recipe/code и отклоняет дубликаты до обучения. sample_weight фактически передаётся в fit; early stopping создаёт внутренний eval_set; двойное применение весов запрещено.

Флаг --experiment-mode code включает run_experiment/run_experiments со свободным Python. До полного CV host выполняет ограниченный preflight на данных только из train части первого fold: fit, predict и проверку сериализации/reload. Ошибка preflight отменяет дорогой CV trial. В recipe mode preflight не нужен, поскольку код строит host из проверенной схемы.

После candidate CV host независимо считает OOF метрики, ошибки классов или сегментов и важность признаков. Для совместимых successful trials может выбрать блендинг до трёх разных кандидатов и class multipliers для Macro F1. Подбор postprocessing и его cross-fitted/forward assessment разделены; calibrated_oof_score показывается отдельно от assessed_score. Это не nested CV: базовые модели уже выбраны с помощью тех же OOF результатов, поэтому мета-оценка не является независимым тестом и может быть оптимистичной. В отчёте сохраняются причина выбора/отказа и применённые параметры.

## Артефакты и отчёты

submission.csv пишется рядом с task.md. Укажите --run-dir, чтобы задать место для журнала и результатов.

| Артефакт | Содержимое |
|---|---|
| cv_plan.json | Общий split/fold plan, validation coverage, временная политика и fingerprint |
| dataset_diagnostics.json | Train/test drift, temporal candidates и adversarial ROC-AUC |
| agent.log, decisions.jsonl | Публичная читаемая и структурированная проекции решений |
| events.jsonl | Первичный журнал событий, LLM usage, ошибок и результатов |
| trial_XXXX/recipe.json, code.py | Recipe, hash и полный код кандидата |
| trial_XXXX/preflight.json | Train-only preflight для свободного code mode |
| trial_XXXX/fold_*.json | Метрики, inner stopping и параметры каждого CV fold |
| trial_XXXX/valid_predictions.csv, oof_indices.npy, oof_fold_ids.npy | OOF предсказания, позиции и принадлежность fold |
| trial_XXXX/valid_probabilities.npy, test_probabilities.npy | Вероятности классов, если модель их поддерживает |
| trial_XXXX/diagnostics (в experiments.json/report.json) | Class F1/recall, confusion, regression segments, OOF summary, feature importance |
| postprocessing.json | Принятие/отказ, blend weights, class multipliers и разделённые scores |
| final_*/model.joblib | Сохранённая итоговая модель или ensemble |
| experiments.json, report.json, report.md | Trial outcomes, выбранный кандидат, refit, submission и пути |

report.json status completed означает, что успешно выполнен хотя бы один LLM candidate; degraded — что успешного LLM candidate нет, хотя рабочий baseline мог сохранить submission; offline — запуск без LLM. Ни один статус не гарантирует улучшения метрики, лучшего качества submission или независимого подтверждения гипотез. Проверяйте best_metric, выбранный trial, postprocessing и final_fit.status отдельно. Финальная проверка submission охватывает схему/строки файла, но не скрытую метрику платформы.

Полный live CV run обеих публичных задач ещё не выполнен, поэтому новых числовых результатов в этом README нет. Исторические прогоны и протоколы находятся в [docs/VALIDATION.md](docs/VALIDATION.md); holdout scores из предыдущих запусков нельзя напрямую сравнивать с temporal CV. Подробности текущего протокола: [docs/RESEARCH_STRATEGY.md](docs/RESEARCH_STRATEGY.md).

## ZIP и Docker

Собрать платформенный ZIP:

~~~bash
.venv/bin/python scripts/package_solution.py
~~~

Результат — dist/solution.zip. В архив входят solution.py, runtime-пакет automl_agent, README, .env.example и requirements-mvp.txt. Данные, secrets, .venv, тесты и результаты запусков исключены.

В Dockerfile установлен Linux runtime libgomp1. Пример запуска:

~~~bash
docker build -t automl-agent:local .
docker run --rm --env-file .env -v "$PWD/data:/data" automl-agent:local \
  /data/task.md --run-dir /data/runs/docker-run
~~~

На текущей macOS-машине Docker CLI отсутствует, поэтому сборка и запуск контейнера здесь не подтверждены. Для отдельной изолированной среды обеспечьте ограничения CPU, памяти, сети и файловой системы на стороне среды запуска.
