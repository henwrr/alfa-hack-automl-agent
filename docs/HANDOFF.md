# Передача контекста (для новой машины / новой сессии Claude)

Прочитай этот файл, затем [README.md](../README.md) и [EXPERIMENTS.md](EXPERIMENTS.md) — в них всё,
что сделано, проверено и отвергнуто. Состояние на 25 октября 2026, ветка `oleg_branch`, коммит `05f5c78`.

## Задача хакатона (Альфа Хак, AI-агент для ML)

- Сдаётся ZIP с `solution.py` в корне; платформа запускает `python solution.py task.md` в Docker:
  8 CPU, 32 GB RAM, **3 часа на задачу**, без интернета, Python 3.12, только библиотеки из `requirements.txt`
  организаторов (есть lightgbm, catboost, xgboost, optuna, pydantic, autogluon, lightautoml, flaml).
- LLM — Qwen3.8-27B через API организаторов, переменные как в `baseline.py`:
  `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`. Свои ключи запрещены, предобученные модели запрещены.
- 3 сабмита в день. Скор задачи = нормировка относительно бейзлайна организаторов (0–100).
- Публичные задачи: регрессия WiDS `site_eui` (RMSE), мультикласс UNSW-NB15 `attack_cat` (Macro F1).
  Финал — 2 скрытые задачи, тот же интерфейс, код не меняется.
- Данные (не в git): Google Drive из PDF задания, папка `data_for_participants`. Локально лежали в
  `~/Alfa/data/{Регрессия,Многоклассовая классификация}/` (task.md + train/test/sample_submission/data_dictionary).

## Команда и ветки

- Репозиторий `github.com/aprelskiyegor/alfa_hackathon` (приватный).
- `oleg_branch` — наш агент (этот код). `aprelskiy_branch` — решение сокомандника: надёжная инфраструктура
  (журнал, resume, supervisor), но LLM писала полный код и почти всегда падала; скор 45.33 / 0.565.
- Пользователь просит подтверждение перед каждым commit/push.
- Приоритеты команды: короткие прогоны (~10 мин) при разработке — норма; сначала всё вокруг модели;
  особенно ценится диагностика «почему модель ошибается» для LLM; подбор порогов F1 — в самом конце.

## Архитектура (agent/)

`task.py` (поиск файлов, target, метрика из текста, авто-разбор дат и текста) → `validation.py` (временные
фолды, если test «в будущем»; иначе повторный holdout 80/20; adversarial check) → `features.py` (язык
операций признаков для LLM + `target_encode`/`target_lag` out-of-fold + поиск ключа сущности) →
`models.py` (LightGBM/CatBoost/XGBoost/ExtraTrees/linear, early stopping, веса классов, log-таргет,
проверка имён параметров) → `diagnostics.py` (ошибки по сегментам/классам, переобучение, важность) →
`researcher.py` + `schemas.py` (LLM-исследователь с pydantic-схемой и structured output; offline-плейбук
и тюнинг) → `runner.py` (цикл: бейзлайн → сильные ходы на быстрой цепочке → LLM, пока она думает — хост
гоняет плейбук → финал: честный ансамбль + усреднение по seed; резерв времени; safety-submission).

## Результаты (локальная валидация, не скор платформы)

| Режим | Регрессия RMSE ↓ | Классификация Macro F1 ↑ |
|---|---|---|
| aprelskiy_branch | 45.33 (случайный holdout) | 0.565 |
| 600 с, без LLM (дефолт) | 33.85 | 0.664 |
| 1 час + qwen3.5:4b | 33.44 | 0.670 |
| 3 часа + qwen3.5:4b | **32.80** (10 631 с, пик 976 МБ) | — |

Главные находки: test регрессии = год 6 (временная валидация обязательна); ключ здания
`floor_area+year_built+facility_type` (79% test) → TE и лаг; в классификации ключ `sbytes`;
усреднение по seed стабильно помогает; подбор порогов классов и EM-сдвиг классов — вредят.

## Не сделано / следующие шаги

1. **Первый сабмит** (`python scripts/package.py` → `dist/solution.zip`); сверить локальную валидацию с публичным скором.
2. **API организаторов** (Qwen 27B): проверить пользу LLM и поддержку structured output (`AGENT_STRUCTURED=1`).
3. **Python 3.12 + версии из requirements.txt** — не проверено (Dockerfile есть, не собирался).
4. Бюджет времени для сабмита: по умолчанию `AGENT_TIME_BUDGET=600`; длинный режим заметно лучше.
5. Мелочь: тюнинг иногда проверяет одну модель дважды (явный параметр = значение по умолчанию).

## Локальная LLM

Ollama + `qwen3.5:4b` (на Mac 8 ГБ 9B не влезала). На машине с RTX 3070 (8 ГБ VRAM) можно пробовать
`qwen3.5:9b` — модель ~7 ГБ, часть слоёв может уйти на CPU. Запуск:
`OPENAI_BASE_URL=http://127.0.0.1:11434 OPENAI_MODEL=qwen3.5:9b AUTOML_PROVIDER=ollama python solution.py task.md`.
Ответы LLM пишутся в `agent_runs/<run>/llm_calls.jsonl`.
