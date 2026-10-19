# alfa_hackathon — AutoResearch-агент (ветка oleg_branch)

Агент сам решает табличную ML-задачу по `task.md`: понимает задачу, строит честную валидацию,
обучает сильный бейзлайн, а затем в цикле **диагностика → гипотеза → эксперимент → сравнение**
улучшает модель и пишет `submission.csv` рядом с `task.md`.

## Запуск

```bash
# с LLM (переменные как в baseline.py)
export OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=...
python solution.py path/to/task.md

# без LLM — эвристический плейбук (для разработки)
python solution.py path/to/task.md --offline --time-budget 300
```

Бюджет времени: `--time-budget` или `AGENT_TIME_BUDGET`, по умолчанию **600 с** (быстрый режим
разработки; лимит платформы — 10800 с). Журнал запуска: `agent_runs/<время>/` рядом с `task.md`
(`agent.log`, `experiments.jsonl`, `context.json`, `report.json`).

Проверка LLM-цикла без ключа: `python tests/fake_llm_server.py 8765`, затем запуск с
`OPENAI_BASE_URL=http://127.0.0.1:8765/v1 OPENAI_MODEL=fake`. Последний запрос к LLM
сохраняется в `tests/last_llm_request.json` — там видно ровно то, что получает модель.

## Как устроено

| Модуль | Что делает |
|---|---|
| `agent/task.py` | находит CSV, target, id, тип задачи, метрику из текста `task.md`; профиль данных |
| `agent/validation.py` | **выбор схемы валидации**: если test целиком «в будущем» по какому-то столбцу — валидация на последних периодах; иначе (стратифицированный) holdout. **Adversarial check** train vs test |
| `agent/models.py` | LightGBM / CatBoost / XGBoost / ExtraTrees / линейная модель; early stopping на внутреннем куске (валидация хоста в обучении не участвует), веса классов, log1p/sqrt таргета, лимит времени на эксперимент |
| `agent/features.py` | язык признаков для LLM: `missing_flags`, `freq_encode`, `group_stats`, `ratio`, `row_stats`, `combine`, `custom` (произвольный pandas-код с AST-фильтром) и др. |
| `agent/diagnostics.py` | **почему модель ошибается**: train vs valid, ошибка по децилям таргета, худшие сегменты (доля ошибки vs доля строк, смещение), крупнейшие ошибки, F1 по классам, топ путаниц, важность признаков |
| `agent/researcher.py` | LLM-исследователь (JSON-спецификации экспериментов) и offline-плейбук как запасной вариант |
| `agent/runner.py` | цикл, учёт времени и резерв на финальное дообучение, safety-submission, журнал |

Принципы:
- метрику считает только хост на своей валидации — LLM не может «объявить» скор;
- LLM описывает эксперимент спецификацией, а не пишет весь пайплайн — меньше падений;
- после каждого улучшения сразу пишется safety-`submission.csv`; финальная модель дообучается на всём train;
- сбой LLM не останавливает агента: раунд делает offline-плейбук.

Из решения в `aprelskiy_branch` перенесены сильные стороны: host-owned метрика (защита от взлома метрики),
гарантированный submission при любых сбоях, атомарная запись, резерв времени на refit, журнал решений.

## Первые результаты (offline, без LLM, бюджет 300 с, MacBook)

| Задача | Валидация | Лучшая модель | Скор | Для сравнения: aprelskiy_branch |
|---|---|---|---|---|
| Регрессия (RMSE ↓) | время: train Year 1–4 → valid Year 5 (test = Year 6) | XGBoost, log1p, индикаторы пропусков | **41.27** | 45.33 на случайном holdout (оптимистичнее) |
| Классификация (Macro F1 ↑) | stratified holdout 20% | LightGBM, sqrt-balanced веса классов | **0.660** | 0.565 |

Находки по данным:
- регрессия: весь test — `Year_Factor=6`, train — годы 1–5 → случайный holdout некорректен;
- классификация: сильный сдвиг train/test (adversarial AUC ≈ 0.87), 38% строк train — дубликаты признаков.

## Дальше
- подключить API организаторов (Qwen) и сравнить LLM-исследователя с offline-плейбуком;
- K-fold/повторная валидация и ансамбль лучших моделей для длинных запусков;
- подбор порогов/весов классов под Macro F1 в конце;
- Dockerfile и сборка ZIP; проверка на Python 3.12 с версиями из `requirements.txt`.
