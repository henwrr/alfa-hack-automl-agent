# Состояние на 20 октября

Причины старых ошибок подтверждены в ROOT_CAUSES.md. Реализованы проверенные model recipes, наследование фактических параметров и дедупликация. Исполнители GPT-6 Luna xhigh добавили fit-протокол, LightGBM/XGBoost, диагностику данных, OOF postprocessing и публичные audit-проекции; их commits интегрированы.

Root подключил CV/OOF: пять IID фолдов, четыре forward-периода для подтверждённого сдвига 1–5 → 6, host-проверки порядка индексов и fingerprint, train-only early stopping, полное финальное обучение с выбранными параметрами. Свободный код проходит preflight fit/predict/serialization. Лимиты max-trials/max-steps теперь опциональны, по умолчанию используется время и адаптивный резерв. Резерв CV берётся из стоимости одного полного refit, а не пяти fold-обучений.

Проверки интеграции: 47 passed (acceptance, recipe agent, CV research agent, budgeting); ещё 72 focused checks по training/recipes/diagnostics/CV/config прошли. Новые поведенческие RED проверки на неверный порядок OOF и вероятности воспроизвели ошибки; исправлены. Независимое CV review выполнено, дополнительные замечания устранены. Старые holdout-тесты теперь явно выбирают --cv-folds 1 и три старых семейства; новые end-to-end тесты используют CV5 и четыре семейства.

Следующее: подключить финальный refit бленда/множителей, проверить весь suite, выполнить обе полные публичные задачи через Ollama, перепроверить CSV/model reload/resume, собрать ZIP и push в aprelskiy_branch. API организаторов пока отсутствует. LightGBM 4.6.0 и XGBoost 2.1.4 установлены; локальный libomp найден в sklearn wheel. Linux image получил libgomp1.
