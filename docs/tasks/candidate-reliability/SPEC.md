# Надёжность LLM-кандидатов и качество исследований

Пользователь поручил выяснить причины провалов предыдущих live-run, исправить их, повторно проверить обе полные публичные задачи и push в aprelskiy_branch. Предыдущий runtime 2fc603c; Ollama qwen3.5:9b, 8 CPU workers максимум, 2×4 потока; API организаторов нет. Модель исполнителей по предпочтению пользователя: GPT-6 Luna xhigh.

## Подтверждённые причины
Regression: DataFrame превратился в ndarray до обращения по именам, синтаксическая ошибка и неверные пары ColumnTransformer вместо троек; сломана проверенная предобработка. Classification: неверные class_weights='Balanced', depth=-1, sample_weight в constructor; вычисленные и неиспользованные веса; фактический LR baseline≈0.488 против LLM0.05/0.03/0.02. get_params/all_params проверены по сохранённым моделям. Краткий профиль и metadata не давали LLM реальных defaults; устаревшая ошибка оставалась в last_failed_code даже после успешных опытов.

## Критерии
| ID | Сценарий | Ожидаемый результат |
|---|---|---|
| R1 | LLM меняет параметры/веса/индикаторы пропусков | Декларативный tool, host валидирует recipe и создаёт полный Python из проверенного pipeline; отказ на неправильных параметрах до дорогого fit |
| R2 | Две recipe-гипотезы | Общие split/ledger/budget, реальные 2×4 CPU, отдельные результаты; старые code tools и resume сохранены |
| R3 | Наследование успешного recipe trial | Defaults берутся из измеренных допустимых параметров; LR CatBoost сохраняется, если явно не изменён; нормализованный recipe fingerprint исключает повторение той же конфигурации |
| R4 | Успешный fitted estimator | В metadata есть bounded JSON фактических параметров, включая CatBoost auto LR/веса, без arbitrary custom getters и без поломки старых metadata contracts |
| R5 | Компактизация/ошибка/успех | LLM сохраняет class counts, доступные пакеты, измеренные параметры и сравнения; устаревшая ошибка не выдаётся за последнее состояние; ранняя остановка без eval_set не обещается |
| R6 | Проверка обеих public через Ollama | Успешные LLM-directed trials на обеих задачах, валидные refit/CSV/reload/resume, реальные изменения применены; качество сравнивается честно с 45.329028 RMSE и0.565019 MacroF1 без гарантии улучшения |
| R7 | Доставка | Полный suite, independent review, ZIP, короткие commits в aprelskiy_branch и push с проверкой remote SHA |

Нет ручных правил под конкретные имена датасетов/target, внешних данных, скрытого score или подмены LLM фиксированным поиском. Исследователь выбирает изменения; host реализует проверенные операции. Свободный Python остаётся расширенным режимом, совместимость старых ответов сохраняется.

## Дополнение пользователя, 20 октября
R8. Запуск без флагов ограничен прежде всего общим временем/резервом, а не шестью trials: optional max-trials/max-steps, достаточный token safeguard; stop reason честный.
R9. Диагностика: матрица ошибок и class F1/recall, важность признаков, regression errors по доступным категориальным сегментам, train/test drift и adversarial AUC. Проверка фактов показала train Year_Factor1–5, test6; этот сдвиг учитывается автоматически по семантике временного признака, без жёсткого имени датасета.
R10. Верифицированные сильные CPU CatBoost/LightGBM/XGBoost с 2000–5000 максимальных итераций и настоящим inner early stopping. Нет использования outer OOF labels для выбора итерации. При недоступной библиотеке — явное исключение из кандидатов.
R11. По умолчанию CV5 для iid; стратификация когдавозможна, forward group CV при подтверждённом future-only временном признаке. Для5train-yeargroups максимум4forwardфолда. OOF coverage и fold metrics записаны; одинаковый план для всех trials, нет сравнения scores разных splitкакоднойметрики.
R12. Host fit options реально поддерживают sample_weight, внутренний eval_set и early stopping. Builder context не получает host validation targets; no-op веса/earlystopping невозможны в structured mode.
R13. Блендинг до3 лучших кандидатов по OOF и постобработка вероятностей для F1. Подбор weights/множителей отделён от оцениваемой meta-части; в отчёте различаются selection/calibration/heldout scores, не обещается гарантированныйприрост. Применённые weights/threshold multipliers воспроизводятся при finalrefit/reload.
R14. Свободный Python перед полным CV проходит маленький fit/predict/serialization preflight; провал описан и дорогой trialнеидёт. Все изменения на обеихполныхpublicчерезOllama, ZIP/tests/review/pushaprelskiy_branch.
