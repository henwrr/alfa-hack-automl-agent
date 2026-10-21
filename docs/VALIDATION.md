# Проверка MVP

Проверено 21 октября 2026: macOS arm64, Python3.12.13, зависимости requirements-mvp.txt, локальная Ollama qwen3.5:9b.

## Публичные задачи

| Показатель | Регрессия | Классификация |
|---|---:|---:|
| Train / test rows | 53308 / 22449 | 87670 / 41166 |
| Валидация | 4 forward-group folds | 5 stratified folds |
| OOF coverage | 0.9602874 | 1.0 |
| Лучший стартовый кандидат | ExtraTrees, RMSE48.63484 | XGBoost, Macro F1 0.6110302 |
| Лучший LLM-кандидат | ExtraTrees, RMSE48.55996 | XGBoost, Macro F1 0.6123388 |
| Успешных LLM trials | 10 | 4 |
| Мета-оценка постобработки | RMSE48.23387 | Macro F1 0.6475200 |
| База на тех же строках мета-оценки | RMSE48.83698 | Macro F1 0.6123388 |
| Final refit на всём train | ok | ok |

В регрессии train содержит Year_Factor1–5, test —6. Внешние folds обучаются только на прошлых периодах, earliest period не получает OOF. Мета-оценка проверяет перенос между более поздними периодами, поэтому её сравнивают с baseline на тех же строках. Adversarial ROC-AUC=1.0, в том числе без временного столбца. Итоговый бленд ExtraTrees/CatBoost/LightGBM:0.5/0.25/0.25.

В классификации adversarial ROC-AUC=0.73345. CatBoost не завершил полный CV за1800s и не получил score. Это ограничение локального времени, не доказательство худшего качества семейства. Итоговый бленд XGBoost/LightGBM:0.75/0.25, с множителями вероятностей классов. Два прерванных trial отмечены interrupted и автоматически не повторялись.

Мета-оценки используют cross-fitted/forward assessment, но базовые модели уже адаптивно выбраны по этим OOF результатам. Это не nested CV и не независимый тест платформы. Скрытые test labels недоступны.

## Проверки

- Полный suite после реализации:378 passed,2 Linear ConvergenceWarning,87.59s.
- Независимо пересчитаны метрики всех успешных публичных trials; проверены code SHA, fit weights и early stopping каждого fold.
- Обе итоговые модели загружены заново; predict на всех test строках совпал с submission.csv. Подтверждён refit на всех train строках.
- По интервалам процессов: максимум2 исследовательских worker и8 назначенных CPU-потоков.
- Completed resume восстановил повреждённые agent.log/decisions.jsonl, сохранив SHA256 событий, модели и CSV без нового обучения или LLM-вызовов.
- Распакованный ZIP запущен из другой cwd на новой синтетической схеме с кириллицей в пути:CV5,240train/17test,Macro F1 0.9832168. Final fit,model reload и resume прошли.

После очистки ветки отдельный export Git index без локальных данных прошёл15 проверок CLI, acceptance и упаковки за7.42s. Проверены локальные Markdown-ссылки и отсутствие датасетов/моделей/логов в составе export.

Данные, сохранённые модели, полные журналы и служебные протоколы разработки не входят в репозиторий. При новых запусках агент создаёт артефакты в --run-dir, а submission.csv — рядом с task.md.

## Повторить проверку

~~~bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/package_solution.py
.venv/bin/python scripts/verify_run.py /path/to/run --output /tmp/run-verification.json
~~~

Тесты создают собственные синтетические данные и не требуют публичных датасетов, Ollama или API-ключа. verify_run.py требует артефакты уже завершённого запуска.

API организаторов, Docker image и непрерывный трёхчасовой soak здесь не проверены. AST guard и supervisor не заменяют изоляцию ОС/контейнера. Поддерживается tabular CSV с одним целевым столбцом; probability/multicolumn submission требует расширения контракта.
