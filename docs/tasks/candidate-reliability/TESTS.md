# Проверки

Baseline: 160 passed, 44.88s. Новые проверки R1–R5 сначала red, затем green; интеграция и public R6/R7 — после неё. Протоколы в logs/; AST/импорт/окружение сами по себе не доказательство красного теста.

Core red:5поведенческихпаденийunknownrecipe/defaultinheritance/stalefailure/recovery +audit/hiddentool/JSONarray/hostdiagnostics. Greenroot/acceptance10passed13.09s. Промежуточныйfullsuite225passed1failed: compactcode схематратитьслишкоммноготокеновпослемetadata; устраненодублированиеbatchschemaвcompactcodeрежиме, acceptanceвосстановленбездобавлениятокенбюджета. НовыйполныйsuiteпослеR8-R14.

## CV и новые defaults
- Новые validation-plan tests: исходно 5 поведенческих падений на NotImplementedError, затем 5 passed.
- Config defaults: исходно 2 падения (fixed six trials / нет CV), затем 15 passed вместе с validation-plan.
- CV worker: исходно неизвестный evaluate_cv и отсутствие preflight artifact; после реализации 36 passed с training/plan/config.
- Host OOF/probability checks: записанный RED `20261020T084143.407205Z-red.json`: два ожидаемых падения (перевёрнутые OOF индексы и probability >1). Исправлены; focused CV/config/plan/research tests проходят.
- Последняя сводная проверка: `tests/test_acceptance.py tests/test_research_cv_agent.py tests/test_recipe_agent.py tests/test_budgeting.py -q`: 47 passed, 18.59s. Старые acceptance проверяют явно выбранный legacy holdout; новые research tests проверяют CV5, четыре семейства, реальные fit weights и model reload.

## Финальный ансамбль и веса
- Полный suite до финального host handoff: 324 passed, 2 известных Linear convergence warnings, 85.05s; JSON log `20261020T084510.561859Z-green.json`.
- Новый RED `20261020T084659.205261Z-red.json` подтвердил потерю weight-only опыта из-за code-hash dedup. Исправление учитывает recipe hash, включающий fit_options.
- Host blending tests исходно падали на stub; теперь проверяют OOF-полусумму двух противоположных ошибок, несовместимый split и отказ при нехватке refit времени. Численная погрешность CSV проверяется с tolerance1e-12.
- Final ensemble helper: 9 passed; root интеграция `test_timefirst/test_ensemble_selection/test_final_ensemble/test_research_cv_agent`: 14 passed, 7.72s. Дополнительный test на единый shortlist/reserve.
- Default trial timeout увеличен до 1800s, чтобы оставить сильным пятифолдным моделям время на обучение. Исследование ограничено deadline и оценкой возможности вместить следующий опыт; число неулучшающих гипотез само по себе его не завершает.

## Полная интеграция и live protocol
- `pytest -q`: 337 passed, 2 прежних warnings, 76.63s; log `20261020T085213.071401Z-green.json`.
- `scripts/verify_run.py` на synthetic CV run независимо пересчитал все 6 метрик, проверил worker overlap, полный refit и совпадение загруженной модели с submission; exit0.
- Actual Ollama wire выявил double-encoded experiments + flat decision fields + один лишний закрывающий символ внутри строки. Первые два вызова были отклонены до fit; дополнительное исправление decoder проверяет точный saved wire без обучения. При этом параметры научно неверной гипотезы не изменяются и по-прежнему проверяются recipes.

## Компактная история и остановка по времени
- History/provider/research integration: 14 passed, 20.17s. Параметры и fit_options сохраняются в последней сбалансированной assistant/tool группе, подробные diagnostics не дублируются в tool result; оригинальные события не изменяются.
- RED `20261020T092432.805172Z-red.json`: восемь неулучшающих опытов всё ещё завершали default run при 1200s остатка. Удалён этот косвенный trial limit; finish допустим, когда по измеренной длительности следующий опыт уже не помещается перед резервом. Green timefirst/history/providers: 12 passed, 5.56s (`20261020T092441.357743Z-green.json`).
