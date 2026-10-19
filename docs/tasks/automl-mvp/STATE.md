# Состояние

Этап: MVP реализован, проверен и подготовлен к передаче, 19.10.2026.

## Решения

CLI; приоритет надёжного harness; срок пользователя 26.10.2026; максимум 3 часа на запуск; численного порога метрик нет. Локальная LLM — Ollama, будущий API подключается baseline env. Изменения интегрированы в aprelskiy_branch после review. Агенты GPT-6 Luna xhigh работали в изолированных worktree; завершённые worktree отправлены в recoverable архив.

## Выполнено

- Созданы CASE_SUMMARY.md, DEEPSEEK_HARNESS_SUMMARY.md и reliability-исследование; первоисточники проверены ведущим. Upstream pin 687f282e8625c6b8d11ff58211a90524be2e4317.
- Сохранены SPEC/PLAN/CONTRACTS/TESTS. Исходные baseline.py и requirements.txt сохранены без изменения содержания.
- Реализованы generic discovery, профиль, фиксированный split, генерируемый build_model, worker и независимый evaluator/validator.
- Реализованы provider adapters, парный tool protocol, bounded context, journal, budgets, supervisor, ранний CSV, выбор победителя, final refit, resume и отчёт.
- Независимое review выявило и подтвердило исправления crash windows учёта usage/ответа и публикации победителя. Для них есть red→green regression tests.
- Проверены отдельная загрузка динамической модели, binary F1, ведущие нули ID, выбор основной метрики и undefined auxiliary r2.
- 65 тестов passed, exit 0. Оба полных public-прогона через qwen3.5:9b завершены с валидными CSV; итоговая версия дополнительно проверена на новой схеме.
- ZIP проверен allowlist-тестами и настоящим offline запуском после распаковки с другой cwd.
- Результаты: docs/VALIDATION.md и docs/validation/2026-10-19.json. Локальные журналы/кандидаты/модели: runs/.

## Следующий шаг пользователя

Запуск по README. Когда появится API организаторов — указать env и выполнить платформенный smoke; для leaderboard самостоятельно отправить dist/solution.zip. Push и сабмит не выполнялись.

## Ограничения

Qwen API недоступен; Docker CLI не установлен. Не подтверждены leaderboard score и неизвестные private-форматы. Эти ограничения не блокируют согласованную локальную приёмку MVP.
