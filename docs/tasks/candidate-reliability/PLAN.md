# План

1. Разобрать предыдущие trials/context/realized params; baseline160 tests. Зафиксировать ROOT_CAUSES.md с доказательствами.
2. Параллельно в изолированных worktree: (a) безопасные model recipes + тесты; (b) фактические fitted parameters + тесты; (c) forensic review старых гипотез и official docs.
3. Root: тесты интеграции до кода, recipe tool/ledger/fingerprint, strategy/context/defaults, совместимость code tools. Интегрировать commits исполнителей и проверить.
4. Полный suite, live Ollama на regression/classification последовательно, исправления по фактическим проблемам, re-score/refit/reload/resume и независимое review.
5. Обновить README/validation evidence, собрать ZIP, commits/push в aprelskiy_branch, remote SHA.

Границы файлов и контракты — CONTRACTS.md. Root единственный писатель ledger, evaluator/submission не ослабляются.

## Сужение до промежуточного MVP, 21 октября
Пользователь попросил довести текущую работу до MVP. Новое исследование регрессии отменено: используем 14 сохранённых успешных CV trials; классификация завершает текущий набор. Минимальный режим --resume --finalize-only позволяет безопасно добавить итоговый refit к завершённому/прерванному запуску, не повторяя CV и не меняя canonical events. Далее независимая проверка обеих моделей/CSV, no-op resume, финальный suite, ZIP и push. Новые исследовательские возможности за пределами этого этапа.
