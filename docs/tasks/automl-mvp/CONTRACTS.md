# Интерфейсы реализации

Эти интерфейсы согласованы оркестратором после чтения CASE_SUMMARY.md, DEEPSEEK_HARNESS_SUMMARY.md и reliability-исследования. Исполнители не меняют чужие файлы. Отклонения сначала согласуются с оркестратором.

## Общие типы

`automl_agent/contracts.py`: TaskSpec, ExperimentResult, ProcessResult, ModelResponse. JSON-совместимые dataclasses; пути — абсолютные строки. TaskSpec.model_context() исключает пути исходных данных и метки валидации.

## Данные, worker и evaluator (исполнитель ml_engine)

- `data.discover_task(task_path: Path, seed=42, validation_fraction=0.2, overrides: dict | None=None) -> TaskSpec` — определить пути/таргет/id/тип/метрику из task.md, словаря, sample и схем. Поддержать CSV рядом с markdown и существующие относительные ссылки из markdown; неоднозначность отвергать. Не использовать конкретные имена публичных целей. Overrides допускают target/task_type/metric.
- `data.profile_task(task: TaskSpec) -> dict` — компактный профиль, категории, пропуски, распределение цели, shape; не включать весь CSV.
- `data.prepare_split(task: TaskSpec, run_dir: Path) -> dict` — фиксированный детерминированный split, stratified при возможности; сохранить индексы и fingerprint. Возвращаемые ключи: `train_features`, `train_target`, `validation_features`, `validation_target`, `test_features`, `full_train_features`, `full_train_target`, `split_fingerprint`. Значения файлов — абсолютные пути CSV. CSV target содержит один столбец с именем target. Таргет не входит в feature CSV, ID не используется как feature.
- `modeling.baseline_code(task: TaskSpec, n_jobs=4, iterations=150) -> str` — исполняемый Python с `build_model(context: dict) -> estimator`. Устойчивый baseline (CatBoost при доступности, разрешённый sklearn fallback), без привязки к данным, корректно обработать категории/NaN/inf. Этот код — начальная точка, LLM может полностью менять estimator/pipeline и создавать собственные преобразования.
- Worker запускается `python -m automl_agent.worker --manifest /abs/manifest.json`. Manifest: `mode` (`evaluate`/`finalize`), `context` (TaskSpec.model_context), `inputs` (только нужные train_features/train_target/validation_features/test_features), `code_path`, `artifact_dir` и `save_model` (bool). В режиме finalize train_features/train_target указывают на full_train. В manifest/контекст worker не передаётся validation_target.
- Worker импортирует кандидатный code_path и вызывает build_model(context); затем сам делает fit/predict. Выходы: `valid_predictions.csv` (evaluate), `test_predictions.csv` (оба режима), по одному столбцу target; для probability-метрик также `valid_probabilities.npy`, `probability_classes.json`. Дополнительно `model.joblib`/cloudpickle и `worker_result.json` с диагностикой. Ошибки -> ненулевой exit code и stderr. Нельзя доверять worker-метрике: scoring всегда в host.
- `evaluation.score_trial(task: TaskSpec, split: dict, trial_dir: Path) -> dict[str, float]` читает validation_target и предсказания, считает primary metric под ключом task.metric + справочные метрики. Не принимает скор, написанный моделью. Обязательные RMSE/macro_f1; при разумной стоимости поддержать mae/r2/accuracy/f1/roc_auc/log_loss.
- `evaluation.publish_submission(task: TaskSpec, predictions_path: Path, destination: Path) -> dict` проверяет размер, колонки, порядок test IDs, классы/числа/NaN/inf, затем атомарно публикует CSV. Возвращает сводку. Публичная `validate_submission(task: TaskSpec, path: Path) -> dict` повторно проверяет существующий файл. При ошибках raise ValueError и не заменять рабочий output.
- Собственные тесты: `tests/test_data.py`, `tests/test_worker.py`, `tests/test_evaluation.py`.

## Provider, journal, supervisor (исполнитель harness_runtime)

- `providers.OpenAICompatibleProvider(base_url: str, api_key: str, model: str, retries=2, extra_body: dict | None=None)` и `OllamaProvider(base_url: str, api_key: str, model: str, retries=2, extra_body: dict | None=None)`.
- Оба: `.complete(messages: list[dict], tools: list[dict], *, timeout: float, max_tokens: int) -> ModelResponse`. Канонический ответ OpenAI message: role/content/tool_calls; function.arguments всегда JSON-строка, каждый call имеет ID. У native Ollama endpoint `/api/chat`, stream=false, think=false, tools; нормализовать dict-arguments, usage. У OpenAI-compatible endpoint `/chat/completions`, tools, parallel_tool_calls=false. Native Ollama base_url может заканчиваться /v1 (нормализовать). `extra_body` — явные специфичные поля конфигурации. Не слать Ollama-опции в чужой endpoint.
- `providers.ProviderError`: `.code: str`, `.retryable: bool`; конечные retries только transient. timeout — весь вызов с повторами и backoff, не независимый таймаут каждой попытки. 401/403/400 не повторять, context overflow отличать. Никаких утечек ключей в error message.
- `journal.EventJournal(path: Path, secrets: list[str] | tuple[str,...]=())`: `.append(event: str, **payload) -> dict`, `.read_events() -> list[dict]`; seq/timestamp/event поля; flush+fsync перед возвратом; поддержать корректную обработку оборванной последней JSONL-строки, отказ на corruption в середине. Маскировать секреты в payload и тексте. Закрытые события остаются append-only.
- `journal.atomic_write_json(path: Path, value: dict) -> None` — atomic replace/fsync.
- `runtime.Budget(total_seconds: float, reserve_seconds: float=60, spent_seconds: float=0)`: `.remaining()`, `.research_remaining()`, `.elapsed()` -> seconds (monotonic), spent сохраняется при resume. Не давать новой итерации съесть final reserve.
- `runtime.run_process(argv: list[str], *, cwd: Path, timeout: float, n_jobs: int=4, max_output_bytes: int=131072, memory_limit_mb: int | None=None, env: dict | None=None) -> ProcessResult`. Свежий process group, без shell, ограниченные stdout/stderr во время чтения, kill group+cleanup при timeout/output limit/memory cap, минимальное env без ключей, BLAS/OpenMP limits. PYTHONPATH передаётся явно из host. Status: `ok`, `error`, `timeout`, `output_limit`, `memory_limit`, `cancelled`. Не заявлять OS sandbox там, где его нет.
- Собственные тесты: `tests/test_runtime.py`, `tests/test_providers.py`, `tests/test_journal.py`.

## CLI, loop и интеграция (ведущий агент)

`solution.py`, `automl_agent/config.py`, `automl_agent/agent.py`, `automl_agent/prompts.py`, `automl_agent/tools.py`: baseline CLI+env; run/resume/state; реальный tools protocol; фиксированный evaluator; early submission и final full refit; отчёт. Инструменты `run_experiment(hypothesis, code)`, `finish_research(reason)`; текстовый JSON fallback для endpoint без native tool calls. Логировать intent до эффекта и result после, включая ошибки/прерванные вызовы. Ограничить дубликаты code hash, шаги, токены, время. Провайдер должен получить результаты опыта; код лучшего и ошибки доступны для следующей гипотезы. Небольшой bounded контекст из durable experiment ledger вместо неограниченного диалога.

## Упаковка и приёмка (исполнитель delivery)

`scripts/package_solution.py`, `scripts/smoke_public.py`, `requirements-mvp.txt`, `requirements-dev.txt`, `.env.example`, `README.md`, `Dockerfile`, `.dockerignore`, `tests/test_packaging.py`, `tests/test_acceptance.py`.

CLI: `python solution.py [task.md] [--time-budget SECONDS] [--trial-timeout SECONDS] [--max-trials N] [--max-steps N] [--max-tokens N] [--llm-timeout SECONDS] [--n-jobs N] [--run-dir PATH] [--resume] [--offline] [--seed N] [--target NAME] [--task-type regression|classification] [--metric NAME]`. Значения env OPENAI_API_KEY/OPENAI_MODEL/OPENAI_BASE_URL сохраняют контракт. `AUTOML_PROVIDER=ollama` выбирает native Ollama, `openai` — default. `--offline` явно означает детерминированный smoke/fallback без LLM; никогда не выдавать его за агентный режим.

Основная проверка без доступа к Qwen — fixture HTTP-server с OpenAI Chat Completions contract: tool call с валидным build_model кодом, ошибка кода, следующая исправленная гипотеза, finish; запуск subprocess solution.py и проверка submission/journal/отчёта. Не вызывать Ollama из unit suite. Реальные публичные прогоны делает ведущий. Архив содержит только runtime и документацию запуска, исключает данные/ключи/тесты/окружение/research logs. Docker — дополнительная воспроизводимая среда; отсутствие локального Docker описывается честно.
