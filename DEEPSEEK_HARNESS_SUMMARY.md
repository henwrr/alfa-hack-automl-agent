# DeepSeek Harness: архитектурная выжимка для AutoResearch

Срез исходников: `deepseek-ai/deepseek-harness`, ветка `master`, commit [`687f282e8625c6b8d11ff58211a90524be2e4317`](https://github.com/deepseek-ai/deepseek-harness/commit/687f282e8625c6b8d11ff58211a90524be2e4317). Проверено по клону `/tmp/deepseek-harness-automl-architecture-20261019`. README и исходники сверялись на этом commit, а ссылки ниже закреплены за ним.

## Короткий вывод

Для AutoResearch полезнее всего перенести четыре решения: один канонический формат сообщений и tool calls между моделями; явный цикл «ответ модели → инструменты → результаты → следующий запрос»; долговременный журнал, из которого восстанавливается контекст; и compaction, которая сохраняет важное состояние, когда журнал перестаёт помещаться в окно модели. DeepSeek Harness доводит эти идеи до большого плагинного приложения. Для MVP стоит взять его границы модулей и контракты, а не всю Cordis-инфраструктуру.

## Устройство Harness: факты по README и коду

### Агентский цикл

`dsh-agent` задаёт публичный контракт создания, продолжения и отмены агента; конкретный драйвер `dsh-agent-loop` подключается как реализация. Внутри цикла один ход (`turn`) состоит из шагов (`step`): перед шагом собираются prompt и доступные tool schemas, вызывается модель, ответ сохраняется, а tool calls исполняются; их результаты попадают в историю для следующего шага. Внешние обработчики могут вклиниться в admission шага и обработку ошибки запроса. Это не видно только из описания: `ReactLoopAgent.turn()` открывает границу хода и вызывает `preStep()`, а `step()` формирует запрос, сохраняет ответ и передаёт найденные вызовы в `executeToolCalls()` ([`agent.ts`: `preStep`, `turn`, `step`, `prepareRequest`, `buildRequest`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/src/agent.ts#L267-L287), [`agent.ts`: turn/step flow](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/src/agent.ts#L298-L398), [`agent.ts`: request/tool dispatch](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/src/agent.ts#L400-L561)).

Публичные сообщения модели унифицированы: assistant tool call содержит ID, имя и аргументы как JSON-строку; стрим адаптера преобразуется в общие блоки `tool-call-delta`, `text-delta`, `usage`, затем в `finish`. Сетевой формат конкретного провайдера переводит только adapter. После успешного ответа loop сохраняет assistant message и пару `tool/call` / `tool/result` для каждого вызова; ошибка инструмента становится результатом с `isError`, а не неявно пропадает из истории ([`types.ts`: `ToolCallBlock`, `StreamChunk`, `ToolSchema`, `GenerateOptions`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm/src/types.ts#L104-L110), [`types.ts`: stream and tool protocol](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm/src/types.ts#L462-L559), [`agent.ts`: сохраняемый ответ и запуск вызовов](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/src/agent.ts#L529-L555), [`tool-calls.ts`: запись вызовов и результатов](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/src/tool-calls.ts#L262-L289)).

Параллельность у Harness объявляется самой capability: вызовы без явного разрешения на безопасный параллельный запуск считаются exclusive. Scheduler ограничивает пул, сохраняет результаты в исходном порядке модели, exclusive-вызовами ставит барьеры, а при отмене перестаёт запускать новые вызовы и закрывает незапущенные пары синтетическими результатами ([`ToolDefinition.isConcurrencySafe` и `ToolExecutionMode`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/tools/src/index.ts#L223-L281), [`executeToolCalls` и `runGroup`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/src/tool-calls.ts#L60-L109), [`runGroup`: bounded pool and ordered commits](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/src/tool-calls.ts#L122-L247)).

### Tool protocol и граница capability

У инструмента отдельно объявлены модельно-видимая схема и исполняемый код. `ToolSchema` — это имя, описание и JSON Schema параметров. `ToolDefinition` добавляет `execute`, типизированное описание результата, рендеринг ответа, timeout и необязательную метку concurrency-safe. В wire schema не уходят `execute`, политики и внутренние callbacks. Runtime проверяет аргументы, прогоняет вызов через pre-execute policy/guard, выполнение и обработку результата; неизвестный инструмент или исключение возвращается как структурированная ошибка, после которой цикл может продолжаться ([`tools/README.md`: контракт и pipeline](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/tools/README.md#L32-L36), [`ToolDefinition`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/tools/src/index.ts#L213-L281), [`ToolRuntime.register`, `schemas`, `executionMode`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/tools/src/index.ts#L1065-L1095), [`tools/README.md`: pipeline и ошибки](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/tools/README.md#L107-L130)).

Для AutoResearch это особенно подходит к одному Python-инструменту: модель выбирает его по схеме, передаёт код или декларативную команду, но политика и фактический исполнитель принадлежат приложению. Если MVP использует только `run_python`, отдельная генерация SDK, PTC и развитая система scope не нужны. Внутренний результат выполнения полезно хранить структурно, а в контекст модели возвращать короткий текст и ссылки на важные артефакты.

### Prompt и контекст

`dsh-system-prompt` собирает упорядоченные секции, переменные, runtime contexts и schemas в одну `PromptAssembly`. Секции становятся system prompt; динамические runtime facts добавляются отдельными user-role snapshots. Registry доступных инструментов подаёт схемы в тот же сборщик, чтобы заявленные возможности соответствовали доступным. Неизвестные переменные считаются ошибкой, а порядок инструментов канонизируется ([`system-prompt/README.md`: регистрация секций и контекста](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/system-prompt/README.md#L54-L86), [`PromptSection`, `PromptContext`, `PromptAssembly`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/system-prompt/src/index.ts#L52-L129), [`assemble` / `renderPrompt`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/system-prompt/src/index.ts#L290-L340), [`SystemPrompt.assemble`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/system-prompt/src/index.ts#L582-L640)).

Такое разделение решает две разные задачи: system instructions остаются относительно стабильными, а меняющиеся данные окружения можно обновлять на каждом шаге. Для AutoML в изменяемый контекст стоит передавать спецификацию задачи, цель/метрику, доступные dataset/artifact paths, ограничения среды, текущий лучший результат и список недавних опытов. Большой schema и полные таблицы результатов лучше оставлять доступными через Python tools, а не повторять в каждом prompt.

### История, память и compaction

Session — append-only журнал событий. Запись события создаёт структурную историю; `deriveMessages()` строит из неё текущие model messages. Заменённые при compaction сообщения остаются в полном журнале, но не попадают в активную поверхность контекста. JSONL persistence подключается отдельно и хранит журнал по сессиям ([`session/README.md`: события и проекция сообщений](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/session/README.md#L12-L52), [`session/README.md`: event sourcing и реконструкция запроса](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/session/README.md#L100-L131), [`session-persistence-jsonl/README.md`: append-only JSONL backend](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/session/session-persistence-jsonl/README.md#L10-L18)).

Compaction включается до превышения окна и после подтверждённой ошибки переполнения. По умолчанию бюджет учитывает окно модели, output cap и headroom; старый сбалансированный диапазон сворачивается в checkpoint, хвост оставляется verbatim. Удаление целого вызова инструмента без парного результата не допускается. Сначала может сработать дешёвое сжатие слишком длинных результатов инструментов. Summarizer получает system prompt, доступные tools и выбранный участок истории, а затем одну инструкцию; в checkpoint попадает только текст. Граница compaction — замена поверхности, сам журнал не стирается ([`compaction-basic/README.md`: thresholds, retention, pruning](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/compaction/compaction-basic/README.md#L60-L90), [`compaction-basic/README.md`: политика и транзакция](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/compaction/compaction-basic/README.md#L104-L125), [`selectCompactableRange`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/compaction/compaction-basic/src/region.ts#L117-L155), [`commitCompactionBody`, `buildSummarizationInput`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/compaction/compaction-basic/src/region.ts#L471-L560), [`summarizeWithLlm`, `frameSummary`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/compaction/compaction-basic/src/summarizer.ts#L120-L194)).

Для AutoML обычной «сводки диалога» недостаточно: агенту нужно точно помнить лучший score, split/seed, параметры, путь к коду, версии данных и причины неудачных запусков. Рекомендация: хранить экспериментальную таблицу отдельно как машинно-читаемый source of truth, а в checkpoint класть небольшой снимок текущего состояния и следующие действия. Суммаризация должна ссылаться на журнал и артефакты, но не заменять их.

### Модель и inference abstraction

`dsh-llm` — нейтральный контракт одного вызова и общей схемы streaming messages; adapter владеет переводом запроса/ответа в API провайдера. `prepareCall()` разрешает параметры, defaults и capabilities модели в одном захваченном вызове, после чего stream отправляется через подготовленный adapter. Это предотвращает смесь конфигурации одного провайдера с результатом другого при смене настроек во время запроса ([`llm/README.md`: границы сервиса и адаптеров](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/README.md#L10-L35), [`LlmAdapter` и `PreparedLlmCall`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm/src/index.ts#L168-L302), [`agent.ts`: `prepareRequest` и bound call](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/src/agent.ts#L564-L613)).

В `dsh-llm-pi-ai` есть настройка ручных gateway-маршрутов: protocol, `baseURL`, model id и metadata можно объявить конфигурацией; среди поддержанных протоколов есть `openai-completions`, `openai-responses` и Anthropic Messages. Это даёт полезный образец seam для OpenAI-compatible endpoint и тестового локального сервера. Для AutoResearch стоит выбрать один внутренний интерфейс OpenAI-compatible, который можно настроить на Ollama во время локальной отладки и на Qwen endpoint при переходе в API. Особенности tool calling, лимита токенов, streaming и auth должны обрабатываться на границе провайдера, а loop не должен знать URL/заголовки endpoint-а ([`llm-pi-ai/README.md`: gateways и схема route](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-pi-ai/README.md#L28-L89), [`provider.ts`: таблица протоколов](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-pi-ai/src/provider.ts#L47-L62), [`provider.ts`: ручной маршрут](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-pi-ai/src/provider.ts#L167-L191), [`config.ts`: route configuration](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/llm-pi-ai/src/config.ts#L326-L354)).

## Карта переноса в AutoResearch

### Рекомендуемая структура MVP

```text
autoresearch/
  agent/loop.py          # один turn/step цикл и конечное число вызовов
  protocol.py            # Message, ToolCall, ToolSpec, ToolResult, FinishReason
  llm/base.py            # provider-neutral interface
  llm/openai_compatible.py # OpenAI-compatible chat/tool calls; base_url из config
  tools/registry.py      # schemas + handlers + валидация аргументов
  tools/python.py        # единственный capability исполнения Python
  context/prompt.py      # статические инструкции + свежие факты задачи
  context/checkpoint.py  # сверка/compact истории в checkpoint
  memory/session.py      # события одного запуска и derived messages
  experiments/ledger.py # метрики, параметры, seed, артефакты и статус опытов
```

Минимальный цикл: собрать system/context и schemas → вызвать provider → сохранить assistant response → если есть tool calls, выполнить их и записать парные результаты → повторить до финального ответа или жёсткого лимита шагов. Передавать в следующий вызов только нормализованный протокол, не сырой ответ Ollama/Qwen. В tool registry для MVP достаточно `python` с JSON Schema для полей, короткой description и явной разницей между успешным JSON-результатом и ошибкой. Начинать последовательное исполнение вызовов; `parallel_safe` добавить только тем операциям, совместное выполнение которых действительно не меняет состояние другого эксперимента.

**В AutoML-контексте Python tool полезно дать такой контракт:** код/операция, входные dataset/artifact references, объявленная цель; в ответ — `stdout` (ограниченный), статус, метрики, структура результатов, созданные файлы и краткая ошибка. Критерий обучения/валидации и расчёт метрики должен быть известен приложению. Модель может предложить эксперимент и анализировать результат, но логика сравнения «лучше/хуже» должна быть детерминированной.

### Приоритеты

**MVP:**

1. Один агент и одна модельная абстракция с настройкой endpoint/model/API key. Начальная конфигурация обслуживает Ollama, смена endpoint-а подключает Qwen без изменений loop.
2. Канонический JSON-протокол tool calls: call id, имя, arguments; парный результат с тем же id, успешным/ошибочным статусом и текстом плюс структурные поля для метрик и путей.
3. Registry из нескольких явно разрешённых инструментов, даже если сначала доступен один Python. Проверять параметры по схеме до вызова; передавать в prompt только объявленные инструменты.
4. Один event log на запуск, где сохраняются входная задача, ответы модели, tool calls/results, параметры опыта, метрики и artifact paths. Отдельный `experiments/ledger` — canonical state опыта; чат — объяснение и ход работы.
5. Короткий системный prompt с целью, правилами эксперимента, доступными данными, стоп-критериями и форматом итогового отчёта; изменяемые значения давать как текущий контекст.
6. Счётчики шагов/модельных вызовов, output tokens и общего времени. Для MVP лучше ограничивать количество экспериментов и явно возвращать агенту причину остановки.

**Следующие этапы:**

- Повтор запроса с типизированными transient/context-overflow ошибками; token-budget-aware compaction и восстановление сессии после рестарта.
- Структурированные checkpoints, сравнение candidate runs, provenance (data hash/version, split, random seed, code hash, package versions) и просмотр сгенерированных артефактов.
- Capability metadata модели: окно контекста, output limit, поддержка tools/JSON и stream. Учитывать поддержку endpoint-а явно, не выводить её из названия модели.
- Параллельные вызовы для чисто read-only операций и расширение registry/подсказок по мере добавления разных ML capabilities.
- Отдельные prompts/providers только тогда, когда появятся различные роли или endpoint-specific инструкции, которые нельзя выразить одним интерфейсом.

## Trade-offs

| Идея Harness | Что выигрываем | Цена | Вывод для MVP |
|---|---|---|---|
| Общий внутренний формат сообщений и ToolCalls | Не привязываем цикл к JSON Ollama/Qwen, единый парный call/result | Нужен перевод каждого endpoint-а в/из общего формата | Перенести сразу |
| JSON Schema для инструментов | Меньше неоднозначных аргументов, можно валидировать до исполнения | Схема расходует контекст и требует аккуратной поддержки | Перенести для каждого tool; не раздувать схемы |
| Append-only log + derived conversation | Можно восстановить запуск и отделить запись от текущего model context | Больше кода и данных, чем хранить последние сообщения | Упрощённый JSONL/event list уместен сразу; не делать миграционный framework |
| Tool result отдельно от display/result metadata | Сохраняются точные метрики/параметры, а model context можно сократить | Два вида представления ответа | Оставить в каноническом tool result только необходимые поля и краткий render |
| Compaction по цене запроса и balanced span | Контекст остаётся в пределах окна, вызовы инструментов не рвутся | Ещё один вызов модели; checkpoint может упустить детали | Включать после роста контекста; дополнять структурным experiment state |
| Плагинные prompt sections и capabilities | Владельцы capability могут добавлять собственные schemas и инструкции | Порядок и жизненный цикл плагинов усложняют систему | В MVP держать один prompt builder и один registry |
| Параллельные safe tools | Снижение задержки на независимых действиях | Гонки и переупорядочивание результатов | Стартовать последовательно; позднее opt-in на read-only |
| Общая provider abstraction и capability metadata | Локальная разработка и перенос между провайдерами | Общий формат не гарантирует одинаковых возможностей endpoint-ов | Перенести abstraction; отдельно описывать поддерживаемые возможности |

## Что не стоит копировать в MVP

- «Всё — плагин», Cordis, scoped plugin contexts, hot reload и provider directory. В Harness это общая основа десктопного/Web-приложения и огромного набора функций; для одного ML агента она замедлит MVP.
- Полный request/event vocabulary с durable headers, токен-кешем, compact stream frames, dynamic tool declaration updates и миграциями исторических форматов. Переносить принцип восстановления, а не все детали хранения.
- PTC режим, Python/TypeScript SDK генерацию, deferred tool loading, subagent coordination, approvals UI, отдельные интерфейсы и каталоги провайдеров. Эти части решают задачи расширяемой платформы и не нужны одиночному агенту с одним Python executor.
- Слепое суммирование всей истории. Для AutoML числовые метрики, параметры и артефакты должны оставаться в структуре и журнале, даже если диалог сжат.
- Предположение, что OpenAI-compatible значит полностью одинаковое поведение. Разные серверы могут отличаться в streaming/tool calling/лимитах; это должны отражать capabilities и адаптер.

## Лицензия и видимые ограничения

Корневой `LICENSE` указывает MIT. Лицензия разрешает использовать, копировать, менять, распространять и продавать копии при сохранении copyright и текста лицензии в копиях или существенных частях; в лицензии также есть отказ от гарантий и ответственности. README отдельно ссылается на [`THIRD_PARTY_NOTICES.md`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/THIRD_PARTY_NOTICES.md) с лицензиями зависимостей. При переносе кода надо проверять уведомления конкретных копируемых файлов и их зависимостей, а не считать весь граф зависимостей частью MIT ([`LICENSE`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/LICENSE#L1-L21), [`README.md`: статус preview и лицензия](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/README.md#L11-L15), [`README.md`: third-party notices](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/README.md#L73-L77)).

По README, проект находится в developer preview и допускает breaking changes. `SAFETY.md` называет его экспериментальным и не прошедшим security audit; явно предупреждает, что model-generated code/commands могут работать с предоставленными файлами, процессами, сетью и credentials, а sandbox/approval не являются гарантией изоляции. Это важно для AutoResearch, потому что центральный инструмент исполняет Python ([`SAFETY.md`](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/SAFETY.md#L5-L23)).

## Ключевые первоисточники

## Выжимка второго исследования: надёжность исполнения

Полное независимое исследование: [DEEPSEEK_HARNESS_RELIABILITY.md](docs/research/DEEPSEEK_HARNESS_RELIABILITY.md). Оркестратор прочитал оба исследования и сверил ключевые границы с upstream-кодом.

- Runtime отделяет исполнение от агента и завершает управляемые процессы при отмене. В Python MVP переносим отдельные процессы и управление process group; у каждого вызова есть более короткий timeout внутри общего трёхчасового дедлайна.
- Вывод ограничивается при чтении, а не после накопления в памяти. Бесконечная печать должна давать структурированный `output_limit`, не исчерпывать память host.
- Retry модели и повтор эксперимента различаются: только временные API-сбои допускают конечный backoff. Python exception возвращается LLM для осмысленной коррекции; незавершённая после crash попытка не перезапускается вслепую.
- Checkpoint сохраняет intent до внешнего эффекта и result до следующего шага. JSONL сам по себе не гарантирует exactly-once; для trial нужны уникальные IDs и проверка артефактов при resume.
- Канонический ML-score считает host на фиксированном validation-target; самооценка модели или поле score из её stdout не участвуют в выборе победителя. Перед публикацией CSV проверяется независимо и записывается атомарно.
- Изолированный subprocess и очищенное env не равны полноценному sandbox. В MVP явно описываются границы: внешнее Docker/OS окружение отвечает за жёсткую изоляцию файлов/сети и совокупный лимит ресурсов; supervisor — за timeouts, процессные группы, RSS/вывод и корректное завершение.

Для MVP эти меры включаются вместе с архитектурными контрактами. Полная плагинная среда DeepSeek не переносится; используется самостоятельная Python-реализация идей без копирования upstream-кода.

### Ссылки на архитектуру

- [README.md](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/README.md) — назначение, plugin architecture, preview и ссылки на архитектуру.
- [`packages/core/agent-loop`](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop) — README и код turn/step.
- [`packages/core/tools`](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/tools) — schemas, registry, execution lifecycle.
- [`packages/core/system-prompt`](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/system-prompt) — prompt and context assembly.
- [`packages/core/session`](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/session) — event log and model-message projection.
- [`packages/compaction/compaction-basic`](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317/packages/compaction/compaction-basic) — thresholding, range selection, checkpoint synthesis.
- [`packages/llm`](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm) — provider-neutral request/stream contract and OpenAI-compatible route.
