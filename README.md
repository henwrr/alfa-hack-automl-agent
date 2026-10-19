# AutoML MVP

CLI-агент для исследования supervised-табличных задач из CSV. Он читает `task.md`, строит baseline, проверяет гипотезы на фиксированной локальной валидации и записывает submission. В MVP заданы регрессия и классификация; неизвестные форматы файлов и задачи без размеченного train CSV не поддерживаются.

## Установка

Нужен Python 3.12 и [uv](https://docs.astral.sh/uv/). Для локальной среды:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-mvp.txt
```

Для тестов установите отдельную dev-зависимость:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

`requirements-mvp.txt` — необходимое подмножество закреплённых версий из организаторского `requirements.txt`. Исходный `requirements.txt` не изменён. В ZIP включается только runtime-подмножество.

## Запуск с LLM

OpenAI-compatible endpoint настраивается через совместимые с baseline переменные окружения:

```bash
export AUTOML_PROVIDER=openai
export OPENAI_API_KEY='ваш ключ'
export OPENAI_MODEL='имя-модели'
export OPENAI_BASE_URL='https://адрес-endpoint/v1'

.venv/bin/python solution.py /path/to/task.md \
  --time-budget 10800 --trial-timeout 600 --max-trials 20 \
  --max-steps 60 --max-tokens 150000 --llm-timeout 180 --n-jobs 4
```

Для локальной **native Ollama** укажите тот же набор baseline-переменных и выберите провайдер:

```bash
export AUTOML_PROVIDER=ollama
export OPENAI_API_KEY=ollama
export OPENAI_MODEL='qwen3.5:9b'
export OPENAI_BASE_URL='http://127.0.0.1:11434'

.venv/bin/python solution.py /path/to/task.md --time-budget 10800
```

Ollama должна быть установлена и запущена отдельно; нужная модель должна быть загружена заранее. Режим `ollama` использует native endpoint, не OpenAI-совместимый `/v1`. Шаблон значений находится в [.env.example](.env.example); приложение само не загружает `.env`, поэтому экспортируйте переменные в shell либо передайте их через менеджер секретов.

Путь к задаче можно дополнить параметрами:

| Параметр | Назначение |
|---|---|
| `--time-budget SECONDS` | общий лимит запуска |
| `--trial-timeout SECONDS` | лимит одного эксперимента |
| `--max-trials N` | предел числа опытов, включая baseline и неудачные кандидаты |
| `--max-steps N` | предел шагов LLM-loop |
| `--max-tokens N` | суммарный бюджет входных и выходных токенов; по умолчанию 100000 |
| `--response-tokens N` | максимум выходных токенов одного ответа; по умолчанию 3072 |
| `--llm-timeout SECONDS` | лимит одного вызова LLM с повторами |
| `--n-jobs N` | число потоков для обучения |
| `--run-dir PATH` | каталог журнала, состояния и артефактов |
| `--resume` | продолжить запуск из указанного `--run-dir` |
| `--offline` | smoke/fallback без вызова LLM |
| `--seed N` | seed разбиения и воспроизводимых шагов |
| `--target NAME` | явное имя целевого столбца |
| `--task-type regression\|classification` | явный тип задачи |
| `--metric NAME` | явная метрика |

Для платформенного лимита в 3 часа задавайте `--time-budget` не выше `10800` секунд; более низкий лимит оставляет время на завершение финальной записи. Возобновляйте задачу с теми же данными, настройками и каталогом `--run-dir`; состояние сверяет fingerprint входов и конфигурацию split.

Перед запросом бюджет проверяется по консервативной UTF-8-оценке сообщений, tool schemas и служебного обрамления. После ответа учитывается usage провайдера; при его отсутствии и при прерванном запросе используется оценка. Денежный лимит команды зависит от тарифа будущего API. Сохранённый ответ и его usage восстанавливаются из журнала без повторной оплаты запроса; начатый Python trial с неизвестным исходом автоматически не повторяется.

По умолчанию submission записывается как `submission.csv` рядом с `task.md`. В `--run-dir` сохраняются `events.jsonl`, `state.json`, `task.json`, `profile.json`, `split_manifest.json`, `experiments.json`, `report.json`, `report.md` и каталоги `trial_0001/` и далее с кодом, manifest, логом процесса и предсказаниями. Отчёт содержит статус, выбранный опыт, метрики и пути к результатам. Передавайте API-ключи через переменные окружения; не помещайте их в `task.md`, аргументы CLI или код.

## Smoke и проверки

`--offline` явно запускает локальный baseline без запросов к LLM. Он подходит для быстрой проверки обнаружения CSV, CLI и записи результата; он не проверяет исследовательский цикл и не считается запуском AutoResearch.

Чтобы проверить обе публичные задачи локально, положите предоставленные CSV рядом с их `task.md`, затем запустите:

```bash
.venv/bin/python scripts/smoke_public.py \
  'Регрессия/task.md' 'Многоклассовая классификация/task.md'
```

Скрипт передаёт CLI только `--offline`, ограничивает smoke бюджетом и числом опытов, удаляет настройки провайдеров из дочернего окружения и складывает артефакты в `runs/public-smoke/`. Данные в репозиторий и ZIP не включаются.

Acceptance-проверка отправляет subprocess CLI синтетический CSV и использует локальный fake OpenAI-compatible HTTP server. Сервер возвращает код с ошибкой, затем исправленную гипотезу и `finish_research`; проверяются submission, отчёт и порядок событий восстановления. Suite не запускает Ollama и не требует живого API.

## Сборка ZIP и Docker

Платформенный архив создаётся в `dist/solution.zip`:

```bash
.venv/bin/python scripts/package_solution.py
```

В корне ZIP находится `solution.py`, затем runtime-пакет `automl_agent`, README, пример окружения и `requirements-mvp.txt`. Упаковщик использует allowlist и стабильные timestamp, порядок и права ZIP-записей. Данные, `.env`, `.venv`, тесты, результаты запусков и исследовательские материалы в него не попадают.

Dockerfile фиксирует Python 3.12 slim и runtime-зависимости; entrypoint запускает `solution.py`. Пример после подготовки локального `.env`:

```bash
docker build -t automl-agent:local .
docker run --rm --env-file .env \
  -v "$PWD/Регрессия:/task" automl-agent:local \
  /task/task.md --run-dir /task/runs/container-run
```

Docker — дополнительная среда; запуск контейнера и сборка образа пока не проверялись на этой машине, потому что Docker CLI здесь недоступен.

## Проверенный статус

**65 тестов проходят.** Обе полные публичные задачи завершены через Ollama qwen3.5:9b: валидные submission.csv, реальные LLM-гипотезы и Python-эксперименты, восстановление после ошибок, сохранённые и повторно загружаемые модели. Итоговая версия также проверена живой Ollama на новой синтетической схеме и запуском распакованного ZIP из другой директории.

Локальная holdout-валидация: RMSE 46.06155 для регрессии и Macro F1 0.56502 для классификации. В этих прогонах лучшим оказался встроенный baseline; более слабые LLM-кандидаты не заменили его. Это не метрики скрытого теста платформы. Подробности, команды и границы — в [docs/VALIDATION.md](docs/VALIDATION.md) репозитория.

API организаторов и Docker пока не проверены. MVP поддерживает supervised tabular CSV с выводом целевых значений или меток классов; неизвестные private-форматы и probability/multicolumn submissions требуют расширения. Supervisor и AST lint не заменяют OS security sandbox; жёсткие ограничения файлов, сети и суммарных ресурсов обеспечивает окружение запуска.

Исследования и план сохранены в репозитории и не входят в runtime ZIP: [CASE_SUMMARY.md](CASE_SUMMARY.md), [DEEPSEEK_HARNESS_SUMMARY.md](DEEPSEEK_HARNESS_SUMMARY.md), [план](docs/tasks/automl-mvp/PLAN.md), [контракты](docs/tasks/automl-mvp/CONTRACTS.md).
