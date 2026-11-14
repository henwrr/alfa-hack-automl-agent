# Универсальный AutoResearch агент

> **Командный проект** хакатона Альфа Хак, кейс «AI-агент для разработки ML-моделей». Перенесён из командного репозитория [aprelskiyegor/alfa_hackathon](https://github.com/aprelskiyegor/alfa_hackathon)
> 
> Финальную версию собрали на основе прототипов участников.

Готовая реализация: [universal_agent](universal_agent/README.md). Агент исследует табличные задачи регрессии и классификации через локальную Ollama или выданный платформой API и сохраняет проверенный submission.csv рядом с task.md.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r universal_agent/requirements-mvp.txt
.venv/bin/python universal_agent/solution.py /path/to/task.md
```

Без ключа используется запущенная локально Ollama с qwen3.5:9b. Установка модели выполняется отдельно.

- [Подробный разбор условий](universal_agent/docs/HACKATHON_REQUIREMENTS.md)
- [Плюсы и минусы всех четырёх агентов](universal_agent/docs/AGENT_REVIEW.md)
- [Повторный аудит обновлённого Ruzik и перенесённые идеи](universal_agent/docs/reviews/ruzik_update_85f4988.md)
- [Какие идеи из каких файлов перенесены](universal_agent/docs/PROVENANCE.md)
- [Результаты тестов и ограничения](universal_agent/docs/VALIDATION.md)

Исходный requirements.txt организаторов сохранён. Локальные all_agents, датасеты, PDF, модели, ключи, журналы и ZIP не публикуются; SHA256 исходных файлов сохранены в [инвентаризации](universal_agent/docs/SOURCE_INVENTORY.json).
