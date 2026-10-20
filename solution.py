"""Точка входа агента.

Запуск (как в baseline.py):
    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... python solution.py task.md

Результат: submission.csv (путь берётся из task.md, по умолчанию рядом с task.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.runner import main  # noqa: E402

if __name__ == "__main__":
    task_path = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")
    sys.exit(main(task_path))
