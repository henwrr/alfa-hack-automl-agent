"""Точка входа платформы: python solution.py path/to/task.md

LLM настраивается как в baseline.py: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL.
Бюджет времени: --time-budget или AGENT_TIME_BUDGET (по умолчанию 10200 с — почти весь лимит платформы
10800 с: платформа запускает без флагов; для разработки задавайте меньше, например --time-budget 600).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.runner import Agent  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AutoResearch-агент для табличных задач")
    parser.add_argument("task_path", nargs="?", default="task.md")
    parser.add_argument("--time-budget", type=float, default=float(os.getenv("AGENT_TIME_BUDGET", 10200)))
    parser.add_argument("--offline", action="store_true", help="без LLM, только эвристический плейбук")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--folds", type=int, default=int(os.getenv("AGENT_FOLDS", 2)),
                        help="число валидационных фолдов (временных периодов или повторов holdout)")
    args = parser.parse_args(argv)
    budget = min(args.time_budget, 10800 - 120)
    agent = Agent(args.task_path, time_budget=budget, seed=args.seed, n_jobs=args.n_jobs,
                  offline=args.offline, run_dir=args.run_dir, n_folds=args.folds)
    report = agent.run()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
