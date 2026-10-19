"""Platform entry point: python solution.py /path/to/task.md."""

from __future__ import annotations

import json
import sys

from automl_agent.config import parse_config


def main(argv=None):
    config = parse_config(argv)
    try:
        from automl_agent.agent import run

        report = run(config)
    except KeyboardInterrupt:
        print("Interrupted; use --resume with the same --run-dir", file=sys.stderr)
        return 130
    except Exception as exc:
        message = str(exc)
        if config.api_key:
            message = message.replace(config.api_key, "[REDACTED]")
        print(f"{type(exc).__name__}: {message}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
