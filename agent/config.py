import os
import time

START = time.time()
TIME_LIMIT = float(os.environ.get("AGENT_TIME_LIMIT_SEC", 3 * 3600))
_SAFETY = float(os.environ.get("AGENT_SAFETY_SEC", 15 * 60))
SAFETY = min(_SAFETY, 0.15 * TIME_LIMIT)
N_JOBS = int(os.environ.get("AGENT_N_JOBS", os.cpu_count() or 8))
SEED = 42
LLM_MAX_CALLS = int(os.environ.get("AGENT_LLM_MAX_CALLS", 30))
LLM_TIMEOUT = float(os.environ.get("AGENT_LLM_TIMEOUT", 180))


def elapsed() -> float:
    return time.time() - START


def remaining() -> float:
    """Секунды до мягкого дедлайна (с запасом SAFETY до жёсткого лимита)."""
    return START + TIME_LIMIT - SAFETY - time.time()


def log(*args):
    print(f"[{elapsed():7.1f}s]", *args, flush=True)
