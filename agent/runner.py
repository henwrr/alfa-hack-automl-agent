import traceback

from .config import log
from .llm import LLM


def main(task_path) -> int:
    spec = ds = None
    llm = LLM()
    try:
        from .task import parse_task
        from .data import Dataset
        from .submission import write_fallback
        from .research import Researcher

        spec = parse_task(task_path, llm)
        ds = Dataset(spec)
        write_fallback(spec, ds)
        Researcher(spec, ds, llm).run()
        log("Done.")
    except Exception:  # noqa: BLE001
        log("FATAL:\n" + traceback.format_exc())
        if spec is not None and ds is not None and not spec.output_path.exists():
            try:
                from .submission import write_fallback
                write_fallback(spec, ds)
            except Exception:  # noqa: BLE001
                log("fallback failed:\n" + traceback.format_exc())
    return 0
