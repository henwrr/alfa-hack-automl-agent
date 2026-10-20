"""Разбор task.md: пути к данным, таргет, id, тип задачи, метрика, формат вывода."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import log
from .metrics import REGISTRY

# порядок важен: более специфичные шаблоны раньше
METRIC_PATTERNS = [
    ("rmsle", r"rmsle|root mean squared log"),
    ("rmse", r"\brmse\b|root mean squared error|корень из (средне)?квадрат"),
    ("mape", r"\bs?mape\b|mean absolute percentage"),
    ("mae", r"\bmae\b|mean absolute error|средн\w+ абсолютн"),
    ("mse", r"\bmse\b|mean squared error"),
    ("r2", r"\br2\b|r\^2|r²|коэффициент детерминации"),
    ("roc_auc", r"roc[\s\-_]?auc|\bauc\b|\bgini\b|джини"),
    ("logloss", r"log[\s\-_]?loss|логлосс|кросс[\s\-]?энтроп|cross[\s\-]?entropy"),
    ("macro_f1", r"macro[\s\-_]?f1|f1[\s\-_]?macro|average\s*=\s*['\"]macro"),
    ("weighted_f1", r"weighted[\s\-_]?f1|f1[\s\-_]?weighted"),
    ("micro_f1", r"micro[\s\-_]?f1|f1[\s\-_]?micro"),
    ("balanced_accuracy", r"balanced[\s\-_]?accuracy"),
    ("f1", r"\bf1\b|f-?мера|f-?score"),
    ("accuracy", r"\baccuracy\b|доля правильных"),
]


@dataclass
class TaskSpec:
    task_path: Path
    task_dir: Path
    text: str
    train_path: Path | None = None
    test_path: Path | None = None
    sample_path: Path | None = None
    dict_path: Path | None = None
    output_path: Path | None = None
    id_col: str | None = None
    target: str | None = None
    task_type: str | None = None  # regression | binary | multiclass
    metric: str | None = None
    proba_output: bool = False
    proba_columns: list = field(default_factory=list)
    submission_columns: list = field(default_factory=list)


def detect_metric(text: str) -> str | None:
    lines = [ln.lower() for ln in text.splitlines()]
    priority = [ln for ln in lines if re.search(r"основн\w* метрик|метрик\w* оценки|primary metric|evaluation metric|^#+.*метрик", ln)]
    metric_lines = [ln for ln in lines if re.search(r"метрик|metric|оценива", ln)]
    for group in (priority, metric_lines, lines):
        for ln in group:
            for name, pat in METRIC_PATTERNS:
                if re.search(pat, ln):
                    return name
    return None


def _resolve(task_dir: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (task_dir / p)


def _find_paths(spec: TaskSpec):
    text = spec.text
    candidates = re.findall(r"[`'\"(\s]([\w\-./\\ ]+?\.csv)[`'\")\s,;:]", " " + text + " ")
    for raw in candidates:
        raw = raw.strip()
        name = Path(raw).name.lower()
        p = _resolve(spec.task_dir, raw)
        if "sample" in name and spec.sample_path is None:
            spec.sample_path = p
        elif ("dictionary" in name or "description" in name) and spec.dict_path is None:
            spec.dict_path = p
        elif "train" in name and spec.train_path is None:
            spec.train_path = p
        elif "test" in name and spec.test_path is None:
            spec.test_path = p
        elif "submission" in name and "sample" not in name and spec.output_path is None:
            spec.output_path = p
    # Fallback: поиск по каталогу задачи
    def glob_one(pattern):
        hits = sorted(spec.task_dir.rglob(pattern), key=lambda x: len(str(x)))
        return hits[0] if hits else None
    if spec.train_path is None or not spec.train_path.exists():
        spec.train_path = glob_one("*train*.csv") or spec.train_path
    if spec.test_path is None or not spec.test_path.exists():
        spec.test_path = glob_one("*test*.csv") or spec.test_path
    if spec.sample_path is None or not spec.sample_path.exists():
        spec.sample_path = glob_one("*sample*.csv")
    if spec.dict_path is None or not spec.dict_path.exists():
        spec.dict_path = glob_one("*dictionary*.csv")
    if spec.output_path is None:
        spec.output_path = spec.task_dir / "submission.csv"


def parse_task(task_path: Path, llm=None) -> TaskSpec:
    task_path = task_path.resolve()
    text = task_path.read_text(encoding="utf-8")
    spec = TaskSpec(task_path=task_path, task_dir=task_path.parent, text=text)
    _find_paths(spec)

    train_cols = list(pd.read_csv(spec.train_path, nrows=200).columns)
    test_cols = list(pd.read_csv(spec.test_path, nrows=200).columns)
    sample = pd.read_csv(spec.sample_path, nrows=200) if spec.sample_path and spec.sample_path.exists() else None

    only_train = [c for c in train_cols if c not in test_cols]
    common_id = [c for c in test_cols if c.lower() in ("id", "index", "row_id", "sample_id")]

    if sample is not None:
        spec.submission_columns = list(sample.columns)
        spec.id_col = next((c for c in sample.columns if c in test_cols), sample.columns[0])
        others = [c for c in sample.columns if c != spec.id_col]
        if len(others) == 1 and others[0] in train_cols:
            spec.target = others[0]
        elif len(others) > 1:
            spec.proba_output = True
            spec.proba_columns = others
    if spec.id_col is None:
        spec.id_col = common_id[0] if common_id else test_cols[0]
    if spec.target is None and len(only_train) >= 1:
        spec.target = only_train[0]

    spec.metric = detect_metric(text)

    # LLM: дозаполнение пробелов
    if llm is not None and (spec.target is None or spec.metric is None):
        ans = llm.chat_json(
            "Ты извлекаешь параметры ML-задачи из описания.",
            "Описание задачи:\n" + text[:8000] + f"\n\nСтолбцы train: {train_cols[:200]}\n"
            "Верни JSON: {\"target\": str, \"id\": str, \"task_type\": \"regression|binary|multiclass\", "
            f"\"metric\": одно из {list(REGISTRY)}}}")
        if isinstance(ans, dict):
            if spec.target is None and ans.get("target") in train_cols:
                spec.target = ans["target"]
            if spec.metric is None and ans.get("metric") in REGISTRY:
                spec.metric = ans["metric"]

    if spec.target is None:
        raise RuntimeError("Не удалось определить таргет")

    # тип задачи по данным + тексту
    y = pd.read_csv(spec.train_path, usecols=[spec.target])[spec.target].dropna()
    low = text.lower()
    nun = y.nunique()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        spec.task_type = "binary" if nun == 2 else "multiclass"
    elif "регресс" in low or "regression" in low:
        spec.task_type = "regression"
    elif nun == 2:
        spec.task_type = "binary"
    elif ("классификац" in low or "classification" in low) and nun <= 100:
        spec.task_type = "multiclass"
    elif nun <= 20 and (y % 1 == 0).all():
        spec.task_type = "multiclass"
    else:
        spec.task_type = "regression"

    if spec.metric is None or REGISTRY[spec.metric][1] != ("reg" if spec.task_type == "regression" else "cls"):
        spec.metric = "rmse" if spec.task_type == "regression" else "macro_f1"

    if not spec.submission_columns:
        spec.submission_columns = [spec.id_col, spec.target]

    log(f"Task: type={spec.task_type} metric={spec.metric} target={spec.target} id={spec.id_col} "
        f"proba_output={spec.proba_output}")
    log(f"Paths: train={spec.train_path} test={spec.test_path} out={spec.output_path}")
    return spec
