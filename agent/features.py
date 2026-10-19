"""Язык признаков, которым пользуется LLM.

LLM не пишет весь пайплайн целиком (у Апрельского это давало много падений),
а описывает признаки списком операций. Для нестандартных идей есть операция
`custom` с функцией transform(df). Все операции применяются к train+test без
таргета, поэтому утечки таргета через них нет.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

OPS_DOC = {
    "drop": '{"op":"drop","cols":[...]} — удалить признаки',
    "to_nan": '{"op":"to_nan","cols":[...],"values":[0,-1]} — заменить «мусорные» значения на пропуск',
    "log1p": '{"op":"log1p","cols":[...]} — log1p(|x|)*sign(x) для тяжёлых хвостов',
    "missing_flags": '{"op":"missing_flags","cols":[...]} — индикаторы пропусков (cols можно опустить = все с пропусками)',
    "freq_encode": '{"op":"freq_encode","cols":[...]} — частота значения категории',
    "group_stats": '{"op":"group_stats","by":["col"],"cols":[...],"stats":["mean","std","rank_pct"]} — статистики числовых признаков внутри группы и отклонение от среднего группы',
    "ratio": '{"op":"ratio","pairs":[["a","b"],...]} — a/b',
    "diff": '{"op":"diff","pairs":[["a","b"],...]} — a-b',
    "product": '{"op":"product","pairs":[["a","b"],...]} — a*b',
    "row_stats": '{"op":"row_stats","cols":[...],"name":"temps","stats":["mean","std","min","max"]} — статистики по строке для группы похожих столбцов',
    "combine": '{"op":"combine","cols":["a","b"]} — новая категория как конкатенация категорий',
    "bin": '{"op":"bin","cols":[...],"q":10} — квантильная дискретизация',
    "custom": '{"op":"custom","code":"def transform(df):\\n    df[\'new\'] = ...\\n    return df"} — произвольный pandas/numpy код; df — train+test без таргета',
}

_ALLOWED_IMPORTS = {"numpy", "pandas", "math", "re"}
_FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
                    "getattr", "setattr", "delattr", "exit", "quit", "breakpoint"}
_SAFE_TO = {"to_numpy", "to_datetime", "to_numeric", "to_timedelta", "to_list", "to_frame", "to_dict", "to_period"}


def _check_custom(code: str) -> None:
    tree = ast.parse(code)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "transform" for n in tree.body):
        raise ValueError("custom: нужна функция transform(df)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] not in _ALLOWED_IMPORTS for a in node.names):
            raise ValueError("custom: разрешены только numpy, pandas, math, re")
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
            raise ValueError("custom: разрешены только numpy, pandas, math, re")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            raise ValueError(f"custom: нельзя вызывать {node.func.id}")
        if not isinstance(node, ast.Attribute):
            continue
        writes_or_reads = (node.attr.startswith("to_") and node.attr not in _SAFE_TO) or node.attr.startswith("read_")
        if node.attr.startswith("__") or writes_or_reads:
            raise ValueError(f"custom: нельзя использовать .{node.attr}")


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _existing(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"нет столбцов {missing}")
    return list(cols)


def apply_ops(df: pd.DataFrame, cat_cols: list[str], ops: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """df — признаки train+test подряд. Возвращает новый df и список категориальных."""
    df = df.copy()
    cats = list(cat_cols)
    for op in ops or []:
        kind = op.get("op")
        if kind == "drop":
            cols = [c for c in op.get("cols", []) if c in df.columns]
            df = df.drop(columns=cols)
            cats = [c for c in cats if c not in cols]
        elif kind == "to_nan":
            for c in _existing(df, op["cols"]):
                df[c] = df[c].where(~df[c].isin(op.get("values", [0])), np.nan)
        elif kind == "log1p":
            for c in _existing(df, op["cols"]):
                x = _num(df, c)
                df[c] = np.sign(x) * np.log1p(np.abs(x))
        elif kind == "missing_flags":
            cols = op.get("cols") or [c for c in df.columns if df[c].isna().any()]
            for c in _existing(df, cols):
                df[f"{c}__isna"] = df[c].isna().astype(np.int8)
        elif kind == "freq_encode":
            for c in _existing(df, op["cols"]):
                df[f"{c}__freq"] = df[c].map(df[c].value_counts(normalize=True, dropna=False)).astype(float)
        elif kind == "group_stats":
            by = op["by"] if isinstance(op["by"], list) else [op["by"]]
            _existing(df, by)
            key = df[by].astype(str).agg("|".join, axis=1)
            name = "_".join(by)
            for c in _existing(df, op["cols"]):
                x = _num(df, c)
                grouped = x.groupby(key)
                for stat in op.get("stats", ["mean"]):
                    if stat == "rank_pct":
                        df[f"{c}__rank_in_{name}"] = grouped.rank(pct=True)
                    elif stat in {"mean", "std", "median", "min", "max", "count"}:
                        df[f"{c}__{stat}_by_{name}"] = grouped.transform(stat)
                    if stat == "mean":
                        df[f"{c}__dev_by_{name}"] = x - df[f"{c}__mean_by_{name}"]
        elif kind in {"ratio", "diff", "product"}:
            for a, b in op["pairs"]:
                _existing(df, [a, b])
                x, y = _num(df, a), _num(df, b)
                if kind == "ratio":
                    df[f"{a}__div__{b}"] = x / y.replace(0, np.nan)
                elif kind == "diff":
                    df[f"{a}__minus__{b}"] = x - y
                else:
                    df[f"{a}__x__{b}"] = x * y
        elif kind == "row_stats":
            cols = _existing(df, op["cols"])
            block = df[cols].apply(pd.to_numeric, errors="coerce")
            name = op.get("name", "row")
            for stat in op.get("stats", ["mean", "std", "min", "max"]):
                df[f"{name}__{stat}"] = getattr(block, stat)(axis=1)
        elif kind == "combine":
            cols = _existing(df, op["cols"])
            new = "__".join(cols)
            df[new] = df[cols].astype(str).agg("|".join, axis=1)
            cats.append(new)
        elif kind == "bin":
            for c in _existing(df, op["cols"]):
                df[f"{c}__bin"] = pd.qcut(_num(df, c), q=int(op.get("q", 10)), labels=False, duplicates="drop")
        elif kind == "custom":
            code = op["code"]
            _check_custom(code)
            scope = {"np": np, "pd": pd}
            exec(compile(code, "<custom_features>", "exec"), scope)  # noqa: S102 — код проверен AST-фильтром
            before = set(df.columns)
            out = scope["transform"](df.copy())
            if not isinstance(out, pd.DataFrame) or len(out) != len(df):
                raise ValueError("custom: transform должна вернуть DataFrame той же длины")
            df = out.reset_index(drop=True) if not out.index.equals(df.index) else out
            for c in df.columns:
                if c not in before and (df[c].dtype == object or str(df[c].dtype) == "category"):
                    cats.append(c)
            cats = [c for c in cats if c in df.columns]
        else:
            raise ValueError(f"неизвестная операция признаков: {kind}")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, list(dict.fromkeys(c for c in cats if c in df.columns))
