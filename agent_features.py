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


# ---------------- операции, которым нужен таргет (выполняет хост, out-of-fold) ----------------

TARGET_OPS_DOC = {
    "target_encode": '{"op":"target_encode","cols":["a","b"],"smoothing":5} — средний таргет группы (ключ = комбинация cols) '
                     'и число её строк в train; хост считает out-of-fold, утечки нет. Сильнейший признак, если '
                     'одни и те же сущности (клиент, здание, устройство) повторяются в train и test — см. entity_keys',
    "target_lag": '{"op":"target_lag","cols":["a","b"]} — таргет той же сущности в последнем ПРЕДЫДУЩЕМ периоде и '
                  'сколько периодов назад он был (только при временной валидации; будущее не используется)',
}


def split_target_ops(ops):
    plain = [op for op in ops or [] if op.get("op") not in TARGET_OPS_DOC]
    target = [op for op in ops or [] if op.get("op") in TARGET_OPS_DOC]
    return plain, target


def _key(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if len(cols) == 1:
        return df[cols[0]].astype(str).to_numpy()
    return df[cols].astype(str).agg("|".join, axis=1).to_numpy()


def _encode(train_key, train_y, apply_key, smoothing):
    """train_y: [n] или [n, k] (one-hot для классов). Возвращает (enc[m, k], count[m])."""
    Y = train_y.reshape(len(train_y), -1).astype(float)
    frame = pd.DataFrame(Y).assign(_k=train_key)
    sums = frame.groupby("_k").sum()
    counts = frame.groupby("_k").size()
    prior = Y.mean(axis=0)
    s = sums.reindex(apply_key).to_numpy()
    c = counts.reindex(apply_key).fillna(0).to_numpy()
    enc = (np.nan_to_num(s) + prior * smoothing) / (c[:, None] + smoothing)
    enc[c == 0] = np.nan  # новая сущность — честный пропуск, модель сама решит
    return enc, c


def _lag(key, times, Y, train_rows, n):
    """Для каждой строки: среднее Y той же сущности в последний период строго раньше её периода."""
    k = Y.shape[1]
    hist = pd.DataFrame(Y[train_rows], columns=[f"y{j}" for j in range(k)]).assign(_k=key[train_rows], _t=times[train_rows])
    hist = hist.dropna(subset=["_t"]).groupby(["_k", "_t"], as_index=False).mean().sort_values("_t")
    hist["_t_prev"] = hist["_t"]
    query = pd.DataFrame({"_k": key, "_t": times, "_row": np.arange(n)}).dropna(subset=["_t"]).sort_values("_t")
    merged = pd.merge_asof(query, hist, on="_t", by="_k", allow_exact_matches=False).sort_values("_row")
    lag = np.full((n, k), np.nan)
    gap = np.full(n, np.nan)
    rows = merged["_row"].to_numpy()
    lag[rows] = merged[[f"y{j}" for j in range(k)]].to_numpy()
    gap[rows] = (merged["_t"] - merged["_t_prev"]).to_numpy()
    return lag, gap


def apply_target_ops(df: pd.DataFrame, ops: list[dict], y_train: np.ndarray, n_train: int, train_rows: np.ndarray,
                     classes: list | None, seed: int = 42, folds: int = 5, time_values=None) -> pd.DataFrame:
    """df — признаки train+test подряд. train_rows — строки train, чьи таргеты разрешено видеть
    (fit-часть при эксперименте, весь train в финале). Для них кодировка out-of-fold,
    для всех остальных строк — по всем train_rows."""
    from sklearn.model_selection import KFold

    df = df.copy()
    if classes:
        index = {str(c): i for i, c in enumerate(classes)}
        Y = np.eye(len(classes))[[index[str(v)] for v in y_train]]
    else:
        Y = np.asarray(y_train, float)
    for op in ops:
        cols = _existing(df, op["cols"])
        key = _key(df, cols)
        name = "__".join(cols)
        if op["op"] == "target_lag":
            if time_values is None:
                raise ValueError("target_lag доступен только при временной валидации (есть столбец времени)")
            lag, gap = _lag(key, np.asarray(time_values, float), Y.reshape(len(Y), -1), train_rows, len(df))
            if classes:
                for j, c in enumerate(classes):
                    df[f"lag_{name}__{c}"] = lag[:, j]
            else:
                df[f"lag_{name}"] = lag[:, 0]
            df[f"lag_{name}__gap"] = gap
            continue
        smoothing = float(op.get("smoothing", 5))
        k = Y.shape[1] if Y.ndim == 2 else 1
        enc = np.full((len(df), k), np.nan)
        cnt = np.zeros(len(df))
        other = np.setdiff1d(np.arange(len(df)), train_rows)
        enc[other], cnt[other] = _encode(key[train_rows], Y[train_rows], key[other], smoothing)
        for a, b in KFold(folds, shuffle=True, random_state=seed).split(train_rows):
            rows_a, rows_b = train_rows[a], train_rows[b]
            enc[rows_b], cnt[rows_b] = _encode(key[rows_a], Y[rows_a], key[rows_b], smoothing)
        if classes:
            for j, c in enumerate(classes):
                df[f"te_{name}__{c}"] = enc[:, j]
        else:
            df[f"te_{name}"] = enc[:, 0]
        df[f"te_{name}__count"] = cnt
    return df


def find_entity_keys(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target: str,
                     task_type: str, exclude=(), max_cols: int = 4, max_seconds: float = 30) -> tuple[list[dict], list[dict]]:
    """Ищем комбинацию столбцов, которая ведёт себя как ID сущности: значения повторяются
    внутри train, встречаются в test, а таргет внутри группы почти постоянен.

    Возвращает (ключи сущности, одиночные столбцы-кандидаты для target_encode)."""
    import time

    start = time.time()
    n = len(train)
    if task_type == "regression":
        y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
        if np.nanmin(y) >= 0 and pd.Series(y).skew() > 1:
            y = np.log1p(y)
        total = np.nanstd(y) or 1.0
    else:
        y = train[target].astype(str).to_numpy()
    date_parts = ("__ts_days", "__year", "__month", "__day", "__dow", "__hour")
    pool = [c for c in feature_cols if c not in exclude and not c.endswith(date_parts) and 10 < train[c].nunique() < 0.95 * n]
    base_purity = 0.0 if task_type == "regression" else float(pd.Series(y).value_counts(normalize=True).iloc[0])
    codes_tr = {c: train[c].astype(str).to_numpy() for c in pool}
    codes_te = {c: test[c].astype(str).to_numpy() for c in pool}

    def evaluate(cols):
        kt = pd.Series(codes_tr[cols[0]]) if len(cols) == 1 else pd.Series(["|".join(t) for t in zip(*(codes_tr[c] for c in cols))])
        ke = pd.Series(codes_te[cols[0]]) if len(cols) == 1 else pd.Series(["|".join(t) for t in zip(*(codes_te[c] for c in cols))])
        match = float(ke.isin(set(kt)).mean())
        size = kt.map(kt.value_counts())
        rep = (size >= 2).to_numpy()
        if rep.sum() < 50:
            return None
        if task_type == "regression":
            frame = pd.DataFrame({"k": kt, "y": y})
            resid = frame["y"] - frame.groupby("k")["y"].transform("mean")
            purity = 1 - float(np.sqrt(np.nanmean(resid[rep] ** 2)) / total)
        else:
            frame = pd.DataFrame({"k": kt[rep], "y": y[rep]})
            top = frame.groupby(["k", "y"]).size().groupby(level=0).max().sum()
            purity = float(top / rep.sum())
        return {"cols": list(cols), "test_rows_with_known_key": round(match, 3), "train_rows_repeated": round(float(rep.mean()), 3),
                "target_purity_within_key": round(purity, 3), "score": match * float(rep.mean()) * purity}

    best, chosen, singles = None, [], []
    while len(chosen) < max_cols and time.time() - start < max_seconds:
        candidates = [r for c in pool if c not in chosen and (r := evaluate(chosen + [c])) is not None]
        if not candidates:
            break
        if not chosen:
            singles = sorted(candidates, key=lambda d: -d["score"])
        top = max(candidates, key=lambda d: d["score"])
        if best and top["score"] <= best["score"] * 1.02:
            break
        best, chosen = top, top["cols"]
    # Ключ полезен, только если таргет внутри него заметно однороднее, чем в среднем по данным.
    # одиночные столбцы с однородным таргетом — кандидаты для target_encode (подсказка LLM)
    te_candidates = [
        {"col": d["cols"][0], "purity": d["target_purity_within_key"], "test_coverage": d["test_rows_with_known_key"]}
        for d in singles if d["target_purity_within_key"] >= max(0.3, base_purity + 0.1)][:6]
    if not best or best["test_rows_with_known_key"] < 0.2 or best["target_purity_within_key"] < max(0.3, base_purity + 0.1):
        return [], te_candidates
    best["score"] = round(best["score"], 3)
    return [best], te_candidates
