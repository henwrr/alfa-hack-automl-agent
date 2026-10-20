"""Загрузка данных, определение типов признаков, базовая предобработка и безопасные FE-действия."""
import re
import warnings

import numpy as np
import pandas as pd

from .config import log

HIGH_CARD = 1000


class Dataset:
    def __init__(self, spec):
        self.spec = spec
        self.train = pd.read_csv(spec.train_path, low_memory=False)
        self.test = pd.read_csv(spec.test_path, low_memory=False)
        self.train = self.train[self.train[spec.target].notna()].reset_index(drop=True)
        self.n_train = len(self.train)
        self.dd = None
        if spec.dict_path is not None and spec.dict_path.exists():
            try:
                self.dd = pd.read_csv(spec.dict_path)
            except Exception:  # noqa: BLE001
                self.dd = None
        self.y_raw = self.train[spec.target]
        self.feature_cols = [c for c in self.test.columns if c != spec.id_col and c in self.train.columns
                             and c != spec.target]
        self.all = pd.concat([self.train[self.feature_cols], self.test[self.feature_cols]],
                             axis=0, ignore_index=True)
        self._infer_types()
        self.base = self._build_base()
        log(f"Data: train={self.train.shape} test={self.test.shape} num={len(self.num_cols)} "
            f"cat={len(self.cat_cols)} dt={len(self.dt_cols)} text={len(self.text_cols)} "
            f"dropped={len(self.dropped)}")

    # ---------- типы ----------
    def _dd_hints(self):
        hints = {}
        if self.dd is None or self.dd.empty:
            return hints
        cols = {c.lower(): c for c in self.dd.columns}
        fcol = cols.get("feature") or cols.get("column") or cols.get("name") or self.dd.columns[0]
        for _, row in self.dd.iterrows():
            text = " ".join(str(row[c]) for c in self.dd.columns if c != fcol).lower()
            if re.search(r"категор|categor|nominal|номинал", text):
                hints[str(row[fcol])] = "cat"
        return hints

    def _infer_types(self):
        hints = self._dd_hints()
        self.num_cols, self.cat_cols, self.dt_cols, self.text_cols, self.dropped = [], [], [], [], []
        n = len(self.all)
        for c in self.feature_cols:
            s = self.all[c]
            nun = s.nunique(dropna=True)
            if nun <= 1:
                self.dropped.append(c)
                continue
            if pd.api.types.is_bool_dtype(s):
                self.cat_cols.append(c)
            elif pd.api.types.is_numeric_dtype(s):
                if hints.get(c) == "cat" and nun <= HIGH_CARD:
                    self.cat_cols.append(c)
                else:
                    self.num_cols.append(c)
            else:
                ss = s.dropna().astype(str)
                sample = ss.sample(min(len(ss), 2000), random_state=0)
                if self._looks_datetime(sample):
                    self.dt_cols.append(c)
                    continue
                num_try = pd.to_numeric(sample, errors="coerce")
                if num_try.notna().mean() > 0.98 and hints.get(c) != "cat":
                    self.num_cols.append(c)
                    self.all[c] = pd.to_numeric(self.all[c], errors="coerce")
                    continue
                avg_len = sample.str.len().mean()
                if avg_len > 40 and nun > 0.3 * n:
                    self.text_cols.append(c)
                else:
                    self.cat_cols.append(c)

    @staticmethod
    def _looks_datetime(sample: pd.Series) -> bool:
        if sample.empty or not sample.str.contains(r"\d{1,4}[-/.:]\d{1,2}", regex=True).mean() > 0.9:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() > 0.9

    # ---------- базовый фрейм ----------
    def _build_base(self) -> pd.DataFrame:
        parts = {}
        for c in self.num_cols:
            parts[c] = self.all[c].astype("float64")
        for c in self.cat_cols:
            parts[c] = self.all[c].astype(str).where(self.all[c].notna(), "__nan__")
        for c in self.dt_cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = pd.to_datetime(self.all[c], errors="coerce", format="mixed")
            parts[f"{c}__year"] = d.dt.year.astype("float64")
            parts[f"{c}__month"] = d.dt.month.astype("float64")
            parts[f"{c}__day"] = d.dt.day.astype("float64")
            parts[f"{c}__dow"] = d.dt.dayofweek.astype("float64")
            if (d.dt.hour.fillna(0) != 0).mean() > 0.05:
                parts[f"{c}__hour"] = d.dt.hour.astype("float64")
            parts[f"{c}__ts"] = (d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
        for c in self.text_cols:
            s = self.all[c].fillna("").astype(str)
            parts[f"{c}__len"] = s.str.len().astype("float64")
            parts[f"{c}__words"] = s.str.split().str.len().astype("float64")
        return pd.DataFrame(parts)

    @property
    def base_cat_cols(self):
        return list(self.cat_cols)

    def column_summary(self, max_cols: int = 120) -> str:
        """Краткое описание признаков для LLM."""
        desc = {}
        if self.dd is not None:
            cols = {c.lower(): c for c in self.dd.columns}
            fcol = cols.get("feature") or self.dd.columns[0]
            dcol = next((self.dd.columns[i] for i, c in enumerate(self.dd.columns)
                         if "desc" in c.lower()), None)
            if dcol is not None:
                desc = dict(zip(self.dd[fcol].astype(str), self.dd[dcol].astype(str)))
        lines = []
        for c in (self.num_cols + self.cat_cols + self.dt_cols + self.text_cols)[:max_cols]:
            s = self.all[c]
            kind = ("num" if c in self.num_cols else "cat" if c in self.cat_cols
                    else "datetime" if c in self.dt_cols else "text")
            na = s.isna().mean()
            extra = ""
            if kind == "num":
                extra = f"min={s.min():.4g} med={s.median():.4g} max={s.max():.4g}"
            elif kind == "cat":
                extra = f"nunique={s.nunique()}"
            d = desc.get(c, "")[:120]
            lines.append(f"- {c} [{kind}] na={na:.0%} {extra} {d}")
        return "\n".join(lines)


# ---------- FE-действия (DSL) ----------
NUM_OPS = {"ratio", "product", "diff", "sum", "log", "row_stats", "group_agg", "freq", "count_nan", "drop"}


def validate_action(a: dict, ds: Dataset) -> dict | None:
    """Проверить и нормализовать действие; None если невалидно."""
    if not isinstance(a, dict) or a.get("type") not in NUM_OPS:
        return None
    num = set(ds.num_cols) | set(c for c in ds.base.columns if c not in ds.cat_cols)
    cats = set(ds.cat_cols)
    allc = num | cats
    t = a["type"]
    try:
        if t in ("ratio", "product", "diff"):
            if a["a"] in num and a["b"] in num and a["a"] != a["b"]:
                return {"type": t, "a": a["a"], "b": a["b"]}
        elif t == "log":
            if a["col"] in num:
                return {"type": t, "col": a["col"]}
        elif t in ("sum", "row_stats", "count_nan"):
            cols = [c for c in a.get("cols", []) if c in (allc if t == "count_nan" else num)]
            if len(cols) >= 2:
                out = {"type": t, "cols": sorted(set(cols))}
                if t == "row_stats":
                    st = [s for s in a.get("stats", ["mean", "std", "min", "max"])
                          if s in ("mean", "std", "min", "max", "median")]
                    out["stats"] = st or ["mean", "std", "min", "max"]
                return out
        elif t == "group_agg":
            agg = a.get("agg", "mean")
            if a["group"] in cats | num and a["col"] in num and agg in ("mean", "std", "median", "min", "max"):
                return {"type": t, "group": a["group"], "col": a["col"], "agg": agg}
        elif t == "freq":
            if a["col"] in allc:
                return {"type": t, "col": a["col"]}
        elif t == "drop":
            cols = [c for c in a.get("cols", []) if c in ds.base.columns]
            if cols:
                return {"type": t, "cols": sorted(set(cols))}
    except (KeyError, TypeError):
        return None
    return None


def apply_actions(base: pd.DataFrame, actions: list) -> pd.DataFrame:
    df = base.copy()
    new = {}
    drops = set()
    eps = 1e-9
    for a in actions:
        t = a["type"]
        if t == "ratio":
            new[f"fe_{a['a']}_div_{a['b']}"] = df[a["a"]] / (df[a["b"]].replace(0, np.nan) + eps)
        elif t == "product":
            new[f"fe_{a['a']}_mul_{a['b']}"] = df[a["a"]] * df[a["b"]]
        elif t == "diff":
            new[f"fe_{a['a']}_sub_{a['b']}"] = df[a["a"]] - df[a["b"]]
        elif t == "log":
            new[f"fe_log_{a['col']}"] = np.sign(df[a["col"]]) * np.log1p(np.abs(df[a["col"]]))
        elif t == "sum":
            new["fe_sum_" + "_".join(a["cols"])[:60]] = df[a["cols"]].sum(axis=1)
        elif t == "row_stats":
            key = "_".join(a["cols"])[:40]
            for st in a["stats"]:
                new[f"fe_{st}_{key}"] = getattr(df[a["cols"]], st)(axis=1)
        elif t == "count_nan":
            sub = df[a["cols"]]
            cnt = sub.isna().sum(axis=1)
            for c in a["cols"]:
                if sub[c].dtype == object:
                    cnt = cnt + (sub[c] == "__nan__")
            new["fe_nan_" + "_".join(a["cols"])[:50]] = cnt
        elif t == "group_agg":
            g = df[a["group"]]
            new[f"fe_{a['agg']}_{a['col']}_by_{a['group']}"] = df[a["col"]].groupby(g).transform(a["agg"])
        elif t == "freq":
            new[f"fe_freq_{a['col']}"] = df[a["col"]].map(df[a["col"]].value_counts(dropna=False))
        elif t == "drop":
            drops.update(a["cols"])
    if new:
        nd = pd.DataFrame(new, index=df.index).replace([np.inf, -np.inf], np.nan).astype("float64")
        nd = nd.loc[:, ~nd.columns.duplicated()]
        nd = nd[[c for c in nd.columns if c not in df.columns]]
        df = pd.concat([df, nd], axis=1)
    if drops:
        df = df.drop(columns=[c for c in drops if c in df.columns])
    return df


def encode_for(model: str, df: pd.DataFrame, cat_cols: list) -> tuple[pd.DataFrame, list]:
    """Кодирование категорий под конкретную модель."""
    df = df.copy()
    cats = [c for c in cat_cols if c in df.columns]
    used = []
    for c in cats:
        nun = df[c].nunique()
        if nun > HIGH_CARD or model == "linear":
            df[f"{c}__freq"] = df[c].map(df[c].value_counts()).astype("float64")
            df = df.drop(columns=[c])
        else:
            if model in ("lgbm", "xgb"):
                df[c] = df[c].astype("category")
            used.append(c)
    return df, used
