"""Синтетические задачи нестандартного вида для проверки устойчивости агента.

python tests/make_synthetic.py OUT_DIR
Создаёт OUT_DIR/{binary_auc,multi_proba,dates_reg}/task.md + CSV.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def frame(n, rng):
    df = pd.DataFrame({
        "num_a": rng.normal(size=n),
        "num_b": rng.exponential(2, size=n),
        "city": rng.choice(["msk", "spb", "kzn", None], size=n, p=[0.5, 0.3, 0.15, 0.05]),
        "segment": rng.choice(list("ABCDE"), size=n),
        "event_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D"),
    })
    df.loc[rng.random(n) < 0.1, "num_b"] = np.nan
    return df


def write(task_dir: Path, text: str, train, test, sample):
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.md").write_text(text, encoding="utf-8")
    train.to_csv(task_dir / "train.csv", index=False)
    test.to_csv(task_dir / "test.csv", index=False)
    sample.to_csv(task_dir / "sample_submission.csv", index=False)


def main(out: Path):
    rng = np.random.default_rng(0)
    n, m = 4000, 1500
    # 1) бинарная, ROC-AUC, id называется client_id, даты строками
    df = frame(n + m, rng)
    logit = 1.5 * df.num_a - 0.4 * df.num_b.fillna(2) + (df.segment == "A") * 1.0 + (df.event_date.dt.month > 6)
    df["is_churn"] = (rng.random(n + m) < 1 / (1 + np.exp(-logit))).astype(int)
    df["event_date"] = df.event_date.dt.strftime("%Y-%m-%d")
    df.insert(0, "client_id", np.arange(100000, 100000 + n + m))
    write(out / "binary_auc", "# Отток\nБинарная классификация оттока клиента `is_churn`. Метрика — ROC-AUC. "
          "Ответ — вероятность оттока.\n", df.iloc[:n], df.iloc[n:].drop(columns="is_churn"),
          pd.DataFrame({"client_id": df.client_id.iloc[n:], "is_churn": 0.5}))
    # 2) мультикласс, log loss, сабмит — по колонке на класс
    df = frame(n + m, rng)
    score = df.num_a + (df.segment.isin(["A", "B"])) * 1.5
    df["tier"] = np.select([score < -0.5, score < 1.0], ["low", "mid"], "high")
    df["event_date"] = df.event_date.dt.strftime("%d.%m.%Y")
    df.insert(0, "row_id", np.arange(n + m))
    sample = pd.DataFrame({"row_id": df.row_id.iloc[n:], "high": 1 / 3, "low": 1 / 3, "mid": 1 / 3})
    write(out / "multi_proba", "# Уровень клиента\nМногоклассовая классификация `tier`. Метрика оценки — log loss. "
          "Для каждого класса нужна вероятность.\n", df.iloc[:n], df.iloc[n:].drop(columns="tier"), sample)
    # 3) регрессия по времени: test позже train по дате
    df = frame(n + m, rng).sort_values("event_date").reset_index(drop=True)
    df["sales"] = 100 + 20 * df.num_a + 5 * df.num_b.fillna(0) + 0.1 * df.event_date.dt.dayofyear + rng.normal(0, 5, n + m)
    df["event_date"] = df.event_date.dt.strftime("%Y-%m-%d %H:%M")
    df.insert(0, "id", np.arange(n + m))
    write(out / "dates_reg", "# Продажи\nРегрессия: предсказать `sales`. Метрика — MAE.\n",
          df.iloc[:n], df.iloc[n:].drop(columns="sales"), pd.DataFrame({"id": df.id.iloc[n:], "sales": 0.0}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
