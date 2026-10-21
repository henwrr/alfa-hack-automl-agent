"""Синтетические задачи нестандартного вида для проверки устойчивости агента.

python tests/make_synthetic.py OUT_DIR
Создаёт OUT_DIR/{binary_auc,multi_proba,dates_reg,text_cls,highcard_rmsle,big_mape}/task.md + CSV.
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

    # 4) текстовая колонка: таргет зависит от слов в отзыве, accuracy
    words = ["быстро", "удобно", "дорого", "медленно", "отлично", "плохо", "нормально", "доставка", "курьер", "товар"]
    good, bad = {"быстро", "удобно", "отлично"}, {"дорого", "медленно", "плохо"}
    texts = [" ".join(rng.choice(words, size=rng.integers(3, 15))) for _ in range(n + m)]
    score = np.array([sum(w in good for w in t.split()) - sum(w in bad for w in t.split()) for t in texts])
    df = frame(n + m, rng).drop(columns="event_date")
    df.insert(0, "review_id", np.arange(n + m))
    df["review_text"] = texts
    df["liked"] = np.where(score + rng.normal(0, 1, n + m) > 0, "yes", "no")
    write(out / "text_cls", "# Отзывы\nКлассификация: понравился ли заказ (`liked`, yes/no) по тексту отзыва. Метрика — accuracy.\n",
          df.iloc[:n], df.iloc[n:].drop(columns="liked"), pd.DataFrame({"review_id": df.review_id.iloc[n:], "liked": "yes"}))
    # 5) категория на ~50 тыс. значений + RMSLE, таргет положительный с хвостом
    nn, mm = 120000, 30000
    users = rng.integers(0, 50000, size=nn + mm)
    user_effect = rng.lognormal(0, 1, 50000)
    df = pd.DataFrame({"user_id": [f"u{u}" for u in users], "x1": rng.normal(size=nn + mm),
                       "x2": rng.choice(["a", "b", "c"], size=nn + mm)})
    df["spend"] = user_effect[users] * np.exp(0.3 * df.x1) * rng.lognormal(0, 0.3, nn + mm)
    df.insert(0, "id", np.arange(nn + mm))
    write(out / "highcard_rmsle", "# Траты\nРегрессия: предсказать `spend`. Метрика — RMSLE.\n",
          df.iloc[:nn], df.iloc[nn:].drop(columns="spend"), pd.DataFrame({"id": df.id.iloc[nn:], "spend": 0.0}))
    # 6) большой датасет: 500 тыс. строк train, MAPE
    nb, mb = 500000, 100000
    X = rng.normal(size=(nb + mb, 15)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(15)])
    df["region"] = rng.choice([f"r{i}" for i in range(30)], size=nb + mb)
    df["price"] = 100 + 10 * df.f0 + 5 * df.f1 ** 2 + df.region.str[1:].astype(int) + rng.normal(0, 3, nb + mb)
    df.insert(0, "id", np.arange(nb + mb))
    write(out / "big_mape", "# Цена\nРегрессия: предсказать `price`. Метрика — MAPE.\n",
          df.iloc[:nb], df.iloc[nb:].drop(columns="price"), pd.DataFrame({"id": df.id.iloc[nb:], "price": 0.0}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
