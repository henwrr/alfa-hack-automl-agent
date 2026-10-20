"""Постобработка: блендинг моделей и подбор весов классов под F1-подобные метрики."""
import numpy as np

from . import metrics as MT


def blend(results: list, y, metric: str, is_reg: bool, n_iter: int = 25):
    """Жадный ансамбль (Caruana, с повторами) по OOF — работает для любого числа моделей."""
    if len(results) == 1:
        return [1.0], results[0]["oof"], results[0]["test"]
    counts = np.zeros(len(results))
    best_i = max(range(len(results)), key=lambda i: MT.signed(metric, MT.score(metric, y, results[i]["oof"])))
    counts[best_i] = 1
    cur = results[best_i]["oof"].copy()
    cur_s = MT.signed(metric, MT.score(metric, y, cur))
    for _ in range(n_iter):
        n = counts.sum()
        cand = [(MT.signed(metric, MT.score(metric, y, (cur * n + r["oof"]) / (n + 1))), i)
                for i, r in enumerate(results)]
        s, i = max(cand)
        if s <= cur_s + 1e-9:
            break
        counts[i] += 1
        cur = (cur * n + results[i]["oof"]) / (n + 1)
        cur_s = s
    w = counts / counts.sum()
    oof = sum(wi * r["oof"] for wi, r in zip(w, results))
    test = sum(wi * r["test"] for wi, r in zip(w, results))
    return [round(float(x), 3) for x in w], oof, test


def class_weights_for_metric(oof, y, metric: str, n_rounds: int = 3):
    """Покоординатный подбор множителей вероятностей классов (argmax(p*w)) под label-метрику."""
    k = oof.shape[1]
    w = np.ones(k)
    base = MT.score(metric, y, oof)
    best = MT.signed(metric, base)
    grid = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
    for _ in range(n_rounds):
        improved = False
        for c in range(k):
            cur = w[c]
            for g in grid:
                w2 = w.copy()
                w2[c] = cur * g
                s = MT.signed(metric, MT.score(metric, y, oof * w2))
                if s > best + 1e-6:
                    best, w = s, w2
                    improved = True
        if not improved:
            break
    return w, MT.score(metric, y, oof * w), base


def tune_class_weights(oof, y, metric: str, seed: int = 0):
    """Подбор весов с проверкой на отложенной половине OOF, чтобы не переобучиться на OOF."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    wa, _, _ = class_weights_for_metric(oof[a], y[a], metric)
    gain_b = MT.signed(metric, MT.score(metric, y[b], oof[b] * wa)) - MT.signed(metric, MT.score(metric, y[b], oof[b]))
    if gain_b <= 0:
        return None, 0.0
    w, full, base = class_weights_for_metric(oof, y, metric)
    return w, MT.signed(metric, full) - MT.signed(metric, base)
