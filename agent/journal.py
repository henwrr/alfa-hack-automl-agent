"""Журнал экспериментов: что пробовали и с каким результатом."""
import json

import numpy as np


class Journal:
    def __init__(self, path, metric):
        self.path = path
        self.metric = metric
        self.entries = []

    def add(self, name, hypothesis, cfg, result, accepted, note=""):
        e = {
            "step": len(self.entries) + 1, "name": name, "hypothesis": hypothesis,
            "config": cfg, "accepted": bool(accepted), "note": note,
        }
        if result is not None:
            e.update(score=round(float(result["score"]), 6),
                     fold_std=round(float(np.std(result["fold_scores"])), 6),
                     time=round(result["time"], 1), n_features=result.get("n_features"))
        self.entries.append(e)
        self.save()

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=1, default=str)
        except Exception:  # noqa: BLE001
            pass

    def summary(self, last: int = 15) -> str:
        lines = []
        for e in self.entries[-last:]:
            s = e.get("score")
            lines.append(f"#{e['step']} {'ПРИНЯТО' if e['accepted'] else 'отклонено'} "
                         f"{self.metric}={s} :: {e['name']} :: {str(e['hypothesis'])[:150]} {e['note']}")
        return "\n".join(lines)

    def tried_keys(self):
        return {json.dumps(e["config"], sort_keys=True, default=str) for e in self.entries}
