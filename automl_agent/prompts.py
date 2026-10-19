"""Bounded model context derived from the durable experiment ledger."""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPT = """You are an autonomous ML researcher. Solve the supplied task by improving a real model, not by claiming a score.
The host owns a fixed train/validation split, evaluates predictions independently, and keeps the best candidate. Validation labels are not available to you. Test template values are placeholders. Never use IDs as predictive features.
Use run_experiment to propose one hypothesis and its complete Python source. The source must define build_model(context), returning an UNFITTED sklearn-compatible estimator or Pipeline with fit(X, y) and predict(X). The host passes raw pandas feature DataFrames and performs training, validation and final refit. Include preprocessing for categorical columns, missing values and infinity. context contains task_type (regression/classification), metric, feature_columns, categorical_columns, classes, seed and n_jobs. Use these keys; do not hardcode this dataset's target or columns when avoidable.
Allowed imports: sklearn, catboost, lightgbm, xgboost, numpy, pandas, scipy, math, functools, collections, typing, itertools, operator, statistics, warnings, re. No reading/writing files, networking, package installation, shell commands, dynamic eval, or introspection. Native ML dependencies may vary; CatBoost and sklearn are the recommended starting points. Use <=context['n_jobs'] threads, CPU only, bounded model sizes (typically 150-600 boosting iterations). Do not fit or predict at module import. Avoid lambdas/local functions if a named transformer can be used. A model must preserve row order; classification predictions are original class labels.
Inspect the previous results and errors. Form a concrete hypothesis about preprocessing, feature engineering, model family, regularization, imbalance or model parameters. Change something meaningful; duplicate code is rejected. You can return an entirely new estimator/pipeline, including custom sklearn transformers. Use the supplied baseline code as a working starting point and keep preprocessing correct.
Return a tool call, not a plan for the human to execute. For a provider without native tool calls, return exactly JSON {"tool":"run_experiment","arguments":{"hypothesis":"...","code":"complete Python source"}}. When at least one generated candidate has succeeded and further experiments are unhelpful, call finish_research with a reason. Never fabricate execution results, score, files, or successful recovery.
"""


def _read_code(result, limit=18000):
    if result is None:
        return ""
    path = Path(result.artifact_dir) / "code.py"
    return path.read_text(encoding="utf-8")[:limit] if path.exists() else ""


def build_messages(task, profile, results, best, runtime_context, feedback="", context_chars=48000, compact=False):
    ledger = [{"trial_id": r.trial_id, "status": r.status, "hypothesis": r.hypothesis[:1500], "metric": r.metric,
               "seconds": round(r.seconds, 2), "error": (r.error or "")[-2500:]} for r in results]
    failed = next((r for r in reversed(results) if r.status != "ok"), None)
    description = task.description
    if len(description) > 9000:
        description = description[:5500] + "\n[description cropped]\n" + description[-3500:]
    content = {"task_description": description, "task": task.model_context(runtime_context["n_jobs"]),
               "data_profile": profile, "experiments": ledger, "best_code": _read_code(best),
               "last_failed_code": _read_code(failed, 10000), "runtime": runtime_context,
               "previous_tool_feedback": feedback[-5000:]}
    limit = max(5000, context_chars // 2) if compact else max(5000, context_chars)
    # Preserve model code and the exact ledger; trim auxiliary profiling first.
    encoded = json.dumps(content, ensure_ascii=False, allow_nan=False, default=str)
    if len(encoded) > limit:
        content["data_profile"] = {"summary": str(profile)[:2000]}
        content["task_description"] = description[:3000] + "\n" + description[-1000:]
        content["last_failed_code"] = _read_code(failed, 5000)
        encoded = json.dumps(content, ensure_ascii=False, allow_nan=False, default=str)
    if len(encoded) > limit:
        content["best_code"] = _read_code(best, max(1500, limit // 3))
        content["experiments"] = ledger[-6:]
        content["last_failed_code"] = _read_code(failed, 1500)
        encoded = json.dumps(content, ensure_ascii=False, allow_nan=False, default=str)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": encoded}]
