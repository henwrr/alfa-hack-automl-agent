"""Bounded model context derived from the durable experiment ledger."""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPT = """You are an autonomous ML researcher. Solve the supplied task by improving a real model, not by claiming a score.
The host owns a fixed train/validation split, evaluates predictions independently, and keeps the best candidate. Validation labels are not available to you. Test template values are placeholders. Never use IDs as predictive features.
Use run_experiment to propose one hypothesis and its complete Python source. The source must define build_model(context), returning an UNFITTED sklearn-compatible estimator or Pipeline with fit(X, y) and predict(X). The host passes raw pandas feature DataFrames and performs training, validation and final refit. Include preprocessing for categorical columns, missing values and infinity. context contains task_type (regression/classification), metric, feature_columns, categorical_columns, classes, seed and n_jobs. Use these keys; do not hardcode this dataset's target or columns when avoidable.
Allowed imports: sklearn, catboost, lightgbm, xgboost, numpy, pandas, scipy, math, functools, collections, typing, itertools, operator, statistics, warnings, re. No reading/writing files, networking, package installation, shell commands, dynamic eval, or introspection. Native ML dependencies may vary; CatBoost and sklearn are the recommended starting points. Use <=context['n_jobs'] threads, CPU only, bounded model sizes (typically 150-600 boosting iterations). Do not fit or predict at module import. Avoid lambdas/local functions if a named transformer can be used. A model must preserve row order; classification predictions are original class labels.
Inspect the previous results and errors. The profile includes available_packages: use packages with a real version, never those marked unavailable. Prefer a focused improvement to the current working code before changing an entire model family. Repair the actual traceback before proposing more complexity. Form a concrete hypothesis about preprocessing, feature engineering, model family, regularization, imbalance or model parameters. Change something meaningful; duplicate code is rejected. You can return an entirely new estimator/pipeline, including custom sklearn transformers. Use the supplied baseline code as a working starting point and keep preprocessing correct. Write hypothesis and finish reason in Russian.
For every run_experiment include decision with concise PUBLIC explanations in Russian: model_family, observations (facts from the profile), rationale (why this change), alternatives (considered models), expected_effect (an untested expectation), risks, previous_result_assessment (what the last measured result implies). Explain decisions for the user; do not supply private internal reasoning. Never invent measurements or call an expectation a result. For generated candidates let errors propagate instead of silently switching models. A deliberate fallback must expose a public fallback_reason attribute; otherwise a successful run may not test the stated hypothesis. The research_strategy is a starting shortlist, not a guaranteed winner. Use experiment model_details to distinguish the actual fitted backend from the hypothesis.
Return a tool call, not a plan for the human to execute. For a provider without native tool calls, return exactly JSON {"tool":"run_experiment","arguments":{"hypothesis":"...","code":"complete Python source"}}. When at least one generated candidate has succeeded and further experiments are unhelpful, call finish_research with a reason. Never fabricate execution results, score, files, or successful recovery.
"""


def _read_code(result, limit=18000):
    if result is None:
        return ""
    path = Path(result.artifact_dir) / "code.py"
    return path.read_text(encoding="utf-8")[:limit] if path.exists() else ""


def build_messages(task, profile, results, best, runtime_context, feedback="", context_chars=48000, compact=False):
    ledger = [{"trial_id": r.trial_id, "status": r.status, "hypothesis": r.hypothesis[:1500], "metric": r.metric,
               "seconds": round(r.seconds, 2), "model_details": r.model_details, "error": (r.error or "")[-2500:]} for r in results]
    failed = next((r for r in reversed(results) if r.status != "ok"), None)
    description = task.description
    if len(description) > 9000:
        description = description[:5500] + "\n[description cropped]\n" + description[-3500:]
    content = {"task_description": description, "task": task.model_context(runtime_context["n_jobs"]),
               "data_profile": {k: v for k, v in profile.items() if k != "research_strategy"},
               "research_strategy": profile.get("research_strategy", {}), "experiments": ledger, "best_code": _read_code(best),
               "last_failed_code": _read_code(failed, 10000), "runtime": runtime_context,
               "previous_tool_feedback": feedback[-5000:]}
    limit = max(5000, context_chars // 2) if compact else max(5000, context_chars)
    if compact:
        # Actually compact auxiliary context even when it is below context_chars.
        # Keep the most recent balanced assistant/tool group in the caller so a
        # failed experiment remains actionable under a tight token budget.
        content["data_profile"] = {key: profile[key] for key in (
            "train_shape", "test_shape", "feature_count", "categorical_columns", "target", "available_packages") if key in profile}
        strategy = profile.get("research_strategy", {})
        content["research_strategy"] = {
            "facts": strategy.get("facts", {}),
            "candidate_shortlist": [{"model_family": candidate["model_family"]}
                                    for candidate in strategy.get("candidate_shortlist", [])],
        }
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
