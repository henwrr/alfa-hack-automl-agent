"""Bounded model context derived from the durable experiment ledger."""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPT = """You are an autonomous ML researcher. Solve the supplied task by improving a real model, not by claiming a score.
The host owns a fixed train/validation split, evaluates predictions independently, and keeps the best candidate. Validation labels are not available to you. Test template values are placeholders. Never use IDs as predictive features.
When runtime.parallel_trials=2 and at least two trial slots remain, prefer run_experiments with an array of two independent focused hypotheses; the host trains them concurrently, each using context.n_jobs threads. Use run_experiment when only one hypothesis or trial slot remains. Never base one candidate on unobserved results of its companion.
Use run_experiment to propose one hypothesis and its complete Python source. The source must define build_model(context), returning an UNFITTED sklearn-compatible estimator or Pipeline with fit(X, y) and predict(X). The host passes raw pandas feature DataFrames and performs training, validation and final refit. Include preprocessing for categorical columns, missing values and infinity. context contains task_type (regression/classification), metric, feature_columns, categorical_columns, classes, seed and n_jobs. Use these keys; do not hardcode this dataset's target or columns when avoidable.
Allowed imports: sklearn, catboost, lightgbm, xgboost, numpy, pandas, scipy, math, functools, collections, typing, itertools, operator, statistics, warnings, re. No reading/writing files, networking, package installation, shell commands, dynamic eval, or introspection. Native ML dependencies may vary; CatBoost and sklearn are the recommended starting points. Use <=context['n_jobs'] threads, CPU only, bounded model sizes (typically 150-600 boosting iterations). Do not fit or predict at module import. Avoid lambdas/local functions if a named transformer can be used. A model must preserve row order; classification predictions are original class labels.
Inspect the previous results and errors. The profile includes available_packages: use packages with a real version, never those marked unavailable. Prefer a focused improvement to the current working code before changing an entire model family. Repair the actual traceback before proposing more complexity. Form a concrete hypothesis about preprocessing, feature engineering, model family, regularization, imbalance or model parameters. Change something meaningful; duplicate code is rejected. You can return an entirely new estimator/pipeline, including custom sklearn transformers. Use the supplied baseline code as a working starting point and keep preprocessing correct. Write hypothesis and finish reason in Russian.
For every run_experiment include decision with concise PUBLIC explanations in Russian: model_family, observations (facts from the profile), rationale (why this change), alternatives (considered models), expected_effect (an untested expectation), risks, previous_result_assessment (what the last measured result implies). Explain decisions for the user; do not supply private internal reasoning. Never invent measurements or call an expectation a result. For generated candidates let errors propagate instead of silently switching models. A deliberate fallback must expose a public fallback_reason attribute; otherwise a successful run may not test the stated hypothesis. The host has already measured the family_comparison before research; use those scores rather than repeating unchanged baselines. The research_strategy is a starting shortlist, not a guaranteed winner. Class labels may be strings: count with np.unique(y, return_counts=True) or pandas rather than np.bincount on raw labels. Use experiment model_details to distinguish the actual fitted backend from the hypothesis.
Return a tool call, not a plan for the human to execute. For a provider without native tool calls, return exactly JSON {"tool":"run_experiment","arguments":{"hypothesis":"...","code":"complete Python source"}}. When at least one generated candidate has succeeded and further experiments are unhelpful, call finish_research with a reason. Never fabricate execution results, score, files, or successful recovery.
"""


COMPACT_SYSTEM_PROMPT = """You are an autonomous tabular ML researcher. Host owns fixed validation labels, independent scoring and best-model selection. Test template values are placeholders; never use IDs as features. Inspect measured family_comparison, model_details and actual tracebacks; fix errors before adding complexity. Do not invent results.
Call run_experiments with experiments:[{hypothesis,code,decision},...] for two independent candidates when runtime.parallel_trials=2 and slots remain; otherwise call run_experiment. code must define build_model(context) returning an UNFITTED estimator/Pipeline with fit(X,y), predict(X); classification uses original labels (possibly strings). Handle NaN, infinity, categorical/unknown values and preserve row order. Count string labels with np.unique(return_counts=True). CPU only, <=context.n_jobs threads, bounded sizes. No fit at import. Host refits on all train.
Only modelling imports: sklearn,catboost,lightgbm,xgboost,numpy,pandas,scipy,math,functools,collections,typing,itertools,operator,statistics,warnings,re. Only use installed packages. No files, networking, shell, package installation, dynamic eval or introspection. Propagate errors; deliberate fallback must expose fallback_reason. Use complete code, no duplicate attempts.
In decision give concise public Russian explanations: model_family, observations, rationale, alternatives, expected_effect, risks, previous_result_assessment. Expectations are untested. Do not supply private internal reasoning. Write hypothesis and finish reason in Russian. After a successful generated trial call finish_research(reason) when more experiments are unhelpful. Return tool calls, or JSON {"tool":"run_experiments","arguments":{"experiments":[...]}} / {"tool":"run_experiment","arguments":{"hypothesis":"...","code":"..."}}.
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
               "data_profile": {k: v for k, v in profile.items() if k not in {"research_strategy", "family_comparison"}},
               "research_strategy": profile.get("research_strategy", {}), "experiments": ledger, "best_code": _read_code(best),
               "last_failed_code": _read_code(failed, 10000), "runtime": runtime_context,
               "previous_tool_feedback": feedback[-5000:], "family_comparison": profile.get("family_comparison", {})}
    limit = max(5000, context_chars // 2) if compact else max(5000, context_chars)
    if compact:
        # Actually compact auxiliary context even when it is below context_chars.
        # Keep the most recent balanced assistant/tool group in the caller so a
        # failed experiment remains actionable under a tight token budget.
        content["data_profile"] = {key: profile[key] for key in (
            "train_shape", "test_shape", "feature_count", "categorical_columns", "target", "available_packages") if key in profile}
        comparison = profile.get("family_comparison", {})
        content["family_comparison"] = {"trials": [{k: row.get(k) for k in ("family", "trial_id", "status", "metric", "seconds")} for row in comparison.get("trials", [])], "selected_trial": comparison.get("selected_trial")}
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
    return [{"role": "system", "content": COMPACT_SYSTEM_PROMPT if compact else SYSTEM_PROMPT}, {"role": "user", "content": encoded}]
