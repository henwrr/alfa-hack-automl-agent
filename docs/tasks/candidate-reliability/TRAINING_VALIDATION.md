# Fit protocol validation

`automl_agent/training.py` adds host-owned `fit_model` for native CatBoost,
LightGBM, and XGBoost estimators and sklearn pipelines. Early stopping clones a
probe, fits its preprocessing only on inner-train rows, evaluates on inner
validation rows, then clones the original model and refits all supplied
outer-training rows with the selected tree count. The API has no outer
validation target. Optional per-row balanced weights are recomputed from each
fit's training labels and routed to the final pipeline step; constructor class
weighting combined with host weights is rejected.

XGBoost classification string labels use a serializable sklearn wrapper that
preserves original `classes_`, `predict`, and `predict_proba`, while retaining
the trusted fitted native estimator at `.estimator` and `._model`. A supplied
`iteration_override` bypasses the probe for native boosters. Ordinary sklearn
estimators ignore that host tree-count hint and report no requested or actual
tree count; requesting early stopping for such an estimator still raises.

Validation:

- Initial behavioral red used an importable `training.py` stub; the test module
  collected successfully and its first 9 behavior tests failed at the explicit
  `NotImplementedError`.
- A follow-up red exposed a reporting bug for sklearn `iteration_override`:
  ExtraTrees correctly kept 5 estimators but the summary incorrectly reported
  5 as `requested_iterations`. The implementation now reports `None` and marks
  the override as ignored.
- `/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest
  tests/test_training.py -q` — **18 passed**. This includes real tiny native
  CatBoost/LightGBM/XGBoost probes and refits, train-only weights, the 10-tree
  native preflight override with joblib reload and prediction, sklearn override
  behavior, pipeline preprocessing and weight routing, XGBoost string-label
  decoding after reload, and chronological/local inner indices.

No public dataset fit, Ollama request, or full project suite was run. Small
native estimator fixtures are used so the preflight and serialization path can
be checked before any expensive CV folds.
