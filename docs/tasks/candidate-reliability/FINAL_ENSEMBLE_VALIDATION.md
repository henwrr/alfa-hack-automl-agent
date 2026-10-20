# Final ensemble refit

`automl_agent.final_ensemble.run_final_ensemble(manifest)` refits one to three
selected members sequentially on all training rows. It reads only the three
declared train/test inputs, rejects validation/OOF labels, validates the target
and feature schemas, and checks normalized positive member weights and the
classification multiplier axis before candidate code is loaded.

Every member is trained through `training.fit_model`. The final fit disables a
new early-stopping probe and passes the host-selected `iteration_override`,
while preserving sample-weight options. The returned details include the
member trial IDs, fitted estimator details, per-member training summaries,
member weights, class multipliers, train row count, and final iteration counts.

The worker writes test predictions, aligned classification probabilities and
class labels, `worker_result.json`, and optionally `model.joblib` atomically.
The saved prediction CSV uses round-trip float precision; the probability NPY
preserves the computed array exactly. The regression and classification tests
reload the serialized ensemble and verify that it reproduces the stored output.

## Verification

- RED: the module imported successfully, but six behavior tests failed at the
  callable `NotImplementedError` stub.
- GREEN: `tests/test_final_ensemble.py` — 9 passed, including weighted CatBoost
  with a 7-tree override, an ExtraTrees classification blend with string labels,
  an ExtraTrees regression blend, serialization reload, and invalid input/schema
  rejection.
- No public-data fit, Ollama call, or host submission write was performed.
