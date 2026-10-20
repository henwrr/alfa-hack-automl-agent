# OOF postprocessing validation

`automl_agent/postprocessing.py` tunes a deterministic blend from aligned OOF
outputs only. It evaluates each held-out fold with weights and, for Macro F1,
class multipliers learned without that fold. IID folds train on all other folds;
temporal folds train only on earlier fold IDs. The first model is the baseline
on the same assessed rows. The final weights are refit on all OOF rows and
`calibrated_oof_score` is therefore descriptive, not an independent estimate.

The blend search uses weights on a 0.25 grid (up to three best-first models).
Macro F1 class multipliers use a bounded two-pass coordinate search over
`[0.5, 0.75, 1, 1.25, 1.5, 2]`. Other supported classification metrics use
their corresponding label or probability score; regression supports RMSE, MAE,
and R2. Results that lack two folds, a required probability array, or a defined
metric are not accepted.

Cross-fitting reduces direct leakage from the evaluated fold, but does not
guarantee generalization. Trying many postprocessing rules or using the same OOF
results repeatedly still makes observed OOF gains optimistic. Class multipliers
are a small probability calibration heuristic; they do not reason over raw LLM
outputs and do not replace a fresh validation set.

Validation: the new tests were first run before implementation and failed at
collection because `automl_agent.postprocessing` did not exist. After the
implementation, all 13 focused tests passed, including IID and forward fold
selection, strict no-gain rejection, string/numeric classes, probability
normalization, and regression blends. The full offline suite passed: 280 tests
in 74.74 seconds;
two existing sklearn convergence warnings were emitted by Linear-family tests.
A synthetic 87,670-row / 10-class / 3-model / 5-fold Macro F1 smoke run took
2.99 seconds after bounding multiplier tuning to the best weight grid point.
Macro F1's encoded confusion-count implementation matched sklearn on 300
randomized class-label cases. No public-data full fit or Ollama run was
performed.
