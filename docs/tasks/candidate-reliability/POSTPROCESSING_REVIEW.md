# Postprocessing integration review

Reviewed the root integration at main HEAD `eee59fa` plus the then-current
working-tree changes in `automl_agent/agent.py`, `automl_agent/worker.py`,
`automl_agent/ensemble.py`, and `tests/test_ensemble_selection.py`. This was a
read-only review; no main files, model fits, public data, or Ollama calls were
used. The review did not run the full test suite.

## Remaining finding

**P2 — the research reserve can undercount a repeated-family ensemble.**
`ResearchRun._update_reserve` adds only the first successful estimate for each
family. `choose_postprocessing` first chooses distinct families, then fills
remaining slots from the full ranked list, which can select multiple models
from a family when fewer than three families are available. The reserve then
omits one or more full refits. The later `choose_postprocessing` time check
recomputes the active members' cost and rejects the blend when there is not
enough remaining time, so it does not start an underfunded ensemble. The
consequence is that research may consume time that would have kept the blend
feasible, and `potential_ensemble_fit_seconds` may be too low. Reserve against
the same at-most-three candidate policy, including repeated families, or limit
the candidate list to the family-diverse entries represented in the reserve.

## Verified behavior

- `choose_postprocessing` filters candidates by the current split fingerprint,
  sorts them by the measured task metric, and passes each candidate's saved
  `oof_fold_ids.npy` only after checking that the fold arrays match. The CV
  worker writes those IDs in the same eligible-row order as the OOF targets and
  predictions. Forward CV uses integer fold numbers in chronological order;
  `temporal=True` therefore trains each meta rule only on earlier folds.
- `baseline_score` and `assessed_score` cover the same meta-assessed held-out
  rows. Each held-out fold gets weights learned without that fold; final weights
  and class multipliers are then refit on all OOF rows. Consequently,
  `calibrated_oof_score` is a full-OOF meta score, not an unbiased estimate.
  The plan warning states that base models were already selected adaptively on
  those OOF results and that this is neither nested CV nor an independent test
  score.
- The selected positive weights and class multipliers are applied first to the
  host-generated test predictions and then carried into the `finalize_ensemble`
  manifest. Zero-weight members are omitted from the final manifest; the final
  worker validates positive weights that sum to one, refits the retained
  members on all training rows, aligns probability axes to task classes, and
  applies the same class multipliers through `PredictionEnsemble`.
- `eee59fa` has added `automl_agent/final_ensemble.py` and the worker dispatch
  used by `ResearchRun.finalize`; the final ensemble refit path is now present
  in main. The agent's accepted-plan path publishes the meta-prediction before
  final refit and overwrites it with the full-data refit output on success.

No other P1/P2 issue was found in the reviewed baseline/assessed/calibrated
score flow, OOF fold alignment, temporal rule, or positive-weight final refit
path.

Root: условный P2 устранён. Выбор списка потенциальных участников вынесен в общий `ensemble_shortlist`; и резерв, и postprocessing используют один и тот же список, включая дополнение повторными семействами. Добавлен regression test на резерв для трёх моделей одного семейства.
