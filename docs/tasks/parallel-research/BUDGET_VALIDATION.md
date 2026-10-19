# Adaptive finalization reserve

`plan_finalization` is a deterministic estimate from the current durable winner. With total budget `T`, startup reserve is `0.15 × T`. A measurement is usable only when the result status is `ok` and its measured wall time is finite and positive; error, timeout, cancelled, interrupted, or malformed results fall back to the startup reserve.

For a usable winner, the projected full-data fit time is `measured_seconds × row_scale × 1.5`. When the original training row count `n` is known, `row_scale = n / (n - ceil(n × validation_fraction))`, matching the evaluator's validation split. Otherwise it is `1 / (1 - validation_fraction)`. The additional write allowance is `min(T, max(5 seconds, 0.01 × T))`. The selected reserve is `min(T, max(0.05 × T, projected_fit + write_allowance))`. `feasible` reports whether the projected fit plus write allowance fits in `T`; clamping the reserve does not hide an over-budget estimate.

`total_seconds` must be finite and positive, `validation_fraction` finite and in `[0, 1)`, and supplied `train_rows` a positive integer that leaves at least one fit row. Invalid arguments raise `ValueError`. Rebuilding the same `ExperimentResult` from the ledger produces the same plan; the function reads no clock and stores no history.

This estimate assumes fit time scales approximately with row count and adds a fixed 1.5 safety multiplier. Actual runtime, data-dependent preprocessing, filesystem load, and model behavior can differ, so the plan does not guarantee finalization will finish. The coordinator still has to enforce the overall run deadline.

## Validation evidence

The clean pre-change suite passed: 90 tests (`logs/20261019T203120.612660Z-baseline.json`). The focused red run failed all 27 then-current budgeting assertions at the deliberate unimplemented stubs, with no collection/import errors (`logs/20261019T203149.781973Z-red.json`). The final focused suite passed 33 tests (`logs/20261019T203240.232406Z-green.json`).

`logs/20261019T203051.149696Z-baseline.json` is an earlier broad capture taken after the new tests were added; it includes their expected stub failures and is superseded by the clean baseline above.
