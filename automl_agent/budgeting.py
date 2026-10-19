"""Pure policies for reserving time for final model fitting."""

from __future__ import annotations

import math
from numbers import Integral, Real

from automl_agent.contracts import ExperimentResult

_INITIAL_FRACTION = 0.15
_MINIMUM_FRACTION = 0.05
_SAFETY_FACTOR = 1.5


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _total_budget(value: object) -> float:
    seconds = _finite_number(value, "total_seconds")
    if seconds <= 0:
        raise ValueError("total_seconds must be positive")
    return seconds


def _measurement_seconds(best: ExperimentResult | None) -> float | None:
    if best is None or getattr(best, "status", None) != "ok":
        return None
    raw_seconds = getattr(best, "seconds", None)
    try:
        seconds = _finite_number(raw_seconds, "best.seconds")
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _initial_plan(total_seconds: float) -> dict:
    return {
        "reserve_seconds": initial_reserve_seconds(total_seconds),
        "estimated_fit_seconds": None,
        "source": "initial",
        "selected_trial": None,
        "row_scale": None,
        "safety_factor": None,
        "overhead_seconds": 0.0,
        "feasible": True,
    }


def initial_reserve_seconds(total_seconds: float) -> float:
    """Return the startup reserve (15% of budget), without a fixed cap."""
    return _total_budget(total_seconds) * _INITIAL_FRACTION


def plan_finalization(
    *,
    total_seconds: float,
    validation_fraction: float,
    best: ExperimentResult | None = None,
    train_rows: int | None = None,
) -> dict:
    """Estimate a full-data fit deadline from a successful validation trial.

    This is a policy estimate, not a runtime guarantee. Recreating it from the
    same durable winning result yields the same plan because it reads no clock
    and keeps no history.
    """
    total = _total_budget(total_seconds)
    fraction = _finite_number(validation_fraction, "validation_fraction")
    if fraction < 0 or fraction >= 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    if train_rows is None:
        row_scale = 1.0 / (1.0 - fraction)
    else:
        if isinstance(train_rows, bool) or not isinstance(train_rows, Integral) or train_rows <= 0:
            raise ValueError("train_rows must be a positive integer")
        validation_rows = math.ceil(int(train_rows) * fraction)
        fit_rows = int(train_rows) - validation_rows
        if fit_rows <= 0:
            raise ValueError("train_rows and validation_fraction leave no fit rows")
        row_scale = int(train_rows) / fit_rows

    measured_seconds = _measurement_seconds(best)
    if measured_seconds is None:
        return _initial_plan(total)

    estimated_fit = measured_seconds * row_scale * _SAFETY_FACTOR
    # Reserve time for writing artifacts and the final result. Scale with the
    # run while retaining a small absolute minimum for short runs.
    overhead = min(total, max(5.0, total * 0.01))
    projected_cost = estimated_fit + overhead
    reserve = min(total, max(total * _MINIMUM_FRACTION, projected_cost))
    return {
        "reserve_seconds": reserve,
        "estimated_fit_seconds": estimated_fit,
        "source": "best",
        "selected_trial": getattr(best, "trial_id", None),
        "row_scale": row_scale,
        "safety_factor": _SAFETY_FACTOR,
        "overhead_seconds": overhead,
        "feasible": projected_cost <= total,
    }
