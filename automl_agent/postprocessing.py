"""OOF-only selection and application of small prediction blends.

This module deliberately owns no model fitting. It selects a low-dimensional
postprocessing rule from aligned out-of-fold predictions and keeps the
assessment rows separate from the rows used to tune that rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


_WEIGHT_STEPS = 4
_CLASS_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
_ALIASES = {
    "root_mean_squared_error": "rmse",
    "macro_f1_score": "macro_f1",
    "f1_macro": "macro_f1",
    "weighted_f1_score": "weighted_f1",
    "auc": "roc_auc",
    "logloss": "log_loss",
}
_REGRESSION_METRICS = {"rmse", "mae", "r2"}
_CLASSIFICATION_METRICS = {
    "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "f1",
    "roc_auc", "log_loss",
}
_PROBABILITY_METRICS = {"roc_auc", "log_loss"}


def _metric_name(metric: str) -> str:
    normalized = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(normalized, normalized)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _same_label(left: Any, right: Any) -> bool:
    try:
        result = left == right
        return bool(result) if np.ndim(result) == 0 else False
    except Exception:
        return False


def _ordered_classes(classes: Sequence[Any], y: np.ndarray) -> list[Any]:
    result = list(classes)
    if not result:
        for label in y.tolist():
            if not any(_same_label(label, existing) for existing in result):
                result.append(label)
    return result


def _class_index(label: Any, classes: Sequence[Any]) -> int:
    for index, candidate in enumerate(classes):
        if _same_label(label, candidate):
            return index
    raise ValueError(f"Prediction contains a class absent from classes: {label!r}.")


def _encode_labels(values: np.ndarray, classes: Sequence[Any]) -> np.ndarray:
    """Vectorize label-to-class encoding for the hot Macro F1 search loop."""
    values = np.asarray(values)
    encoded = np.full(len(values), -1, dtype=np.int64)
    for index, label in enumerate(classes):
        try:
            mask = np.asarray(values == label)
            if mask.shape != values.shape:
                raise ValueError
        except Exception:
            mask = np.asarray([_same_label(value, label) for value in values], dtype=bool)
        encoded[mask] = index
    return encoded


def _macro_f1_encoded(actual: np.ndarray, predicted: np.ndarray, class_count: int) -> float | None:
    """Macro F1 from encoded labels; include every declared class."""
    if class_count == 0:
        return None
    valid_actual = actual >= 0
    if not valid_actual.any():
        return None
    # Keep one extra prediction column for labels outside the declared class
    # axis: those are false negatives, but are not counted as any class's FP.
    predicted = np.asarray(predicted).copy()
    predicted[predicted < 0] = class_count
    width = class_count + 1
    confusion = np.bincount(
        actual[valid_actual] * width + predicted[valid_actual],
        minlength=class_count * width,
    ).reshape(class_count, width)
    true_positive = np.diag(confusion[:, :class_count]).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted_count = confusion[:, :class_count].sum(axis=0).astype(float)
    denominator = support + predicted_count
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(per_class.mean())


def _macro_f1_score(y: np.ndarray, predictions: np.ndarray, classes: Sequence[Any]) -> float | None:
    """Encode labels once and score the full declared class axis."""
    return _macro_f1_encoded(
        _encode_labels(y, classes), _encode_labels(predictions, classes), len(classes)
    )


def _ordered_unique(values: np.ndarray) -> list[Any]:
    unique: list[Any] = []
    for value in values.tolist():
        if not any(_same_label(value, existing) for existing in unique):
            unique.append(value)
    try:
        return sorted(unique)
    except TypeError:
        # Keep the first-seen order when mixed fold ID types are not comparable.
        return unique


def _weights_grid(count: int) -> list[tuple[float, ...]]:
    if count == 1:
        return [(1.0,)]
    result: list[tuple[float, ...]] = []

    def add(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == count - 1:
            result.append(tuple(value / _WEIGHT_STEPS for value in (*prefix, remaining)))
            return
        for value in range(remaining, -1, -1):
            add((*prefix, value), remaining - value)

    add((), _WEIGHT_STEPS)
    return result


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("Each probability array must have shape (rows, classes).")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative.")
    totals = probabilities.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Each probability row must have a positive sum.")
    return probabilities / totals[:, None]


def _blend_probabilities(
    probabilities: Sequence[np.ndarray], weights: Sequence[float]
) -> np.ndarray:
    normalized = [_normalize_probability_rows(np.asarray(value, dtype=float)) for value in probabilities]
    if not normalized:
        raise ValueError("At least one probability array is required.")
    shape = normalized[0].shape
    if any(value.shape != shape for value in normalized):
        raise ValueError("Probability arrays must have identical aligned shapes.")
    blended = sum(weight * value for weight, value in zip(weights, normalized, strict=True))
    totals = blended.sum(axis=1)
    if (totals <= 0).any() or not np.isfinite(blended).all():
        raise ValueError("Blended probabilities are invalid.")
    return blended / totals[:, None]


def _apply_multiplier(probabilities: np.ndarray, multipliers: Sequence[float] | None) -> np.ndarray:
    if multipliers is None:
        return probabilities
    values = np.asarray(multipliers, dtype=float)
    if values.shape != (probabilities.shape[1],) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("class_multipliers must contain one finite nonnegative value per class.")
    adjusted = probabilities * values[None, :]
    totals = adjusted.sum(axis=1)
    # A zero multiplier vector is never generated by the selector, but fail
    # clearly if a caller supplies one to the application helper.
    if (totals <= 0).any() or not np.isfinite(adjusted).all():
        raise ValueError("Adjusted probabilities must have a positive finite row sum.")
    return adjusted / totals[:, None]


def _weighted_vote(
    predictions: Sequence[np.ndarray], classes: Sequence[Any], weights: Sequence[float]
) -> np.ndarray:
    labels = np.empty(len(predictions[0]), dtype=object)
    for row in range(len(labels)):
        votes = np.zeros(len(classes), dtype=float)
        for prediction, weight in zip(predictions, weights, strict=True):
            votes[_class_index(prediction[row], classes)] += weight
        # np.argmax makes ties deterministic according to the declared class axis.
        labels[row] = classes[int(np.argmax(votes))]
    return labels


def apply_postprocessing(
    predictions: Sequence[np.ndarray],
    probabilities: Sequence[np.ndarray] | None,
    classes: Sequence[Any],
    weights: Sequence[float],
    class_multipliers: Sequence[float] | None,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply a normalized weighted blend; return labels/values and probabilities."""
    if not predictions:
        raise ValueError("At least one prediction array is required.")
    arrays = [np.asarray(value) for value in predictions]
    if any(value.ndim != 1 for value in arrays) or any(len(value) != len(arrays[0]) for value in arrays):
        raise ValueError("Prediction arrays must be one-dimensional and aligned.")
    weight_values = np.asarray(weights, dtype=float)
    if weight_values.shape != (len(arrays),) or not np.isfinite(weight_values).all() or (weight_values < 0).any():
        raise ValueError("weights must contain one finite nonnegative value per model.")
    total_weight = float(weight_values.sum())
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value.")
    weight_values /= total_weight

    if task_type == "regression":
        numeric = [np.asarray(value, dtype=float) for value in arrays]
        if any(not np.isfinite(value).all() for value in numeric):
            raise ValueError("Regression predictions must be finite.")
        blended = sum(weight * value for weight, value in zip(weight_values, numeric, strict=True))
        return np.asarray(blended, dtype=float), None
    if task_type != "classification":
        raise ValueError("task_type must be 'classification' or 'regression'.")

    class_values = list(classes)
    if probabilities is not None:
        probability_arrays = [np.asarray(value, dtype=float) for value in probabilities]
        if len(probability_arrays) != len(arrays):
            raise ValueError("There must be one probability array per prediction array.")
        blended = _blend_probabilities(probability_arrays, weight_values)
        if blended.shape[0] != len(arrays[0]) or blended.shape[1] != len(class_values):
            raise ValueError("Probability rows/classes do not match predictions/classes.")
        adjusted = _apply_multiplier(blended, class_multipliers)
        labels = np.asarray([class_values[index] for index in np.argmax(adjusted, axis=1)])
        return labels, adjusted
    if class_multipliers is not None:
        raise ValueError("class_multipliers require class probabilities.")
    return _weighted_vote(arrays, class_values, weight_values), None


def _score(
    y: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    classes: Sequence[Any],
    metric: str,
) -> float | None:
    try:
        if metric in _REGRESSION_METRICS:
            actual = np.asarray(y, dtype=float)
            predicted = np.asarray(predictions, dtype=float)
            if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
                return None
            if metric == "rmse":
                value = float(np.sqrt(mean_squared_error(actual, predicted)))
            elif metric == "mae":
                value = float(mean_absolute_error(actual, predicted))
            else:
                if len(actual) < 2:
                    return None
                value = float(r2_score(actual, predicted))
        elif metric in _CLASSIFICATION_METRICS:
            labels = list(classes)
            if metric in _PROBABILITY_METRICS:
                if probabilities is None:
                    return None
                probs = _normalize_probability_rows(np.asarray(probabilities, dtype=float))
                if metric == "log_loss":
                    value = float(log_loss(y, probs, labels=labels))
                elif len(labels) == 2:
                    positive = next((label for label in labels if _same_label(label, 1)), None)
                    if positive is None:
                        try:
                            positive = sorted(labels)[-1]
                        except TypeError:
                            positive = labels[-1]
                    positive_index = _class_index(positive, labels)
                    binary_y = np.asarray([_same_label(value, positive) for value in y], dtype=int)
                    value = float(roc_auc_score(binary_y, probs[:, positive_index]))
                else:
                    value = float(roc_auc_score(
                        y, probs, labels=labels, multi_class="ovr", average="macro"
                    ))
            elif metric == "macro_f1":
                macro_score = _macro_f1_score(y, predictions, labels)
                if macro_score is None:
                    return None
                value = macro_score
            elif metric == "weighted_f1":
                value = float(f1_score(y, predictions, labels=labels, average="weighted", zero_division=0))
            elif metric == "accuracy":
                value = float(accuracy_score(y, predictions))
            elif metric == "balanced_accuracy":
                value = float(balanced_accuracy_score(y, predictions))
            else:
                if len(labels) != 2:
                    return None
                positive = next((label for label in labels if _same_label(label, 1)), None)
                if positive is None:
                    try:
                        positive = sorted(labels)[-1]
                    except TypeError:
                        positive = labels[-1]
                value = float(f1_score(
                    y, predictions, average="binary", pos_label=positive, zero_division=0
                ))
        else:
            return None
    except (ValueError, TypeError, IndexError, FloatingPointError):
        return None
    return value if math.isfinite(value) else None


def _better(candidate: float | None, current: float | None, maximize: bool) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return candidate > current if maximize else candidate < current


def _model_probabilities(
    all_probabilities: Sequence[np.ndarray] | None,
    model_indices: Sequence[int],
    rows: np.ndarray,
) -> list[np.ndarray] | None:
    if all_probabilities is None:
        return None
    return [all_probabilities[index][rows] for index in model_indices]


def _tune_multipliers(
    y: np.ndarray,
    blended_probabilities: np.ndarray,
    classes: Sequence[Any],
    maximize: bool,
) -> tuple[list[float], float | None]:
    """Tune only the selected blend, without repeating the weights grid."""
    multipliers = [1.0] * len(classes)
    encoded_y = _encode_labels(y, classes)

    def score(candidate: Sequence[float]) -> float | None:
        # Row normalization cannot change argmax, so skip that allocation in
        # the hot coordinate-search loop. Final application still normalizes.
        prediction_indices = np.argmax(
            blended_probabilities * np.asarray(candidate, dtype=float)[None, :], axis=1
        )
        return _macro_f1_encoded(encoded_y, prediction_indices, len(classes))

    best_score = score(multipliers)
    for _ in range(2):
        for class_index in range(len(classes)):
            best_value = multipliers[class_index]
            for value in _CLASS_MULTIPLIERS:
                candidate = list(multipliers)
                candidate[class_index] = value
                candidate_score = score(candidate)
                if _better(candidate_score, best_score, maximize):
                    best_score = candidate_score
                    best_value = value
            multipliers[class_index] = best_value
    return multipliers, best_score


def _fit_meta(
    y: np.ndarray,
    predictions: Sequence[np.ndarray],
    probabilities: Sequence[np.ndarray] | None,
    classes: Sequence[Any],
    metric: str,
    maximize: bool,
    task_type: str,
) -> tuple[list[float], list[float] | None, float | None]:
    """Search weight grid first, then calibrate only its best Macro F1 blend."""
    best_weights = [1.0] + [0.0] * (len(predictions) - 1)
    best_multipliers: list[float] | None = None
    best_score: float | None = None

    for weights_tuple in _weights_grid(len(predictions)):
        weights = list(weights_tuple)
        if task_type == "regression":
            fitted, _ = apply_postprocessing(predictions, None, classes, weights, None, task_type)
            multipliers = None
            score = _score(y, fitted, None, classes, metric)
        else:
            multipliers = None
            labels, blended_probabilities = apply_postprocessing(
                predictions, probabilities, classes, weights, None, task_type
            )
            score = _score(y, labels, blended_probabilities, classes, metric)
        if _better(score, best_score, maximize):
            best_weights = weights
            best_multipliers = multipliers
            best_score = score

    if (
        task_type == "classification"
        and metric == "macro_f1"
        and probabilities is not None
        and len(classes) >= 2
    ):
        _, blended_probabilities = apply_postprocessing(
            predictions, probabilities, classes, best_weights, None, task_type
        )
        assert blended_probabilities is not None
        best_multipliers, best_score = _tune_multipliers(
            y, blended_probabilities, classes, maximize
        )
    return best_weights, best_multipliers, best_score


def _evaluate_rule(
    y: np.ndarray,
    predictions: Sequence[np.ndarray],
    probabilities: Sequence[np.ndarray] | None,
    classes: Sequence[Any],
    weights: Sequence[float],
    multipliers: Sequence[float] | None,
    metric: str,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    values, adjusted_probabilities = apply_postprocessing(
        predictions, probabilities, classes, weights, multipliers, task_type
    )
    score = _score(y, values, adjusted_probabilities, classes, metric)
    return values, adjusted_probabilities, score


def select_postprocessing(
    *,
    y: np.ndarray,
    predictions: list[np.ndarray],
    probabilities: list[np.ndarray] | None,
    classes: list[Any],
    fold_ids: np.ndarray,
    metric: str,
    maximize: bool,
    temporal: bool = False,
) -> dict[str, Any]:
    """Select a small blend using cross-fitted or strictly forward OOF scoring.

    Models are expected in best-first order and all arrays must already be
    aligned to the same OOF rows and class axis. The reported assessed score is
    computed only on rows whose fold was excluded from meta-training.
    """
    y_values = np.asarray(y)
    folds = np.asarray(fold_ids)
    if y_values.ndim != 1 or folds.ndim != 1 or len(y_values) != len(folds):
        raise ValueError("y and fold_ids must be one-dimensional and aligned.")
    if not predictions:
        raise ValueError("At least one model prediction array is required.")
    if len(y_values) == 0:
        raise ValueError("OOF arrays must not be empty.")

    warnings: list[str] = []
    model_predictions = [np.asarray(value) for value in predictions]
    if len(model_predictions) > 3:
        model_predictions = model_predictions[:3]
        warnings.append("Использованы только первые три модели в порядке best-first.")
    if any(value.ndim != 1 or len(value) != len(y_values) for value in model_predictions):
        raise ValueError("Every prediction array must be one-dimensional and aligned with y.")

    metric_value = _metric_name(metric)
    task_type = "regression" if metric_value in _REGRESSION_METRICS else "classification"
    if metric_value not in _REGRESSION_METRICS | _CLASSIFICATION_METRICS:
        return {
            "accepted": False,
            "weights": [1.0] + [0.0] * (len(model_predictions) - 1),
            "class_multipliers": None,
            "baseline_score": None,
            "assessed_score": None,
            "calibrated_oof_score": None,
            "assessment": "forward" if temporal else "cross_fitted",
            "fold_scores": [],
            "warnings": [f"Метрика {metric!r} не поддерживается для postprocessing."],
        }

    model_probabilities: list[np.ndarray] | None = None
    class_values = [] if task_type == 'regression' else _ordered_classes(classes, y_values)
    if probabilities is not None:
        if len(probabilities) < len(model_predictions):
            raise ValueError("There must be a probability array for every selected model.")
        model_probabilities = [np.asarray(value, dtype=float) for value in probabilities[:len(model_predictions)]]
        expected_shape = (len(y_values), len(class_values))
        if any(value.shape != expected_shape for value in model_probabilities):
            raise ValueError("Probability arrays must match the OOF rows and declared class axis.")
        # Validate data up front so that a bad late fold cannot partially score.
        model_probabilities = [_normalize_probability_rows(value) for value in model_probabilities]
    elif metric_value in _PROBABILITY_METRICS:
        warnings.append(f"Метрика {metric_value} требует вероятности; postprocessing пропущен.")
        return {
            "accepted": False,
            "weights": [1.0] + [0.0] * (len(model_predictions) - 1),
            "class_multipliers": None,
            "baseline_score": None,
            "assessed_score": None,
            "calibrated_oof_score": None,
            "assessment": "forward" if temporal else "cross_fitted",
            "fold_scores": [],
            "warnings": warnings,
        }
    if task_type == "classification":
        if len(class_values) < 2:
            warnings.append("Для классификации нужны как минимум два известных класса.")
            return {
                "accepted": False,
                "weights": [1.0] + [0.0] * (len(model_predictions) - 1),
                "class_multipliers": None,
                "baseline_score": None,
                "assessed_score": None,
                "calibrated_oof_score": None,
                "assessment": "forward" if temporal else "cross_fitted",
                "fold_scores": [],
                "warnings": warnings,
            }
        if metric_value == "macro_f1" and model_probabilities is None:
            warnings.append("Вероятностей нет; class multipliers отключены, доступно только голосование.")
        if metric_value == "f1" and len(class_values) != 2:
            warnings.append("Метрика f1 поддерживается только для бинарной классификации.")
            return {
                "accepted": False,
                "weights": [1.0] + [0.0] * (len(model_predictions) - 1),
                "class_multipliers": None,
                "baseline_score": None,
                "assessed_score": None,
                "calibrated_oof_score": None,
                "assessment": "forward" if temporal else "cross_fitted",
                "fold_scores": [],
                "warnings": warnings,
            }

    if task_type == "regression":
        try:
            y_values = np.asarray(y_values, dtype=float)
            model_predictions = [np.asarray(value, dtype=float) for value in model_predictions]
        except (TypeError, ValueError) as exc:
            raise ValueError("Regression targets and predictions must be numeric.") from exc
        if not np.isfinite(y_values).all() or any(not np.isfinite(value).all() for value in model_predictions):
            raise ValueError("Regression targets and predictions must be finite.")

    unique_folds = _ordered_unique(folds)
    if len(unique_folds) < 2:
        weights, multipliers, calibrated_score = _fit_meta(
            y_values, model_predictions, model_probabilities, class_values,
            metric_value, maximize, task_type,
        )
        baseline = _score(
            y_values,
            model_predictions[0],
            model_probabilities[0] if model_probabilities is not None else None,
            class_values,
            metric_value,
        )
        warnings.append("Недостаточно OOF folds для независимой оценки: нужно минимум два; выбор не принят.")
        return {
            "accepted": False,
            "weights": [float(value) for value in weights],
            "class_multipliers": [float(value) for value in multipliers] if multipliers is not None else None,
            "baseline_score": baseline,
            "assessed_score": None,
            "calibrated_oof_score": calibrated_score,
            "assessment": "forward" if temporal else "cross_fitted",
            "fold_scores": [],
            "warnings": warnings,
        }

    heldout_indices: list[np.ndarray] = []
    heldout_outputs: list[np.ndarray] = []
    heldout_output_probabilities: list[np.ndarray] = []
    heldout_fold_rows: list[dict[str, Any]] = []
    for fold_position, fold in enumerate(unique_folds):
        train_folds = unique_folds[:fold_position] if temporal else [value for value in unique_folds if not _same_label(value, fold)]
        if not train_folds:
            continue
        holdout_mask = np.asarray([_same_label(value, fold) for value in folds], dtype=bool)
        train_mask = np.asarray(
            [any(_same_label(value, allowed) for allowed in train_folds) for value in folds],
            dtype=bool,
        )
        test_rows = np.flatnonzero(holdout_mask)
        train_rows = np.flatnonzero(train_mask)
        if len(test_rows) == 0 or len(train_rows) == 0:
            continue
        fold_predictions_train = [value[train_rows] for value in model_predictions]
        fold_probabilities_train = _model_probabilities(model_probabilities, range(len(model_predictions)), train_rows)
        weights, multipliers, _ = _fit_meta(
            y_values[train_rows], fold_predictions_train, fold_probabilities_train,
            class_values, metric_value, maximize, task_type,
        )
        fold_predictions_test = [value[test_rows] for value in model_predictions]
        fold_probabilities_test = _model_probabilities(model_probabilities, range(len(model_predictions)), test_rows)
        values, adjusted_probabilities, assessed_fold_score = _evaluate_rule(
            y_values[test_rows], fold_predictions_test, fold_probabilities_test,
            class_values, weights, multipliers, metric_value, task_type,
        )
        heldout_outputs.append(np.asarray(values))
        if adjusted_probabilities is not None:
            heldout_output_probabilities.append(np.asarray(adjusted_probabilities))
        if task_type == "regression":
            baseline_values = fold_predictions_test[0]
            baseline_probabilities = None
        else:
            baseline_values = fold_predictions_test[0]
            baseline_probabilities = fold_probabilities_test[0] if fold_probabilities_test is not None else None
        baseline_fold_score = _score(
            y_values[test_rows], baseline_values, baseline_probabilities, class_values, metric_value
        )
        heldout_indices.extend(test_rows.tolist())
        heldout_fold_rows.append({
            "fold": _json_value(fold),
            "train_folds": [_json_value(value) for value in train_folds],
            "rows": int(len(test_rows)),
            "baseline_score": baseline_fold_score,
            "assessed_score": assessed_fold_score,
        })

    if not heldout_indices:
        warnings.append("Не удалось сформировать независимую оценку по OOF folds.")
        weights = [1.0] + [0.0] * (len(model_predictions) - 1)
        multipliers = None
        return {
            "accepted": False,
            "weights": weights,
            "class_multipliers": multipliers,
            "baseline_score": None,
            "assessed_score": None,
            "calibrated_oof_score": None,
            "assessment": "forward" if temporal else "cross_fitted",
            "fold_scores": heldout_fold_rows,
            "warnings": warnings,
        }

    all_heldout = np.asarray(heldout_indices, dtype=int)
    heldout_predictions = [value[all_heldout] for value in model_predictions]
    heldout_probabilities = _model_probabilities(model_probabilities, range(len(model_predictions)), all_heldout)
    baseline_score = _score(
        y_values[all_heldout],
        heldout_predictions[0],
        heldout_probabilities[0] if heldout_probabilities is not None else None,
        class_values,
        metric_value,
    )
    cross_fitted_values = np.concatenate(heldout_outputs)
    cross_fitted_probabilities = (
        np.concatenate(heldout_output_probabilities, axis=0)
        if heldout_output_probabilities else None
    )
    assessed_score = _score(
        y_values[all_heldout], cross_fitted_values, cross_fitted_probabilities,
        class_values, metric_value,
    )

    final_weights, final_multipliers, calibrated_score = _fit_meta(
        y_values, model_predictions, model_probabilities, class_values,
        metric_value, maximize, task_type,
    )
    accepted = _better(assessed_score, baseline_score, maximize)
    if assessed_score is None or baseline_score is None:
        warnings.append("Метрика не определена на независимых validation rows; выбор не принят.")
    elif not accepted:
        warnings.append("OOF postprocessing не улучшил основную метрику на независимых folds.")

    return {
        "accepted": bool(accepted),
        "weights": [float(value) for value in final_weights],
        "class_multipliers": [float(value) for value in final_multipliers] if final_multipliers is not None else None,
        "baseline_score": baseline_score,
        "assessed_score": assessed_score,
        "calibrated_oof_score": calibrated_score,
        "assessment": "forward" if temporal else "cross_fitted",
        "fold_scores": heldout_fold_rows,
        "warnings": warnings,
    }
