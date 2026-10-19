"""Validated, declarative model recipes with standalone candidate generators."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from . import families


_FAMILIES = ("CatBoost", "ExtraTrees", "Linear")
_TASK_TYPES = ("classification", "regression")
_HOST_CONTROLLED = {"n_jobs", "thread_count", "random_state", "random_seed", "seed"}
_REJECTED_GUIDANCE = {
    "class_weights": (
        "class_weights is not a valid recipe parameter; use CatBoost auto_class_weights "
        "or sklearn class_weight."
    ),
    "sample_weight": (
        "sample_weight is per-row fit data, not a recipe parameter; use CatBoost "
        "auto_class_weights='Balanced' or 'SqrtBalanced', or sklearn class_weight="
        "'balanced' (ExtraTrees also accepts 'balanced_subsample')."
    ),
    "early_stopping_rounds": (
        "early_stopping_rounds needs a validated eval_set; recipe candidates do not "
        "receive one, so early stopping is unsupported."
    ),
}


def _task_type(task_type):
    if task_type not in _TASK_TYPES:
        raise ValueError("task_type must be 'classification' or 'regression'.")
    return task_type


def _family(family):
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {', '.join(_FAMILIES)}.")
    return family


def _default_parameters(family, task_type):
    if family == "CatBoost":
        parameters = {"iterations": 150, "auto_class_weights": None}
        if task_type == "classification":
            return parameters
        parameters.pop("auto_class_weights")
        return parameters
    if family == "ExtraTrees":
        parameters = {
            "n_estimators": 180,
            "max_leaf_nodes": 128,
            "min_samples_leaf": 3,
            "max_features": 0.8,
            "max_depth": None,
        }
        if task_type == "classification":
            parameters["class_weight"] = None
        return parameters
    if task_type == "classification":
        return {"C": 0.5, "max_iter": 300, "solver": "saga", "class_weight": None}
    return {}


def default_recipe(family, task_type):
    """Return a complete normalized recipe for one supported model family."""
    return normalize_recipe({"family": _family(family)}, _task_type(task_type))


def _integer(name, value, minimum, maximum):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}].")
    return value


def _number(name, value, minimum, maximum, *, exclusive_minimum=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        relation = "greater than" if exclusive_minimum else "at least"
        raise ValueError(f"{name} must be a finite number {relation} {minimum} and at most {maximum}.")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number.")
    minimum_valid = normalized > minimum if exclusive_minimum else normalized >= minimum
    if not minimum_valid or normalized > maximum:
        left = "(" if exclusive_minimum else "["
        raise ValueError(f"{name} must be in {left}{minimum}, {maximum}].")
    return normalized


def _strict_mapping(value, field_name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings.")
    return value


def _normalize_parameter(family, task_type, name, value):
    if name in _HOST_CONTROLLED:
        raise ValueError(f"{name} is host-controlled and cannot be changed in a recipe.")
    if name in _REJECTED_GUIDANCE:
        raise ValueError(_REJECTED_GUIDANCE[name])

    if family == "CatBoost":
        if name == "iterations":
            return _integer(name, value, 10, 1000)
        if name == "depth":
            return _integer(name, value, 3, 10)
        if name == "learning_rate":
            return _number(name, value, 0, 1, exclusive_minimum=True)
        if name == "l2_leaf_reg":
            return _number(name, value, 0, 100)
        if name == "border_count":
            return _integer(name, value, 16, 255)
        if name == "auto_class_weights":
            if task_type != "classification":
                raise ValueError("auto_class_weights is available only for classification.")
            if value == "None":
                return None
            if value not in (None, "Balanced", "SqrtBalanced"):
                raise ValueError("auto_class_weights must be None, 'Balanced', or 'SqrtBalanced'.")
            return value
    elif family == "ExtraTrees":
        if name == "n_estimators":
            return _integer(name, value, 10, 1000)
        if name == "max_leaf_nodes":
            return _integer(name, value, 2, 2048)
        if name == "min_samples_leaf":
            return _integer(name, value, 1, 100)
        if name == "max_depth":
            if value is None:
                return None
            return _integer(name, value, 2, 64)
        if name == "max_features":
            if value is None or value in ("sqrt", "log2"):
                return value
            return _number(name, value, 0, 1, exclusive_minimum=True)
        if name == "class_weight":
            if task_type != "classification":
                raise ValueError("class_weight is available only for classification.")
            if value not in (None, "balanced", "balanced_subsample"):
                raise ValueError("class_weight must be None, 'balanced', or 'balanced_subsample'.")
            return value
    else:
        if task_type == "classification":
            if name == "C":
                return _number(name, value, 0, 100, exclusive_minimum=True)
            if name == "max_iter":
                return _integer(name, value, 100, 3000)
            if name == "solver":
                if value != "saga":
                    raise ValueError("Linear classifier solver is fixed to 'saga'.")
                return "saga"
            if name == "class_weight":
                if value not in (None, "balanced"):
                    raise ValueError("Linear class_weight must be None or 'balanced'.")
                return value

    raise ValueError(f"Unknown or unsupported {family} recipe parameter {name!r} for {task_type}.")


def normalize_recipe(recipe, task_type):
    """Validate recipe shape, task-specific options, types, and bounds."""
    task_type = _task_type(task_type)
    recipe = _strict_mapping(recipe, "recipe")
    unknown_top_level = set(recipe) - {"family", "parameters", "preprocessing"}
    if unknown_top_level:
        key = sorted(unknown_top_level)[0]
        if key in _REJECTED_GUIDANCE:
            raise ValueError(_REJECTED_GUIDANCE[key])
        if key in _HOST_CONTROLLED:
            raise ValueError(f"{key} is host-controlled and cannot be changed in a recipe.")
        raise ValueError(f"Unknown recipe field {key!r}; allowed fields are family, parameters, preprocessing.")
    if "family" not in recipe:
        raise ValueError("recipe.family is required.")
    family = _family(recipe["family"])

    parameters = dict(_default_parameters(family, task_type))
    supplied_parameters = _strict_mapping(recipe.get("parameters", {}), "parameters")
    for name, value in supplied_parameters.items():
        parameters[name] = _normalize_parameter(family, task_type, name, value)

    preprocessing_defaults = {"add_missing_indicators": False}
    preprocessing = _strict_mapping(recipe.get("preprocessing", {}), "preprocessing")
    unknown_preprocessing = set(preprocessing) - set(preprocessing_defaults)
    if unknown_preprocessing:
        key = sorted(unknown_preprocessing)[0]
        raise ValueError(
            f"Unknown preprocessing option {key!r}; only add_missing_indicators is supported."
        )
    if "add_missing_indicators" in preprocessing:
        value = preprocessing["add_missing_indicators"]
        if type(value) is not bool:
            raise ValueError("add_missing_indicators must be a boolean.")
        if family == "CatBoost" and value:
            raise ValueError("CatBoost uses native missing-value handling; add_missing_indicators must be false.")
        preprocessing_defaults["add_missing_indicators"] = value

    return {
        "family": family,
        "parameters": parameters,
        "preprocessing": preprocessing_defaults,
    }


def recipe_code(recipe, task_type, n_jobs=4):
    """Build complete standalone Python source from a validated recipe."""
    normalized = normalize_recipe(recipe, task_type)
    if type(n_jobs) is not int or not 1 <= n_jobs <= 8:
        raise ValueError("n_jobs must be an integer in [1, 8]; worker parallelism is host-controlled.")
    source = families._source(n_jobs, normalized["family"])
    expected = "def build_model(context):"
    if source.count(expected) != 1:
        raise RuntimeError("The verified family template does not contain one build_model function.")
    source = source.replace(expected, "def _build_template_model(context):", 1)
    params = {f"model__{key}": value for key, value in normalized["parameters"].items()}
    if normalized["family"] == "CatBoost":
        # CatBoost treats the constructor default as unset and rejects JSON null
        # for auto_class_weights; a None recipe value means preserve that default.
        params = {key: value for key, value in params.items() if value is not None}
    wrapper = "\n\ndef build_model(context):\n    model = _build_template_model(context)\n"
    wrapper += f"    model.set_params(**{params!r})\n"
    if normalized["preprocessing"]["add_missing_indicators"]:
        wrapper += "    preprocess = model.named_steps['preprocess']\n"
        wrapper += "    if not any(name == 'numeric' for name, _, _ in preprocess.transformers):\n"
        wrapper += "        raise ValueError('add_missing_indicators requires at least one numeric feature column.')\n"
        wrapper += "    model.set_params(preprocess__numeric__impute__add_indicator=True)\n"
    wrapper += "    return model\n"
    return source.rstrip() + wrapper


def recipe_fingerprint(recipe, task_type):
    """Return SHA-256 of canonical normalized JSON for recipe deduplication."""
    normalized = normalize_recipe(recipe, task_type)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def recipe_catalog(task_type):
    """Describe supported recipe fields and safe options for a model task."""
    task_type = _task_type(task_type)
    families_catalog = {}
    if task_type == "classification":
        families_catalog["CatBoost"] = {
            "parameters": {
                "iterations": {"type": "integer", "minimum": 10, "maximum": 1000, "default": 150},
                "depth": {"type": "integer", "minimum": 3, "maximum": 10, "default": "native"},
                "learning_rate": {"type": "number", "exclusive_minimum": 0, "maximum": 1, "default": "native automatic"},
                "l2_leaf_reg": {"type": "number", "minimum": 0, "maximum": 100, "default": "native"},
                "border_count": {"type": "integer", "minimum": 16, "maximum": 255, "default": "native"},
                "auto_class_weights": {"type": "enum", "options": [None, "Balanced", "SqrtBalanced"], "default": None},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False, "fixed": True}},
        }
        families_catalog["ExtraTrees"] = {
            "parameters": {
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 1000, "default": 180},
                "max_leaf_nodes": {"type": "integer", "minimum": 2, "maximum": 2048, "default": 128},
                "min_samples_leaf": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
                "max_features": {"type": "number|string|null", "options": ["sqrt", "log2", None], "exclusive_minimum": 0, "maximum": 1, "default": 0.8},
                "max_depth": {"type": "integer|null", "minimum": 2, "maximum": 64, "default": None},
                "class_weight": {"type": "enum", "options": [None, "balanced", "balanced_subsample"], "default": None},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        }
        families_catalog["Linear"] = {
            "parameters": {
                "C": {"type": "number", "exclusive_minimum": 0, "maximum": 100, "default": 0.5},
                "max_iter": {"type": "integer", "minimum": 100, "maximum": 3000, "default": 300},
                "solver": {"type": "enum", "options": ["saga"], "default": "saga", "fixed": True},
                "class_weight": {"type": "enum", "options": [None, "balanced"], "default": None},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        }
    else:
        families_catalog["CatBoost"] = {
            "parameters": {
                "iterations": {"type": "integer", "minimum": 10, "maximum": 1000, "default": 150},
                "depth": {"type": "integer", "minimum": 3, "maximum": 10, "default": "native"},
                "learning_rate": {"type": "number", "exclusive_minimum": 0, "maximum": 1, "default": "native automatic"},
                "l2_leaf_reg": {"type": "number", "minimum": 0, "maximum": 100, "default": "native"},
                "border_count": {"type": "integer", "minimum": 16, "maximum": 255, "default": "native"},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False, "fixed": True}},
        }
        families_catalog["ExtraTrees"] = {
            "parameters": {
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 1000, "default": 180},
                "max_leaf_nodes": {"type": "integer", "minimum": 2, "maximum": 2048, "default": 128},
                "min_samples_leaf": {"type": "integer", "minimum": 1, "maximum": 100, "default": 3},
                "max_features": {"type": "number|string|null", "options": ["sqrt", "log2", None], "exclusive_minimum": 0, "maximum": 1, "default": 0.8},
                "max_depth": {"type": "integer|null", "minimum": 2, "maximum": 64, "default": None},
            },
            "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}},
        }
        families_catalog["Linear"] = {"parameters": {}, "preprocessing": {"add_missing_indicators": {"type": "boolean", "default": False}}}

    return {
        "task_type": task_type,
        "families": families_catalog,
        "host_controlled": ["n_jobs", "thread_count", "random_state", "random_seed", "seed"],
        "unsupported": {
            "class_weights": "Use CatBoost auto_class_weights or sklearn class_weight.",
            "sample_weight": "Per-row fit weights are outside the recipe API.",
            "early_stopping_rounds": "Requires an eval_set; recipe candidates do not receive one.",
        },
    }
