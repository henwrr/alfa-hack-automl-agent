from types import SimpleNamespace

from automl_agent import recipes
from automl_agent import search


def _result(recipe, metric=0.5, trial_id="trial_0001"):
    return SimpleNamespace(
        recipe=recipe,
        metric=metric,
        status="ok",
        trial_id=trial_id,
    )


def test_initial_recipes_start_fast_models_and_always_include_linear_regression():
    available = {
        "lightgbm": "4.6.0",
        "catboost": "1.2.8",
        "xgboost": "2.1.4",
    }

    candidates = recipes.initial_research_recipes("regression", available)
    families = [candidate["family"] for candidate in candidates]

    assert families[:5] == ["LightGBM", "Linear", "ExtraTrees", "XGBoost", "CatBoost"]
    linear = next(candidate for candidate in candidates if candidate["family"] == "Linear")
    assert linear["parameters"]["alpha"] == 1.0
    assert "Ridge" in recipes.recipe_code(linear, "regression")


def test_regression_ridge_alpha_has_a_small_deterministic_local_search_space():
    base = recipes.default_recipe("Linear", "regression")
    result = _result(base)

    proposal = search.next_search_proposal([result], "regression", False, seed=42, step=0)

    assert proposal is not None
    alpha = proposal["parameters"].get("alpha")
    assert alpha in {0.1, 10.0}
    assert proposal["base_trial_id"] == "trial_0001"
    assert search.next_search_proposal([result], "regression", False, seed=42, step=0) == proposal


def test_metric_appropriate_sqrt_balanced_is_proposed_before_feature_chain():
    base = recipes.default_recipe("LightGBM", "classification")
    results = [_result(base)]

    proposal = search.next_feature_or_weight_proposal(
        results,
        "classification",
        "macro_f1",
        ["amount", "kind"],
        ["kind"],
        seed=42,
        step=0,
    )

    assert proposal["base_trial_id"] == "trial_0001"
    assert proposal["fit_options"]["sample_weight"] == "sqrt_balanced"
    assert search.next_feature_or_weight_proposal(
        results, "classification", "macro_f1", ["amount", "kind"], ["kind"], 42, 0
    ) == proposal


def test_feature_chain_uses_only_declared_fit_schema_and_is_deduplicated():
    base = recipes.default_recipe("LightGBM", "classification")
    weighted = {
        **base,
        "fit_options": {
            "early_stopping_rounds": 0,
            "inner_validation_fraction": 0.15,
            "sample_weight": "sqrt_balanced",
        },
    }
    results = [_result(base, metric=0.4), _result(weighted, metric=0.52, trial_id="trial_0002")]

    proposal = search.next_feature_or_weight_proposal(
        results,
        "classification",
        "macro_f1",
        ["amount", "kind"],
        ["kind"],
        seed=42,
        step=1,
    )

    assert proposal["base_trial_id"] == "trial_0002"
    assert proposal["preprocessing"]["feature_ops"] == [
        {"op": "freq_encode", "cols": ["kind"]},
        {"op": "missing_flags", "cols": ["amount"]},
    ]
    with_feature_ops = {
        **weighted,
        "preprocessing": proposal["preprocessing"],
    }
    second_base = search.next_feature_or_weight_proposal(
        [*results, _result(with_feature_ops, trial_id="trial_0003")],
        "classification",
        "macro_f1",
        ["amount", "kind"],
        ["kind"],
        seed=42,
        step=2,
    )
    assert second_base["base_trial_id"] == "trial_0001"
    assert second_base["preprocessing"] == proposal["preprocessing"]
    assert search.next_feature_or_weight_proposal(
        [
            *results,
            _result(with_feature_ops, trial_id="trial_0003"),
            _result({**base, "preprocessing": second_base["preprocessing"]}, trial_id="trial_0004"),
        ],
        "classification",
        "macro_f1",
        ["amount", "kind"],
        ["kind"],
        seed=42,
        step=3,
    ) is None


def test_balanced_weight_is_not_proposed_for_non_f1_metrics_or_regression():
    base = recipes.default_recipe("LightGBM", "classification")

    non_f1_proposal = search.next_feature_or_weight_proposal(
        [_result(base)], "classification", "log_loss", ["amount"], [], 42, 0
    )
    assert non_f1_proposal is None or "fit_options" not in non_f1_proposal
    regression_proposal = search.next_feature_or_weight_proposal(
        [_result(recipes.default_recipe("Linear", "regression"))],
        "regression",
        "rmse",
        ["amount"],
        [],
        42,
        0,
    )
    assert regression_proposal is None or "fit_options" not in regression_proposal

