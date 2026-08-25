import importlib.util

import pytest

from src.services.candidate_selection import (
    INSUFFICIENT_ELIGIBLE_CANDIDATES,
    MANDATORY_ANCHOR_MODEL_IDS,
    MODEL_DISABLED,
    PROBLEM_TYPE_INCOMPATIBLE,
    CANDIDATE_COUNT_ROW_THRESHOLD,
    compute_eligibility,
    evaluate_pre_training_gate,
    select_candidates,
    target_candidate_count,
)
from src.services.model_registry import (
    DEPENDENCY_UNAVAILABLE,
    FAMILY_BAGGING_ENSEMBLE,
    FAMILY_BOOSTING_ENSEMBLE,
    FAMILY_LINEAR,
    MODEL_REGISTRY,
    PROBLEM_TYPE_REGRESSION,
    ModelEntry,
    _package_available,
)


def _entry(
    model_id,
    family=FAMILY_BOOSTING_ENSEMBLE,
    problem_types=(PROBLEM_TYPE_REGRESSION,),
    enabled=True,
    is_core=True,
    required_package=None,
    runtime_cost_tier="high",
):
    return ModelEntry(
        model_id=model_id,
        display_name=model_id,
        family=family,
        problem_types=problem_types,
        enabled=enabled,
        is_core=is_core,
        required_package=required_package,
        runtime_cost_tier=runtime_cost_tier,
        factory=lambda: object(),
    )


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_all_six_registered_models_produce_eligibility_results():
    results = compute_eligibility()
    assert set(results.keys()) == set(MODEL_REGISTRY.keys())
    for model_id, result in results.items():
        assert result.model_id == model_id
        assert result.registered is True


def test_core_enabled_regression_models_are_eligible_in_normal_environment():
    results = compute_eligibility(problem_type=PROBLEM_TYPE_REGRESSION)
    for model_id in ("Ridge", "RandomForestRegressor", "GradientBoostingRegressor"):
        assert results[model_id].eligible is True
        assert results[model_id].reason_code is None


def test_missing_optional_dependency_produces_dependency_unavailable(monkeypatch):
    _package_available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    results = compute_eligibility(problem_type=PROBLEM_TYPE_REGRESSION)
    result = results["XGBoostRegressor"]
    assert result.dependency_available is False
    assert result.eligible is False
    assert result.reason_code == DEPENDENCY_UNAVAILABLE
    assert result.reason_params == {"model_id": "XGBoostRegressor", "required_package": "xgboost"}
    _package_available.cache_clear()


def test_disabled_model_produces_model_disabled_reason():
    synthetic_registry = {"DisabledModel": _entry("DisabledModel", enabled=False, required_package=None, is_core=True)}
    results = compute_eligibility(registry=synthetic_registry)
    result = results["DisabledModel"]
    assert result.enabled is False
    assert result.eligible is False
    assert result.reason_code == MODEL_DISABLED
    assert result.reason_params == {"model_id": "DisabledModel"}


def test_problem_type_incompatibility_produces_structured_reason():
    synthetic_registry = {
        "ClassifierOnlyModel": _entry(
            "ClassifierOnlyModel",
            problem_types=("classification",),
            required_package=None,
            is_core=True,
        )
    }
    results = compute_eligibility(problem_type=PROBLEM_TYPE_REGRESSION, registry=synthetic_registry)
    result = results["ClassifierOnlyModel"]
    assert result.problem_type_compatible is False
    assert result.eligible is False
    assert result.reason_code == PROBLEM_TYPE_INCOMPATIBLE
    assert result.reason_params == {
        "model_id": "ClassifierOnlyModel",
        "requested_problem_type": PROBLEM_TYPE_REGRESSION,
        "supported_problem_types": ["classification"],
    }


def test_eligibility_is_deterministic():
    first = compute_eligibility()
    second = compute_eligibility()
    assert first == second


def test_eligibility_does_not_mutate_model_registry():
    snapshot = dict(MODEL_REGISTRY)
    compute_eligibility()
    assert MODEL_REGISTRY == snapshot


# ---------------------------------------------------------------------------
# Candidate-count policy
# ---------------------------------------------------------------------------


def test_candidate_count_below_threshold_targets_two():
    assert target_candidate_count(CANDIDATE_COUNT_ROW_THRESHOLD - 1) == 2


def test_candidate_count_at_threshold_targets_three():
    assert target_candidate_count(CANDIDATE_COUNT_ROW_THRESHOLD) == 3


def test_candidate_count_above_threshold_targets_three():
    assert target_candidate_count(CANDIDATE_COUNT_ROW_THRESHOLD + 1) == 3


def test_candidate_count_rejects_negative_row_count():
    with pytest.raises(ValueError):
        target_candidate_count(-1)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def test_two_candidate_path_selects_mandatory_anchors():
    results = compute_eligibility()
    selected = select_candidates(results, target_count=2)
    assert selected == list(MANDATORY_ANCHOR_MODEL_IDS)


def test_three_candidate_path_includes_anchors_plus_one_boosting_candidate():
    results = compute_eligibility()
    selected = select_candidates(results, target_count=3)
    assert selected[:2] == list(MANDATORY_ANCHOR_MODEL_IDS)
    assert len(selected) == 3
    assert selected[2] in {"GradientBoostingRegressor", "XGBoostRegressor", "LightGBMRegressor", "CatBoostRegressor"}


def test_optional_packages_absent_falls_back_to_gradient_boosting(monkeypatch):
    _package_available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    results = compute_eligibility()
    selected = select_candidates(results, target_count=3)
    assert selected == ["Ridge", "RandomForestRegressor", "GradientBoostingRegressor"]
    _package_available.cache_clear()


def test_fallback_order_prefers_lower_runtime_cost_tier():
    synthetic_registry = {
        "Ridge": MODEL_REGISTRY["Ridge"],
        "RandomForestRegressor": MODEL_REGISTRY["RandomForestRegressor"],
        "ZModelLowCost": _entry("ZModelLowCost", runtime_cost_tier="low"),
        "AModelMediumCost": _entry("AModelMediumCost", runtime_cost_tier="medium"),
    }
    results = compute_eligibility(registry=synthetic_registry)
    selected = select_candidates(results, target_count=3, registry=synthetic_registry)
    assert selected == ["Ridge", "RandomForestRegressor", "ZModelLowCost"]


def test_fallback_order_uses_model_id_alphabetical_on_tier_tie():
    synthetic_registry = {
        "Ridge": MODEL_REGISTRY["Ridge"],
        "RandomForestRegressor": MODEL_REGISTRY["RandomForestRegressor"],
        "Zeta": _entry("Zeta", runtime_cost_tier="high"),
        "Alpha": _entry("Alpha", runtime_cost_tier="high"),
    }
    results = compute_eligibility(registry=synthetic_registry)
    selected = select_candidates(results, target_count=3, registry=synthetic_registry)
    assert selected == ["Ridge", "RandomForestRegressor", "Alpha"]


def test_ineligible_models_cannot_enter_selected_candidates():
    synthetic_registry = {
        "Ridge": MODEL_REGISTRY["Ridge"],
        "RandomForestRegressor": MODEL_REGISTRY["RandomForestRegressor"],
        "DisabledBoosting": _entry("DisabledBoosting", enabled=False, runtime_cost_tier="low"),
    }
    results = compute_eligibility(registry=synthetic_registry)
    selected = select_candidates(results, target_count=3, registry=synthetic_registry)
    assert "DisabledBoosting" not in selected
    assert selected == ["Ridge", "RandomForestRegressor"]


def test_candidate_pool_contains_no_duplicate_model_ids():
    results = compute_eligibility()
    selected = select_candidates(results, target_count=3)
    assert len(selected) == len(set(selected))


def test_selection_does_not_mutate_registry_state():
    snapshot = dict(MODEL_REGISTRY)
    results = compute_eligibility()
    select_candidates(results, target_count=3)
    assert MODEL_REGISTRY == snapshot


# ---------------------------------------------------------------------------
# Pre-training gate
# ---------------------------------------------------------------------------


def test_gate_passes_when_enough_eligible_candidates_for_target_two():
    result = evaluate_pre_training_gate(train_row_count=1000)
    assert result.status == "PASS"
    assert result.target_candidate_count == 2
    assert result.selected_model_ids == tuple(MANDATORY_ANCHOR_MODEL_IDS)
    assert result.reason_code is None


def test_gate_passes_when_enough_eligible_candidates_for_target_three():
    result = evaluate_pre_training_gate(train_row_count=CANDIDATE_COUNT_ROW_THRESHOLD)
    assert result.status == "PASS"
    assert result.target_candidate_count == 3
    assert len(result.selected_model_ids) == 3
    assert result.reason_code is None


def test_gate_fails_when_fewer_eligible_candidates_than_required():
    synthetic_registry = {"Ridge": MODEL_REGISTRY["Ridge"]}
    result = evaluate_pre_training_gate(
        train_row_count=CANDIDATE_COUNT_ROW_THRESHOLD, registry=synthetic_registry
    )
    assert result.status == "FAIL"
    assert result.target_candidate_count == 3
    assert result.reason_code == INSUFFICIENT_ELIGIBLE_CANDIDATES


def test_gate_failure_reason_params_are_auditable():
    synthetic_registry = {"Ridge": MODEL_REGISTRY["Ridge"]}
    result = evaluate_pre_training_gate(
        train_row_count=CANDIDATE_COUNT_ROW_THRESHOLD, registry=synthetic_registry
    )
    assert result.reason_params == {
        "required_count": 3,
        "available_count": 1,
        "available_model_ids": ["Ridge"],
    }


def test_gate_never_produces_post_training_reason_code():
    result = evaluate_pre_training_gate(train_row_count=1000)
    assert result.reason_code != "insufficient_successful_candidates"
    synthetic_registry = {"Ridge": MODEL_REGISTRY["Ridge"]}
    fail_result = evaluate_pre_training_gate(
        train_row_count=CANDIDATE_COUNT_ROW_THRESHOLD, registry=synthetic_registry
    )
    assert fail_result.reason_code != "insufficient_successful_candidates"
