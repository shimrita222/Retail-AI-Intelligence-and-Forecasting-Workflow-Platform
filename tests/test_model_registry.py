import importlib.util

import pytest

from src.services.model_registry import (
    DEPENDENCY_UNAVAILABLE,
    FAMILY_BAGGING_ENSEMBLE,
    FAMILY_BOOSTING_ENSEMBLE,
    FAMILY_LINEAR,
    MODEL_REGISTRY,
    PROBLEM_TYPE_REGRESSION,
    ModelEntry,
    get_registry,
    is_available,
    list_available_entries,
    unavailable_reason,
    _package_available,
)

CORE_MODEL_IDS = {"Ridge", "RandomForestRegressor", "GradientBoostingRegressor"}
OPTIONAL_MODEL_IDS = {"XGBoostRegressor", "LightGBMRegressor", "CatBoostRegressor"}
VALID_FAMILIES = {FAMILY_LINEAR, FAMILY_BAGGING_ENSEMBLE, FAMILY_BOOSTING_ENSEMBLE}
VALID_COST_TIERS = {"low", "medium", "high"}

EXPECTED_FAMILY = {
    "Ridge": FAMILY_LINEAR,
    "RandomForestRegressor": FAMILY_BAGGING_ENSEMBLE,
    "GradientBoostingRegressor": FAMILY_BOOSTING_ENSEMBLE,
    "XGBoostRegressor": FAMILY_BOOSTING_ENSEMBLE,
    "LightGBMRegressor": FAMILY_BOOSTING_ENSEMBLE,
    "CatBoostRegressor": FAMILY_BOOSTING_ENSEMBLE,
}

EXPECTED_PACKAGE = {
    "XGBoostRegressor": "xgboost",
    "LightGBMRegressor": "lightgbm",
    "CatBoostRegressor": "catboost",
}


def test_registry_contains_exactly_six_entries():
    assert set(MODEL_REGISTRY.keys()) == CORE_MODEL_IDS | OPTIONAL_MODEL_IDS


def test_every_entry_has_a_valid_nonempty_model_id():
    for key, entry in MODEL_REGISTRY.items():
        assert isinstance(entry.model_id, str) and entry.model_id
        assert entry.model_id == key


def test_model_ids_are_unique():
    ids = [entry.model_id for entry in MODEL_REGISTRY.values()]
    assert len(ids) == len(set(ids))


def test_every_entry_has_a_valid_family():
    for entry in MODEL_REGISTRY.values():
        assert entry.family in VALID_FAMILIES
        assert entry.family == EXPECTED_FAMILY[entry.model_id]


def test_family_is_not_derived_from_display_name_text():
    # Family values must be a controlled vocabulary, not substrings of
    # display_name (e.g. must not just be "boosting" scraped from the name).
    for entry in MODEL_REGISTRY.values():
        assert entry.family not in entry.display_name


def test_regression_is_present_in_problem_types_for_all_current_entries():
    for entry in MODEL_REGISTRY.values():
        assert PROBLEM_TYPE_REGRESSION in entry.problem_types
        assert isinstance(entry.problem_types, tuple)


def test_enabled_is_explicit_boolean_for_all_current_entries():
    for entry in MODEL_REGISTRY.values():
        assert entry.enabled is True


def test_core_optional_classification_is_explicit_and_correct():
    for model_id in CORE_MODEL_IDS:
        assert MODEL_REGISTRY[model_id].is_core is True
    for model_id in OPTIONAL_MODEL_IDS:
        assert MODEL_REGISTRY[model_id].is_core is False


def test_core_entries_have_no_required_package():
    for model_id in CORE_MODEL_IDS:
        assert MODEL_REGISTRY[model_id].required_package is None


def test_optional_entries_declare_their_required_package():
    for model_id, package in EXPECTED_PACKAGE.items():
        assert MODEL_REGISTRY[model_id].required_package == package


def test_is_core_and_required_package_are_never_inconsistent():
    with pytest.raises(ValueError):
        ModelEntry(
            model_id="Broken",
            display_name="Broken",
            family=FAMILY_LINEAR,
            problem_types=(PROBLEM_TYPE_REGRESSION,),
            enabled=True,
            is_core=True,
            required_package="some_package",  # contradicts is_core=True
            runtime_cost_tier="low",
            factory=lambda: object(),
        )


def test_runtime_cost_tier_is_valid_operational_metadata():
    for entry in MODEL_REGISTRY.values():
        assert entry.runtime_cost_tier in VALID_COST_TIERS


def test_core_model_factories_construct_working_estimators():
    ridge = MODEL_REGISTRY["Ridge"].factory()
    assert ridge.alpha == 1.0

    rf = MODEL_REGISTRY["RandomForestRegressor"].factory()
    assert rf.n_estimators == 100
    assert rf.random_state == 42

    gbr = MODEL_REGISTRY["GradientBoostingRegressor"].factory()
    assert gbr.n_estimators == 100
    assert gbr.random_state == 42


def test_importing_registry_never_fails_regardless_of_optional_packages():
    # Import already happened at module load; if an optional package were
    # imported eagerly, a missing package would have raised on collection.
    assert "XGBoostRegressor" in MODEL_REGISTRY


def test_availability_reflects_find_spec_result(monkeypatch):
    _package_available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    entry = MODEL_REGISTRY["XGBoostRegressor"]
    assert is_available(entry) is False
    _package_available.cache_clear()


def test_availability_true_when_find_spec_finds_the_package(monkeypatch):
    _package_available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    entry = MODEL_REGISTRY["LightGBMRegressor"]
    assert is_available(entry) is True
    _package_available.cache_clear()


def test_unavailable_reason_is_none_for_available_entry():
    entry = MODEL_REGISTRY["Ridge"]
    assert unavailable_reason(entry) is None


def test_unavailable_reason_is_structured_not_a_bare_string(monkeypatch):
    _package_available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    entry = MODEL_REGISTRY["CatBoostRegressor"]
    reason = unavailable_reason(entry)
    assert reason == {
        "reason_code": DEPENDENCY_UNAVAILABLE,
        "reason_params": {"model_id": "CatBoostRegressor", "required_package": "catboost"},
    }
    _package_available.cache_clear()


def test_list_available_entries_excludes_unavailable_optional_packages(monkeypatch):
    _package_available.cache_clear()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    available_ids = {e.model_id for e in list_available_entries()}
    assert available_ids == CORE_MODEL_IDS
    _package_available.cache_clear()


def test_get_registry_returns_a_copy_not_the_live_dict():
    snapshot = get_registry()
    snapshot["Injected"] = None
    assert "Injected" not in MODEL_REGISTRY


def test_registry_lookup_is_deterministic():
    first = get_registry()
    second = get_registry()
    assert first == second
    for model_id in MODEL_REGISTRY:
        assert MODEL_REGISTRY[model_id] is MODEL_REGISTRY[model_id]


def test_registry_is_extensible_without_modifying_this_module_or_others():
    """Registry/factory/availability extension boundary only.

    This proves a new ModelEntry can be constructed and driven through
    is_available()/unavailable_reason() using only this module's public
    dataclass/constants, with no model_id-specific branching required. It
    does NOT prove the complete future-model extension requirement across
    the trainer/evaluation/Flow path -- that full acceptance test is
    required later, once the generalized training/selection path exists.
    """

    def _fake_factory():
        return object()

    fake_entry = ModelEntry(
        model_id="FakeRegressor",
        display_name="Fake Regressor",
        family=FAMILY_BOOSTING_ENSEMBLE,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=False,
        required_package="totally_nonexistent_package_xyz",
        runtime_cost_tier="high",
        factory=_fake_factory,
    )

    assert is_available(fake_entry) is False
    reason = unavailable_reason(fake_entry)
    assert reason["reason_code"] == DEPENDENCY_UNAVAILABLE
    assert reason["reason_params"]["model_id"] == "FakeRegressor"

    core_fake_entry = ModelEntry(
        model_id="FakeCoreRegressor",
        display_name="Fake Core Regressor",
        family=FAMILY_LINEAR,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=True,
        required_package=None,
        runtime_cost_tier="low",
        factory=_fake_factory,
    )
    assert is_available(core_fake_entry) is True
    assert unavailable_reason(core_fake_entry) is None
