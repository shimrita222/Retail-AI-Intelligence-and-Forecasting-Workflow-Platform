"""Closed, extensible registry of candidate regression models.

This module is the single source of truth for MODEL ELIGIBILITY/CAPABILITY
METADATA: which model_ids exist, what family/problem-type they belong to,
whether they are currently enabled, whether they are core or optional, and
whether their runtime dependency is actually importable in this environment.

It intentionally does NOT decide candidate SELECTION, FALLBACK ORDER, or
FINAL PERFORMANCE SELECTION -- those are separate concerns handled by later
Phase 3 slices, per the frozen Phase 3 implementation contract. It also does
NOT yet implement a combined compute_eligibility() (registered + enabled +
dependency_available + problem_type_compatible => eligible); that belongs to
a later slice. What this module guarantees is that the metadata needed to
compute eligibility later already exists on ModelEntry, so that later
computation will not require changing this dataclass again.

Five distinct concepts, deliberately kept separate and never conflated:
  - registered:              the model_id has a ModelEntry in MODEL_REGISTRY.
  - enabled:                 ModelEntry.enabled is True (a policy switch;
                              currently no runtime/UI toggle exists, so it is
                              just a static default in Slice 1).
  - dependency_available:    is_available(entry) is True (core: always True;
                              optional: importlib.util.find_spec succeeds).
  - problem_type_compatible: the target problem type is in entry.problem_types.
  - eligible:                the (future) combination of all of the above.

Core vs. optional is machine-readable via `ModelEntry.is_core` (not inferred
from `required_package` alone, and never from display_name text): core
models ship with the project's existing required dependency (scikit-learn),
so `is_available()` is always True for them. Optional models require a
third-party package (xgboost / lightgbm / catboost) that may not be
installed; their factories perform the import lazily, inside the factory
function, so importing this module -- or running the rest of the test suite
-- never fails just because an optional package is absent.

`family` is a small, controlled, language-neutral vocabulary
(FAMILY_LINEAR / FAMILY_BAGGING_ENSEMBLE / FAMILY_BOOSTING_ENSEMBLE) so a
later candidate-selection policy can reason about model shape without
string-matching display names or encoding any business preference.

`problem_types` is a tuple so a model can (in the future) support more than
one problem type without a schema change; currently every active entry is
regression-only because Weekly_Sales is a continuous target. No
classification model is registered.

`runtime_cost_tier` is declared, static, operational/resource metadata
(engineering cost to train/serve) -- never a measurement, never a proxy for
model quality/accuracy/business value, and never used by this module to pick
a winner. Per the frozen contract amendment, it may inform pre-training
Fallback Order in a later slice, but must never participate in final winner
tie-breaking.

Extending the registry: adding a new candidate model requires adding exactly
one new `ModelEntry` to `MODEL_REGISTRY` below -- no other file in this
project needs to change for the model to become registry-visible and
availability-checkable. (This module proves only the registry/factory/
availability boundary accepts a new compatible model definition without
special-casing -- see `test_registry_is_extensible_without_modifying_this_module_or_others`.
It does not, by itself, prove the full future extension requirement across
training/evaluation/Flow; that full acceptance test is required later, once
the generalized training/selection path exists.)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

# Reason codes are structured (code + params), never bare English strings, so
# that a future presentation layer can localize (Hebrew/English) without
# touching this decision logic.
DEPENDENCY_UNAVAILABLE = "dependency_unavailable"

# Controlled, language-neutral family vocabulary. Deliberately small and
# shape-based (never a business-preference ranking).
FAMILY_LINEAR = "linear"
FAMILY_BAGGING_ENSEMBLE = "bagging_ensemble"
FAMILY_BOOSTING_ENSEMBLE = "boosting_ensemble"

# Only regression is implemented; Weekly_Sales is a continuous target.
PROBLEM_TYPE_REGRESSION = "regression"


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    display_name: str
    family: str
    problem_types: tuple[str, ...]
    enabled: bool
    is_core: bool  # explicit, machine-readable core/optional classification
    required_package: str | None  # None => core model, no optional dependency
    runtime_cost_tier: str  # "low" | "medium" | "high" -- declared, operational metadata only
    factory: Callable[[], Any]

    def __post_init__(self) -> None:
        # is_core and required_package must never disagree, so downstream
        # code can trust either signal without cross-checking the other.
        if self.is_core != (self.required_package is None):
            raise ValueError(
                f"{self.model_id}: is_core={self.is_core} is inconsistent with "
                f"required_package={self.required_package!r}"
            )


def _ridge_factory() -> Any:
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0)


def _random_forest_factory() -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=100, random_state=42)


def _gradient_boosting_factory() -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(n_estimators=100, random_state=42)


def _xgboost_factory() -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(n_estimators=100, random_state=42)


def _lightgbm_factory() -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=100, random_state=42)


def _catboost_factory() -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "Ridge": ModelEntry(
        model_id="Ridge",
        display_name="Ridge Regression",
        family=FAMILY_LINEAR,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=True,
        required_package=None,
        runtime_cost_tier="low",
        factory=_ridge_factory,
    ),
    "RandomForestRegressor": ModelEntry(
        model_id="RandomForestRegressor",
        display_name="Random Forest Regressor",
        family=FAMILY_BAGGING_ENSEMBLE,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=True,
        required_package=None,
        runtime_cost_tier="medium",
        factory=_random_forest_factory,
    ),
    "GradientBoostingRegressor": ModelEntry(
        model_id="GradientBoostingRegressor",
        display_name="Gradient Boosting Regressor",
        family=FAMILY_BOOSTING_ENSEMBLE,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=True,
        required_package=None,
        runtime_cost_tier="medium",
        factory=_gradient_boosting_factory,
    ),
    "XGBoostRegressor": ModelEntry(
        model_id="XGBoostRegressor",
        display_name="XGBoost Regressor",
        family=FAMILY_BOOSTING_ENSEMBLE,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=False,
        required_package="xgboost",
        runtime_cost_tier="high",
        factory=_xgboost_factory,
    ),
    "LightGBMRegressor": ModelEntry(
        model_id="LightGBMRegressor",
        display_name="LightGBM Regressor",
        family=FAMILY_BOOSTING_ENSEMBLE,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=False,
        required_package="lightgbm",
        runtime_cost_tier="high",
        factory=_lightgbm_factory,
    ),
    "CatBoostRegressor": ModelEntry(
        model_id="CatBoostRegressor",
        display_name="CatBoost Regressor",
        family=FAMILY_BOOSTING_ENSEMBLE,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=False,
        required_package="catboost",
        runtime_cost_tier="high",
        factory=_catboost_factory,
    ),
}


@lru_cache(maxsize=None)
def _package_available(package_name: str) -> bool:
    """Cached, side-effect-free import-spec check. Never installs anything."""
    return importlib.util.find_spec(package_name) is not None


def is_available(entry: ModelEntry) -> bool:
    """True if entry's runtime dependency is importable (core entries: always True).

    This is `dependency_available` only -- it does not consider `enabled` or
    `problem_types`. Combining those into a single eligibility decision is
    deferred to a later slice's compute_eligibility().
    """
    if entry.required_package is None:
        return True
    return _package_available(entry.required_package)


def get_registry() -> dict[str, ModelEntry]:
    return dict(MODEL_REGISTRY)


def list_available_entries() -> list[ModelEntry]:
    """All registry entries whose dependency is currently importable.

    Availability only -- does not filter on `enabled` or `problem_types`.
    """
    return [entry for entry in MODEL_REGISTRY.values() if is_available(entry)]


def unavailable_reason(entry: ModelEntry) -> dict[str, Any] | None:
    """Structured reason_code/reason_params for why entry is unavailable, or None if available."""
    if is_available(entry):
        return None
    return {
        "reason_code": DEPENDENCY_UNAVAILABLE,
        "reason_params": {
            "model_id": entry.model_id,
            "required_package": entry.required_package,
        },
    }
