import math

import numpy as np
import pandas as pd
import pytest

from src.services.ml_trainer import (
    CANDIDATE_TRAINING_FAILED,
    NON_FINITE_METRIC,
    UNKNOWN_MODEL_ID,
    CandidateOutcome,
    train_and_select_model,
    train_candidates,
)
from src.services.model_registry import (
    FAMILY_BOOSTING_ENSEMBLE,
    MODEL_REGISTRY,
    ModelEntry,
    PROBLEM_TYPE_REGRESSION,
)


def _multi_series_df(n_weeks=30):
    dates = pd.date_range("2010-01-01", periods=n_weeks, freq="7D")
    frames = []
    rng = np.random.default_rng(42)
    for store in (1, 2):
        for dept in (1, 2):
            base = 100.0 * store + 10.0 * dept
            sales = base + rng.normal(0, 5, n_weeks) + np.arange(n_weeks) * 1.5
            frames.append(
                pd.DataFrame(
                    {
                        "Store": store,
                        "Dept": dept,
                        "Date": dates,
                        "Weekly_Sales": sales,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


FEATURE_COLUMNS = ["Store", "Dept"]


def _entry(model_id, factory, family=FAMILY_BOOSTING_ENSEMBLE, required_package=None, is_core=True):
    return ModelEntry(
        model_id=model_id,
        display_name=model_id,
        family=family,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=is_core,
        required_package=required_package,
        runtime_cost_tier="low",
        factory=factory,
    )


class _AlwaysFailsEstimator:
    def fit(self, X, y):
        raise ValueError("synthetic construction-time-ok-but-fit-fails")


class _PredictFailsEstimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        raise ValueError("synthetic predict failure")


class _ConstantEstimator:
    def __init__(self, value):
        self._value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self._value)


# ---------------------------------------------------------------------------
# Registry factory resolution
# ---------------------------------------------------------------------------


def test_trainer_resolves_models_through_model_registry():
    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["Ridge", "RandomForestRegressor"])
    assert run["outcomes"]["Ridge"].status == "success"
    assert run["outcomes"]["RandomForestRegressor"].status == "success"


def test_no_independent_candidate_factories_constant_remains():
    import src.services.ml_trainer as ml_trainer

    assert not hasattr(ml_trainer, "CANDIDATE_FACTORIES")


def test_only_requested_selected_model_ids_are_instantiated():
    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["Ridge"])
    assert set(run["outcomes"].keys()) == {"Ridge"}


def test_unknown_model_id_is_handled_deterministically():
    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["TotallyUnregisteredModel"])
    outcome = run["outcomes"]["TotallyUnregisteredModel"]
    assert outcome.status == "failed"
    assert outcome.reason_code == UNKNOWN_MODEL_ID
    assert outcome.reason_params == {"model_id": "TotallyUnregisteredModel"}


def test_optional_dependency_absence_does_not_break_core_execution():
    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["Ridge", "RandomForestRegressor", "GradientBoostingRegressor"])
    for model_id in ("Ridge", "RandomForestRegressor", "GradientBoostingRegressor"):
        assert run["outcomes"][model_id].status == "success"


# ---------------------------------------------------------------------------
# Dynamic training / failure isolation
# ---------------------------------------------------------------------------


def test_successful_two_model_execution():
    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["Ridge", "RandomForestRegressor"])
    assert all(o.status == "success" for o in run["outcomes"].values())


def test_successful_three_model_sklearn_only_execution():
    df = _multi_series_df()
    run = train_candidates(
        df, FEATURE_COLUMNS, ["Ridge", "RandomForestRegressor", "GradientBoostingRegressor"]
    )
    assert all(o.status == "success" for o in run["outcomes"].values())
    assert len(run["outcomes"]) == 3


def test_candidate_construction_failure_does_not_block_others():
    def _broken_factory():
        raise RuntimeError("synthetic construction failure")

    registry = dict(MODEL_REGISTRY)
    registry["BrokenConstruct"] = _entry("BrokenConstruct", _broken_factory)

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["BrokenConstruct", "Ridge"], registry=registry)

    broken = run["outcomes"]["BrokenConstruct"]
    assert broken.status == "failed"
    assert broken.reason_code == CANDIDATE_TRAINING_FAILED
    assert broken.reason_params["stage"] == "construct"
    assert broken.reason_params["exception_type"] == "RuntimeError"
    assert run["outcomes"]["Ridge"].status == "success"


def test_candidate_fit_failure_does_not_block_others():
    registry = dict(MODEL_REGISTRY)
    registry["BrokenFit"] = _entry("BrokenFit", lambda: _AlwaysFailsEstimator())

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["BrokenFit", "Ridge"], registry=registry)

    broken = run["outcomes"]["BrokenFit"]
    assert broken.status == "failed"
    assert broken.reason_code == CANDIDATE_TRAINING_FAILED
    assert broken.reason_params["stage"] == "fit"
    assert run["outcomes"]["Ridge"].status == "success"


def test_candidate_predict_failure_does_not_block_others():
    registry = dict(MODEL_REGISTRY)
    registry["BrokenPredict"] = _entry("BrokenPredict", lambda: _PredictFailsEstimator())

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["BrokenPredict", "Ridge"], registry=registry)

    broken = run["outcomes"]["BrokenPredict"]
    assert broken.status == "failed"
    assert broken.reason_code == CANDIDATE_TRAINING_FAILED
    assert broken.reason_params["stage"] == "predict"
    assert run["outcomes"]["Ridge"].status == "success"


def test_failed_candidate_is_never_represented_as_successful():
    registry = dict(MODEL_REGISTRY)
    registry["BrokenFit"] = _entry("BrokenFit", lambda: _AlwaysFailsEstimator())

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["BrokenFit"], registry=registry)
    outcome = run["outcomes"]["BrokenFit"]
    assert outcome.status != "success"
    assert outcome.metrics is None
    assert outcome.fitted_model is None


def test_trainer_never_substitutes_a_failed_candidate():
    registry = dict(MODEL_REGISTRY)
    registry["BrokenFit"] = _entry("BrokenFit", lambda: _AlwaysFailsEstimator())

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["BrokenFit"], registry=registry)
    # Only the exact requested id appears -- nothing else was substituted in.
    assert set(run["outcomes"].keys()) == {"BrokenFit"}


# ---------------------------------------------------------------------------
# Non-finite metrics
# ---------------------------------------------------------------------------


_MARKER = 999999.0


def _selective_fake_evaluate(original, fake_metrics):
    """Return `fake_metrics` only for predictions from our marked constant
    estimator; delegate every other candidate (e.g. Ridge) to the real
    `_evaluate`, so patching one candidate's metrics never affects siblings.
    """

    def _wrapped(y_true, y_pred):
        if np.all(np.asarray(y_pred) == _MARKER):
            return dict(fake_metrics)
        return original(y_true, y_pred)

    return _wrapped


def test_nan_metric_makes_candidate_unusable(monkeypatch):
    import src.services.ml_trainer as ml_trainer

    original = ml_trainer._evaluate
    monkeypatch.setattr(
        ml_trainer, "_evaluate", _selective_fake_evaluate(original, {"MAE": float("nan"), "RMSE": 1.0, "R2": 0.5})
    )
    registry = dict(MODEL_REGISTRY)
    registry["NanMetric"] = _entry("NanMetric", lambda: _ConstantEstimator(_MARKER))

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["NanMetric", "Ridge"], registry=registry)

    outcome = run["outcomes"]["NanMetric"]
    assert outcome.status == "failed"
    assert outcome.reason_code == NON_FINITE_METRIC
    assert outcome.reason_params == {"model_id": "NanMetric", "non_finite_metric_names": ["MAE"]}
    assert outcome.metrics is None
    assert run["outcomes"]["Ridge"].status == "success"


def test_positive_inf_metric_makes_candidate_unusable(monkeypatch):
    import src.services.ml_trainer as ml_trainer

    original = ml_trainer._evaluate
    monkeypatch.setattr(
        ml_trainer,
        "_evaluate",
        _selective_fake_evaluate(original, {"MAE": 1.0, "RMSE": float("inf"), "R2": 0.5}),
    )
    registry = dict(MODEL_REGISTRY)
    registry["InfMetric"] = _entry("InfMetric", lambda: _ConstantEstimator(_MARKER))

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["InfMetric", "Ridge"], registry=registry)

    outcome = run["outcomes"]["InfMetric"]
    assert outcome.status == "failed"
    assert outcome.reason_code == NON_FINITE_METRIC
    assert outcome.reason_params["non_finite_metric_names"] == ["RMSE"]
    assert run["outcomes"]["Ridge"].status == "success"


def test_negative_inf_metric_makes_candidate_unusable(monkeypatch):
    import src.services.ml_trainer as ml_trainer

    original = ml_trainer._evaluate
    monkeypatch.setattr(
        ml_trainer,
        "_evaluate",
        _selective_fake_evaluate(original, {"MAE": 1.0, "RMSE": 1.0, "R2": float("-inf")}),
    )
    registry = dict(MODEL_REGISTRY)
    registry["NegInfMetric"] = _entry("NegInfMetric", lambda: _ConstantEstimator(_MARKER))

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["NegInfMetric", "Ridge"], registry=registry)

    outcome = run["outcomes"]["NegInfMetric"]
    assert outcome.status == "failed"
    assert outcome.reason_code == NON_FINITE_METRIC
    assert outcome.reason_params["non_finite_metric_names"] == ["R2"]
    assert run["outcomes"]["Ridge"].status == "success"


def test_no_raw_non_finite_value_appears_in_reason_params(monkeypatch):
    import src.services.ml_trainer as ml_trainer

    original = ml_trainer._evaluate
    monkeypatch.setattr(
        ml_trainer, "_evaluate", _selective_fake_evaluate(original, {"MAE": float("nan"), "RMSE": 1.0, "R2": 0.5})
    )
    registry = dict(MODEL_REGISTRY)
    registry["NanMetric"] = _entry("NanMetric", lambda: _ConstantEstimator(_MARKER))

    df = _multi_series_df()
    run = train_candidates(df, FEATURE_COLUMNS, ["NanMetric"], registry=registry)
    outcome = run["outcomes"]["NanMetric"]
    for value in outcome.reason_params.values():
        if isinstance(value, float):
            assert math.isfinite(value)


# ---------------------------------------------------------------------------
# No winner authority yet (Slice 3A boundary)
# ---------------------------------------------------------------------------


def test_train_candidates_does_not_reference_runtime_cost_tier():
    # Structural check, not a doc-text scan: CandidateOutcome (the only data
    # train_candidates() produces or consumes) carries no runtime_cost_tier
    # field at all, so nothing downstream can read it from this module's
    # output regardless of what its docstrings say in prose.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(CandidateOutcome)}
    assert "runtime_cost_tier" not in field_names


# Note: the Slice 3A boundary test that asserted tie_break_applied /
# insufficient_successful_candidates were NOT YET implemented has been
# removed -- Slice 3B-Selection now implements both by design (see
# tests/test_ml_trainer_selection.py), so that assertion is obsolete, not
# weakened.


# ---------------------------------------------------------------------------
# Backward-compatible train_and_select_model()
# ---------------------------------------------------------------------------


def test_train_and_select_model_default_still_uses_legacy_two_candidates():
    df = _multi_series_df()
    result = train_and_select_model(df, FEATURE_COLUMNS)
    assert set(result["candidate_metrics"].keys()) == {"Ridge", "RandomForestRegressor"}
    assert result["post_training_viability"]["status"] == "PASS"


def test_train_and_select_model_honors_explicit_selected_model_ids():
    df = _multi_series_df()
    result = train_and_select_model(df, FEATURE_COLUMNS, selected_model_ids=["Ridge", "RandomForestRegressor"])
    assert set(result["candidate_metrics"].keys()) == {"Ridge", "RandomForestRegressor"}
    assert result["selected_model_name"] in {"Ridge", "RandomForestRegressor"}


def test_train_and_select_model_excludes_failed_candidates_from_usable_pool():
    registry = dict(MODEL_REGISTRY)
    registry["BrokenFit"] = _entry("BrokenFit", lambda: _AlwaysFailsEstimator())

    df = _multi_series_df()
    result = train_and_select_model(
        df,
        FEATURE_COLUMNS,
        selected_model_ids=["BrokenFit", "Ridge", "RandomForestRegressor"],
        registry=registry,
    )
    assert "BrokenFit" not in result["candidate_metrics"]
    assert result["post_training_viability"]["status"] == "PASS"
    assert result["selected_model_name"] in {"Ridge", "RandomForestRegressor"}
    assert result["candidate_outcomes"]["BrokenFit"].status == "failed"


def test_train_and_select_model_declares_no_winner_when_only_one_candidate_usable():
    registry = dict(MODEL_REGISTRY)
    registry["BrokenFit"] = _entry("BrokenFit", lambda: _AlwaysFailsEstimator())

    df = _multi_series_df()
    result = train_and_select_model(
        df, FEATURE_COLUMNS, selected_model_ids=["BrokenFit", "Ridge"], registry=registry
    )
    assert result["post_training_viability"]["status"] == "FAIL"
    assert result["post_training_viability"]["reason_code"] == "insufficient_successful_candidates"
    assert result["selected_model_name"] is None


def test_train_and_select_model_declares_no_winner_when_all_candidates_fail():
    # Migrated from the Slice 3A transitional behavior (which raised
    # RuntimeError): the frozen Checkpoint 2 policy now handles "0 usable"
    # uniformly with "1 usable" as a structured FAIL, never an exception --
    # a stronger, more auditable guarantee than a bare raise.
    registry = {"BrokenFit": _entry("BrokenFit", lambda: _AlwaysFailsEstimator())}
    df = _multi_series_df()
    result = train_and_select_model(df, FEATURE_COLUMNS, selected_model_ids=["BrokenFit"], registry=registry)
    assert result["post_training_viability"]["status"] == "FAIL"
    assert result["post_training_viability"]["reason_code"] == "insufficient_successful_candidates"
    assert result["selected_model_name"] is None
