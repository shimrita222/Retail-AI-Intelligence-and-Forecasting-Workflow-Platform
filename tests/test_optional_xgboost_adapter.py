"""Real (non-mocked) optional-dependency adapter proof for the Phase 3
architecture. This is the test the optional-dependency CI leg is built
around (see .github/workflows/ci.yml, job `test-optional`).

Locally (xgboost not installed): the two real-execution tests below are
skipped via `pytest.mark.skipif` on dependency absence only -- they never
mask a genuine failure once xgboost IS importable, since the skip condition
checks only import availability, nothing about test outcome. The structural
test (no XGBoost-specific branch in ml_trainer.py) always runs, with or
without xgboost installed, since it never imports xgboost itself.
"""

import importlib.util
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.services.ml_trainer import train_candidates
from src.services.model_registry import MODEL_REGISTRY

_XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
_SKIP_REASON = "xgboost not installed locally; executed for real in the optional-dependency CI leg"


def _synthetic_regression_df(n=60):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="7D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3.0 * x1 - 2.0 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"Date": dates, "x1": x1, "x2": x2, "Weekly_Sales": y})


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason=_SKIP_REASON)
def test_real_xgboost_registry_entry_uses_the_real_xgboost_package():
    entry = MODEL_REGISTRY["XGBoostRegressor"]
    assert entry.required_package == "xgboost"
    model = entry.factory()
    assert model.__class__.__module__.startswith("xgboost")


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason=_SKIP_REASON)
def test_real_xgboost_adapter_trains_through_the_generic_trainer():
    """Exercises the actual MODEL_REGISTRY entry, the actual factory, the
    actual xgboost estimator, and the actual train_candidates() execution
    path -- no monkeypatching, no mocked estimator, no bypassed function.
    """
    df = _synthetic_regression_df()
    run = train_candidates(df, ["x1", "x2"], ["XGBoostRegressor"])
    outcome = run["outcomes"]["XGBoostRegressor"]

    assert outcome.status == "success"
    assert outcome.reason_code is None
    assert outcome.fitted_model is not None
    assert outcome.fitted_model.__class__.__module__.startswith("xgboost")

    assert outcome.metrics is not None
    for metric_name in ("MAE", "RMSE", "R2"):
        assert math.isfinite(outcome.metrics[metric_name])


def test_ml_trainer_has_no_xgboost_specific_branch():
    # Runs unconditionally (no xgboost import needed): proves the trainer's
    # source contains no model_id-specific special-casing for XGBoost --
    # the tests above succeed purely through the generic ModelEntry contract.
    import src.services.ml_trainer as ml_trainer

    assert "xgboost" not in inspect.getsource(ml_trainer).lower()
