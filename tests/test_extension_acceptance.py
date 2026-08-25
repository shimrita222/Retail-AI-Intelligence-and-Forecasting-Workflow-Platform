"""Full generalized extension acceptance test (Phase 3 completion audit gap).

Proves that a brand-new, never-registered-in-production estimator can flow
through the ENTIRE Phase 3 architecture -- training, evaluation, Post-
Training Viability, deterministic winner selection, artifact persistence,
and model-card rendering -- using only the pre-existing `registry=`
injection point that `train_and_select_model()` already exposes. No
production file is modified for this test, and the test-only model is never
added to `src.services.model_registry.MODEL_REGISTRY`.

This closes the gap the Phase 3 completion audit identified: earlier tests
proved extensibility at each layer independently (registry, execution,
selection); this test proves one continuous, unbroken path across all of
them, exactly as the audit required.
"""

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from src.agents.scientist_crew import render_model_card_markdown
from src.services.ml_trainer import save_artifacts, train_and_select_model
from src.services.model_registry import (
    FAMILY_BAGGING_ENSEMBLE,
    MODEL_REGISTRY,
    PROBLEM_TYPE_REGRESSION,
    ModelEntry,
)

FUTURE_MODEL_ID = "FutureCompatibleRegressor"


def _future_compatible_factory():
    return KNeighborsRegressor(n_neighbors=3)


def _future_compatible_entry() -> ModelEntry:
    """A model never registered in production MODEL_REGISTRY, constructed
    entirely from the public ModelEntry contract -- standing in for "a
    future compatible estimator someone adds later".
    """
    return ModelEntry(
        model_id=FUTURE_MODEL_ID,
        display_name="Future Compatible Regressor (test-only, never registered in production)",
        family=FAMILY_BAGGING_ENSEMBLE,
        problem_types=(PROBLEM_TYPE_REGRESSION,),
        enabled=True,
        is_core=True,
        required_package=None,
        runtime_cost_tier="low",
        factory=_future_compatible_factory,
    )


def _synthetic_df(n=40):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=n, freq="7D")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5.0 * x1 - 1.5 * x2 + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"Date": dates, "x1": x1, "x2": x2, "Weekly_Sales": y})


def test_future_model_is_never_present_in_production_registry():
    assert FUTURE_MODEL_ID not in MODEL_REGISTRY


def test_full_extension_acceptance_path(tmp_path):
    """new ModelEntry -> registry-compatible injection -> train_and_select_model()
    -> evaluation -> Post-Training Viability -> winner selection ->
    save_artifacts() -> evaluation_report.json -> render_model_card_markdown().

    Every step below calls the exact same production function RetailFlow
    itself calls; only the `registry=` argument (a pre-existing parameter
    on train_and_select_model()/train_candidates(), not added for this
    test) differs from a normal run.
    """
    test_registry = dict(MODEL_REGISTRY)
    test_registry[FUTURE_MODEL_ID] = _future_compatible_entry()

    df = _synthetic_df()

    # train_and_select_model() -> train_candidates() -> per-candidate
    # execution -> evaluate_post_training_viability() -> select_winner().
    # All internal, all production code, all unmodified.
    result = train_and_select_model(
        df,
        ["x1", "x2"],
        selected_model_ids=["Ridge", FUTURE_MODEL_ID],
        registry=test_registry,
    )

    assert result["post_training_viability"]["status"] == "PASS"
    assert set(result["candidate_metrics"].keys()) == {"Ridge", FUTURE_MODEL_ID}
    assert result["selected_model_name"] in {"Ridge", FUTURE_MODEL_ID}
    assert result["candidate_outcomes"][FUTURE_MODEL_ID].status == "success"

    # save_artifacts() -> evaluation_report.json. Registry-agnostic:
    # save_artifacts() never imports or references MODEL_REGISTRY.
    paths = save_artifacts(result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    assert FUTURE_MODEL_ID in report["candidate_metrics"]
    assert FUTURE_MODEL_ID in report["candidate_outcomes"]
    assert FUTURE_MODEL_ID in report["usable_candidates"]

    # render_model_card_markdown() -> also registry-agnostic: it only reads
    # training_result["candidate_outcomes"]/["usable_candidates"].
    model_card_md = render_model_card_markdown(result)
    assert f"`{FUTURE_MODEL_ID}`" in model_card_md
