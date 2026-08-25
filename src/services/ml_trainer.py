"""Deterministic ML training/evaluation engine.

Slice 3A generalizes this module to train an explicit, pre-approved list of
model_ids (`selected_model_ids`), resolved through the single authoritative
`src.services.model_registry.MODEL_REGISTRY`, with per-candidate failure
isolation and non-finite-metric exclusion. No LLM is involved in training,
evaluation, or any decision here.

This module does NOT select candidates. Candidate eligibility/selection is
decided exactly once, upstream, by
`src.services.candidate_selection.evaluate_pre_training_gate()` (invoked by
RetailFlow before this module is ever called). `train_candidates()` and
`train_and_select_model()` below must never call `select_candidates()` or
`evaluate_pre_training_gate()` themselves, and must never add, drop, or
substitute a candidate beyond the exact `selected_model_ids` they were given.

This module also does NOT implement the final frozen Phase 3 winner policy
(lowest RMSE, tie -> model_id alphabetical) or the Post-Training Viability
Check (`insufficient_successful_candidates`) -- both are Slice 3B-Selection's
responsibility. `train_and_select_model()`'s winner logic (plain lowest-RMSE
among successful candidates, Python `min()`'s incidental first-inserted-wins
behavior on an exact tie) is preserved ONLY as a TRANSITIONAL, backward-
compatible default for callers that do not yet pass `selected_model_ids`
explicitly (i.e. pre-Slice-3 tests). It is explicitly NOT the frozen Phase 3
winner rule.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.services.model_registry import MODEL_REGISTRY, ModelEntry

TRAIN_FRACTION = 0.8
TARGET_COLUMN = "Weekly_Sales"
DATE_COLUMN = "Date"

# TRANSITIONAL ONLY: fixed default candidate list for legacy direct callers
# of train_and_select_model() that do not pass selected_model_ids explicitly
# (pre-Slice-3 tests/callers). This is a literal backward-compatibility
# default, not a candidate-selection decision -- it never calls
# select_candidates() or evaluate_pre_training_gate().
_LEGACY_DEFAULT_MODEL_IDS: tuple[str, ...] = ("Ridge", "RandomForestRegressor")

# Structured, language-neutral per-candidate failure reason codes.
CANDIDATE_TRAINING_FAILED = "candidate_training_failed"
NON_FINITE_METRIC = "non_finite_metric"
UNKNOWN_MODEL_ID = "unknown_model_id"

_METRIC_NAMES = ("MAE", "RMSE", "R2")


@dataclass(frozen=True)
class CandidateOutcome:
    model_id: str
    status: str  # "success" | "failed"
    metrics: dict[str, float] | None
    fitted_model: Any | None
    reason_code: str | None
    reason_params: dict[str, Any] | None


def chronological_split(
    df: pd.DataFrame, date_col: str = DATE_COLUMN, train_fraction: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by a single global date cutoff: first train_fraction of the
    calendar timeline goes to train, the remaining tail goes to test.
    """
    unique_dates = sorted(pd.to_datetime(df[date_col]).unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough distinct dates to perform a chronological split")

    cutoff_idx = max(0, min(len(unique_dates) - 2, int(len(unique_dates) * train_fraction) - 1))
    cutoff_date = pd.Timestamp(unique_dates[cutoff_idx])

    train_df = df[pd.to_datetime(df[date_col]) <= cutoff_date].copy()
    test_df = df[pd.to_datetime(df[date_col]) > cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty train or test partition")

    return train_df, test_df, cutoff_date


def _evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _run_single_candidate(
    model_id: str,
    entry: ModelEntry | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> CandidateOutcome:
    """Execute one pre-approved candidate in isolation: resolve -> construct
    -> fit -> predict -> evaluate. Any failure at any stage, or a non-finite
    metric, produces a structured "failed" outcome without raising -- the
    caller is responsible for continuing with the remaining candidates.
    """
    if entry is None:
        return CandidateOutcome(
            model_id=model_id,
            status="failed",
            metrics=None,
            fitted_model=None,
            reason_code=UNKNOWN_MODEL_ID,
            reason_params={"model_id": model_id},
        )

    stage = "construct"
    try:
        model = entry.factory()
        stage = "fit"
        model.fit(X_train, y_train)
        stage = "predict"
        predictions = model.predict(X_test)
        stage = "evaluate"
        metrics = _evaluate(y_test, predictions)
    except Exception as exc:  # noqa: BLE001 - isolate any candidate failure, never abort the others
        return CandidateOutcome(
            model_id=model_id,
            status="failed",
            metrics=None,
            fitted_model=None,
            reason_code=CANDIDATE_TRAINING_FAILED,
            reason_params={"model_id": model_id, "stage": stage, "exception_type": exc.__class__.__name__},
        )

    non_finite_metric_names = [name for name in _METRIC_NAMES if not math.isfinite(metrics[name])]
    if non_finite_metric_names:
        # Never coerce a non-finite value into a fake finite one, and never
        # place the raw NaN/inf float into JSON-facing reason metadata.
        return CandidateOutcome(
            model_id=model_id,
            status="failed",
            metrics=None,
            fitted_model=None,
            reason_code=NON_FINITE_METRIC,
            reason_params={"model_id": model_id, "non_finite_metric_names": non_finite_metric_names},
        )

    return CandidateOutcome(
        model_id=model_id,
        status="success",
        metrics=metrics,
        fitted_model=model,
        reason_code=None,
        reason_params=None,
    )


def train_candidates(
    df: pd.DataFrame,
    feature_columns: list[str],
    selected_model_ids: list[str],
    target_column: str = TARGET_COLUMN,
    registry: dict[str, ModelEntry] | None = None,
) -> dict[str, Any]:
    """Deterministically execute exactly the pre-approved `selected_model_ids`.

    Trains only what it is told to train -- never selects, never adds, never
    substitutes a candidate. Isolates per-candidate construct/fit/predict/
    evaluate failures and non-finite metrics so one bad candidate never
    aborts the others. Returns raw per-candidate outcomes only; no winner is
    chosen here (that is Slice 3B-Selection's responsibility).
    """
    source = registry if registry is not None else MODEL_REGISTRY
    train_df, test_df, cutoff_date = chronological_split(df)

    X_train, y_train = train_df[feature_columns], train_df[target_column]
    X_test, y_test = test_df[feature_columns], test_df[target_column]

    outcomes: dict[str, CandidateOutcome] = {}
    for model_id in selected_model_ids:
        entry = source.get(model_id)
        outcomes[model_id] = _run_single_candidate(model_id, entry, X_train, y_train, X_test, y_test)

    return {
        "outcomes": outcomes,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split_cutoff_date": str(cutoff_date.date()),
    }


def train_and_select_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
    selected_model_ids: list[str] | None = None,
    registry: dict[str, ModelEntry] | None = None,
) -> dict[str, Any]:
    """Backward-compatible entrypoint used by RetailFlow and existing tests.

    `selected_model_ids`, when provided, must be the exact, already
    gate-approved list from
    `candidate_selection.evaluate_pre_training_gate().selected_model_ids` --
    this function does not validate, re-select, or second-guess that list.

    TRANSITIONAL: if `selected_model_ids` is omitted, defaults to the legacy
    two-candidate list (`_LEGACY_DEFAULT_MODEL_IDS`) for callers that predate
    Slice 3 -- this is a fixed literal default, not a selection decision, and
    never calls select_candidates()/evaluate_pre_training_gate(). The winner
    chosen below (lowest RMSE among successful, finite candidates) is a
    TRANSITIONAL selection, not the frozen Phase 3 final winner policy (RMSE
    ascending, tie -> model_id alphabetical), which Slice 3B-Selection will
    implement.
    """
    ids = list(selected_model_ids) if selected_model_ids is not None else list(_LEGACY_DEFAULT_MODEL_IDS)
    run = train_candidates(df, feature_columns, ids, target_column=target_column, registry=registry)
    outcomes = run["outcomes"]

    successful = {model_id: outcome for model_id, outcome in outcomes.items() if outcome.status == "success"}
    if not successful:
        raise RuntimeError(
            "train_and_select_model: no candidate succeeded "
            f"(outcomes={ {model_id: outcome.reason_code for model_id, outcome in outcomes.items()} })"
        )

    # TRANSITIONAL winner selection -- see module/function docstrings.
    selected_name = min(successful, key=lambda model_id: successful[model_id].metrics["RMSE"])

    return {
        "selected_model_name": selected_name,
        "fitted_models": {model_id: outcome.fitted_model for model_id, outcome in successful.items()},
        "candidate_metrics": {model_id: outcome.metrics for model_id, outcome in successful.items()},
        "candidate_outcomes": outcomes,  # additive: full per-candidate audit trail, including failures
        "feature_columns": run["feature_columns"],
        "target_column": run["target_column"],
        "train_rows": run["train_rows"],
        "test_rows": run["test_rows"],
        "split_cutoff_date": run["split_cutoff_date"],
    }


def save_artifacts(result: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Persist the selected model (.joblib) and a JSON evaluation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "selected_model.joblib"
    joblib.dump(result["fitted_models"][result["selected_model_name"]], model_path)

    report = {
        "selected_model_name": result["selected_model_name"],
        "candidate_metrics": result["candidate_metrics"],
        "feature_columns": result["feature_columns"],
        "target_column": result["target_column"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "split_cutoff_date": result["split_cutoff_date"],
    }
    if "modeling_population" in result:
        report["modeling_population"] = result["modeling_population"]
    report_path = output_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {"model_path": model_path, "report_path": report_path}
