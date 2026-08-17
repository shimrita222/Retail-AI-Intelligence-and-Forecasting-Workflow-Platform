"""Deterministic ML training/evaluation/selection engine.

Trains exactly two candidate models (Ridge, RandomForestRegressor) on a
chronological train/test split, evaluates MAE/RMSE/R2 on the held-out
future weeks, and deterministically selects the candidate with the
lowest RMSE. No LLM is involved in training, evaluation, or selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_FRACTION = 0.8
TARGET_COLUMN = "Weekly_Sales"
DATE_COLUMN = "Date"

CANDIDATE_FACTORIES = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}


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


def train_and_select_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Fit both candidates on the chronological train split, evaluate on
    the chronological test split, and deterministically pick the lowest-RMSE
    candidate. Returns fitted models, per-candidate metrics, and split info.
    """
    train_df, test_df, cutoff_date = chronological_split(df)

    X_train, y_train = train_df[feature_columns], train_df[target_column]
    X_test, y_test = test_df[feature_columns], test_df[target_column]

    fitted_models: dict[str, Any] = {}
    candidate_metrics: dict[str, dict[str, float]] = {}

    for name, factory in CANDIDATE_FACTORIES.items():
        model = factory()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        candidate_metrics[name] = _evaluate(y_test, predictions)
        fitted_models[name] = model

    selected_name = min(candidate_metrics, key=lambda n: candidate_metrics[n]["RMSE"])

    return {
        "selected_model_name": selected_name,
        "fitted_models": fitted_models,
        "candidate_metrics": candidate_metrics,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split_cutoff_date": str(cutoff_date.date()),
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
