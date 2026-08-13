import joblib
import numpy as np
import pandas as pd
import pytest

from src.services.data_ingestion import ingest_data
from src.services.feature_pipeline import (
    LAG_PERIODS,
    ROLLING_WINDOW,
    engineer_features,
    get_feature_columns,
)
from src.services.ml_trainer import chronological_split, save_artifacts, train_and_select_model

RAW_DIR = "data/raw"


def _synthetic_series_df():
    """One store/dept with 12 consecutive weeks and one known Type/markdown pattern."""
    dates = pd.date_range("2010-01-01", periods=12, freq="7D")
    sales = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0]
    return pd.DataFrame(
        {
            "Store": 1,
            "Dept": 1,
            "Date": dates,
            "Weekly_Sales": sales,
            "IsHoliday": [False] * 12,
            "Type": "A",
            "Size": 100000,
            "Temperature": 50.0,
            "Fuel_Price": 3.0,
            "MarkDown1": [np.nan] * 6 + [10.0] * 6,
            "MarkDown2": np.nan,
            "MarkDown3": np.nan,
            "MarkDown4": np.nan,
            "MarkDown5": np.nan,
            "CPI": 200.0,
            "Unemployment": 8.0,
        }
    )


def test_lag_features_have_no_leakage():
    df = engineer_features(_synthetic_series_df())
    # Row for week index i (0-based) should have Lag1 == sales[i-1] and Lag4 == sales[i-4]
    row_week5 = df[df["Date"] == pd.Timestamp("2010-01-01") + pd.Timedelta(days=7 * 5)].iloc[0]
    assert row_week5["Weekly_Sales_Lag1"] == 140.0  # previous week's (index 4) actual sales
    assert row_week5["Weekly_Sales_Lag4"] == 110.0  # four weeks prior (index 1)
    # No lag column may equal or exceed the row's own current sales value by construction bias
    assert (df["Weekly_Sales_Lag1"] < df["Weekly_Sales"]).all()


def test_rows_without_enough_history_are_dropped():
    df = engineer_features(_synthetic_series_df())
    # With ROLLING_WINDOW=4 and max(LAG_PERIODS)=4, the first 4 weeks lack full history
    assert len(df) == 12 - ROLLING_WINDOW
    assert df["Weekly_Sales_Lag1"].isna().sum() == 0
    assert df["Weekly_Sales_Lag4"].isna().sum() == 0
    assert df[f"Weekly_Sales_RollingMean{ROLLING_WINDOW}"].isna().sum() == 0


def test_markdown_imputed_to_zero():
    df = engineer_features(_synthetic_series_df())
    assert df["MarkDown2"].isna().sum() == 0
    assert (df["MarkDown2"] == 0.0).all()


def test_type_one_hot_encoded():
    df = engineer_features(_synthetic_series_df())
    assert "Type_A" in df.columns
    assert "Type" not in df.columns
    assert (df["Type_A"] == 1).all()


def test_get_feature_columns_excludes_identifiers_and_target():
    df = engineer_features(_synthetic_series_df())
    cols = get_feature_columns(df)
    for excluded in ("Store", "Dept", "Date", "Weekly_Sales"):
        assert excluded not in cols
    assert "Weekly_Sales_Lag1" in cols


def test_chronological_split_boundary_is_strict():
    dates = pd.date_range("2010-01-01", periods=100, freq="7D")
    df = pd.DataFrame({"Date": dates, "Weekly_Sales": np.arange(100.0)})
    train_df, test_df, cutoff = chronological_split(df, train_fraction=0.8)

    assert train_df["Date"].max() <= cutoff
    assert test_df["Date"].min() > cutoff
    assert train_df["Date"].max() < test_df["Date"].min()
    assert len(train_df) + len(test_df) == len(df)
    assert 0.7 <= len(train_df) / len(df) <= 0.85


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
                        "IsHoliday": False,
                        "Type": "A" if store == 1 else "B",
                        "Size": 100000,
                        "Temperature": 50.0,
                        "Fuel_Price": 3.0,
                        "MarkDown1": np.nan,
                        "MarkDown2": np.nan,
                        "MarkDown3": np.nan,
                        "MarkDown4": np.nan,
                        "MarkDown5": np.nan,
                        "CPI": 200.0,
                        "Unemployment": 8.0,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_train_and_select_model_trains_both_candidates_and_selects_lowest_rmse():
    df = engineer_features(_multi_series_df())
    feature_columns = get_feature_columns(df)
    result = train_and_select_model(df, feature_columns)

    assert set(result["candidate_metrics"].keys()) == {"Ridge", "RandomForestRegressor"}
    for metrics in result["candidate_metrics"].values():
        assert {"MAE", "RMSE", "R2"}.issubset(metrics.keys())
        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0

    best_rmse = min(m["RMSE"] for m in result["candidate_metrics"].values())
    assert result["candidate_metrics"][result["selected_model_name"]]["RMSE"] == best_rmse
    assert result["selected_model_name"] in {"Ridge", "RandomForestRegressor"}


def test_save_artifacts_writes_joblib_and_report(tmp_path):
    df = engineer_features(_multi_series_df())
    feature_columns = get_feature_columns(df)
    result = train_and_select_model(df, feature_columns)

    paths = save_artifacts(result, tmp_path)
    assert paths["model_path"].exists()
    assert paths["report_path"].exists()

    loaded_model = joblib.load(paths["model_path"])
    sample = df[feature_columns].iloc[:3]
    predictions = loaded_model.predict(sample)
    assert len(predictions) == 3


def test_full_pipeline_on_real_ingested_data(tmp_path):
    clean_df = ingest_data(RAW_DIR, tmp_path / "clean_data.csv")
    # Use a small slice of stores to keep the RandomForest fit fast in CI
    subset = clean_df[clean_df["Store"].isin([1, 2, 3])].copy()
    engineered = engineer_features(subset)
    feature_columns = get_feature_columns(engineered)

    result = train_and_select_model(engineered, feature_columns)
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0

    paths = save_artifacts(result, tmp_path / "artifacts")
    assert paths["model_path"].exists()
    assert paths["report_path"].exists()
