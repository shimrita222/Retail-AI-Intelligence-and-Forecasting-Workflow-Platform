"""Deterministic feature engineering for the retail sales forecasting model.

All transformations here are pure pandas/numpy: no LLM calls, no
randomness. Lag and rolling features use `shift(1)` before any window
aggregation so that a row's features are built exclusively from data
strictly earlier than that row's own date (leakage prevention).
"""

from __future__ import annotations

import pandas as pd

LAG_PERIODS = (1, 4)
ROLLING_WINDOW = 4
MARKDOWN_COLUMNS = ("MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5")
GROUP_KEYS = ("Store", "Dept")
IDENTIFIER_COLUMNS = ("Store", "Dept", "Date")
TARGET_COLUMN = "Weekly_Sales"


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped_sales = df.groupby(list(GROUP_KEYS))[TARGET_COLUMN]
    for lag in LAG_PERIODS:
        df[f"{TARGET_COLUMN}_Lag{lag}"] = grouped_sales.transform(lambda s, lag=lag: s.shift(lag))
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped_sales = df.groupby(list(GROUP_KEYS))[TARGET_COLUMN]

    def _shifted_rolling_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()

    def _shifted_rolling_std(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()

    df[f"{TARGET_COLUMN}_RollingMean{ROLLING_WINDOW}"] = grouped_sales.transform(_shifted_rolling_mean)
    df[f"{TARGET_COLUMN}_RollingStd{ROLLING_WINDOW}"] = grouped_sales.transform(_shifted_rolling_std)
    return df


def _impute_markdowns(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill MarkDown nulls for modeling, but first record a
    `{col}_was_missing` indicator per column so the model (and any later
    inspection) can still distinguish "markdown information was not
    recorded" from "an observed value of zero" -- the two are not the same
    fact and must not be silently collapsed into one number. See
    data/dataset_manifest.json for why NA must not be read as "no promotion".
    """
    df = df.copy()
    for col in MARKDOWN_COLUMNS:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0.0)
    return df


def _impute_economic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fill sparse CPI/Unemployment nulls with the per-store median.

    Ridge/RandomForest cannot accept NaN inputs, and CPI/Unemployment are
    store-level regional indicators, so a per-store median is a stable,
    deterministic stand-in for missing weeks of that same store's series.
    """
    df = df.copy()
    for col in ("CPI", "Unemployment"):
        if col in df.columns:
            df[col] = df.groupby("Store")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(int)
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], prefix="Type")
        for type_col in [c for c in df.columns if c.startswith("Type_")]:
            df[type_col] = df[type_col].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full deterministic feature set from clean_data.

    Steps: sort chronologically per (Store, Dept) -> lag features ->
    rolling window statistics -> markdown/economic imputation ->
    categorical encoding -> drop rows without enough history for lags.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(list(GROUP_KEYS) + ["Date"]).reset_index(drop=True)

    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _impute_markdowns(df)
    df = _impute_economic_indicators(df)
    df = _encode_categoricals(df)

    engineered_cols = [f"{TARGET_COLUMN}_Lag{lag}" for lag in LAG_PERIODS] + [
        f"{TARGET_COLUMN}_RollingMean{ROLLING_WINDOW}",
        f"{TARGET_COLUMN}_RollingStd{ROLLING_WINDOW}",
    ]
    df = df.dropna(subset=engineered_cols).reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model input column names: everything except identifiers/target."""
    exclude = set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN}
    return [c for c in df.columns if c not in exclude]
