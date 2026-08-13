"""Deterministic ingestion of the Retail Data Analytics source CSVs.

Reads stores.csv, features.csv, and train.csv from a raw data directory,
joins them on the documented keys, performs primary-key/target null
cleaning, and writes a single clean_data.csv artifact. No LLM calls,
no randomness: identical inputs always produce identical output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_SOURCE_FILES = ("stores.csv", "features.csv", "train.csv")
PRIMARY_KEYS = ("Store", "Dept", "Date")
TARGET_COLUMN = "Weekly_Sales"


class DataIngestionError(Exception):
    """Raised when raw source files are missing or malformed."""


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise DataIngestionError(f"Required source file not found: {path}")
    return pd.read_csv(path, **kwargs)


def _parse_dates(df: pd.DataFrame, column: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], format="%d/%m/%Y", errors="coerce")
    return df


def _coerce_bool(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    if df[column].dtype == object:
        df[column] = (
            df[column].astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False})
        )
    df[column] = df[column].astype(bool)
    return df


def load_raw_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load and lightly normalize the three raw source tables."""
    raw_dir = Path(raw_dir)

    missing = [f for f in REQUIRED_SOURCE_FILES if not (raw_dir / f).exists()]
    if missing:
        raise DataIngestionError(
            f"Missing required raw source file(s) in {raw_dir}: {missing}. "
            f"Expected exactly: {list(REQUIRED_SOURCE_FILES)}"
        )

    stores = _read_csv(raw_dir / "stores.csv")
    features = _read_csv(raw_dir / "features.csv")
    train = _read_csv(raw_dir / "train.csv")

    features = _parse_dates(features)
    train = _parse_dates(train)

    features = _coerce_bool(features, "IsHoliday")
    train = _coerce_bool(train, "IsHoliday")

    return {"stores": stores, "features": features, "train": train}


def ingest_data(raw_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Join stores/features/train deterministically and write clean_data.csv.

    Join plan:
      1. train (Store, Dept, Date) LEFT JOIN stores on (Store)
      2. result LEFT JOIN features on (Store, Date)
    Rows missing a primary key (Store, Dept, Date) or the target
    (Weekly_Sales) are dropped, since the contract validator requires
    zero nulls there. All other nulls (CPI, Unemployment, MarkDown*)
    are left intact for the deterministic feature pipeline to handle.
    """
    tables = load_raw_tables(raw_dir)
    stores, features, train = tables["stores"], tables["features"], tables["train"]

    merged = train.merge(stores, on="Store", how="left", validate="many_to_one")

    merged = merged.merge(
        features,
        on=["Store", "Date"],
        how="left",
        suffixes=("", "_features"),
        validate="many_to_one",
    )

    if "IsHoliday_features" in merged.columns:
        merged = merged.drop(columns=["IsHoliday_features"])

    before = len(merged)
    merged = merged.dropna(subset=list(PRIMARY_KEYS) + [TARGET_COLUMN])
    dropped = before - len(merged)
    if dropped:
        logger.warning(
            "Dropped %d row(s) missing a primary key or %s during ingestion", dropped, TARGET_COLUMN
        )

    merged["Store"] = merged["Store"].astype(int)
    merged["Dept"] = merged["Dept"].astype(int)
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged[TARGET_COLUMN] = merged[TARGET_COLUMN].astype(float)

    merged = merged.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("Wrote clean data (%d rows, %d cols) to %s", *merged.shape, output_path)

    return merged
