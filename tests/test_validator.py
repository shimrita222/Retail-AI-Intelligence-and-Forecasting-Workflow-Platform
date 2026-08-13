import pandas as pd
import pytest

from src.services.contract_validator import validate_dataframe
from src.services.data_ingestion import ingest_data, load_raw_tables

RAW_DIR = "data/raw"


def _base_contract():
    return {
        "primary_keys": ["Store", "Dept", "Date"],
        "target_column": "Weekly_Sales",
        "columns": {
            "Store": {"dtype": "int", "nullable": False},
            "Dept": {"dtype": "int", "nullable": False},
            "Date": {"dtype": "datetime", "nullable": False},
            "Weekly_Sales": {"dtype": "float", "nullable": False, "min": -5000, "max": 800000},
            "IsHoliday": {"dtype": "bool", "nullable": False},
        },
    }


def _good_df():
    return pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Dept": [1, 2, 1],
            "Date": pd.to_datetime(["2010-02-05", "2010-02-05", "2010-02-05"]),
            "Weekly_Sales": [24924.5, 46039.49, 10000.0],
            "IsHoliday": [False, True, False],
        }
    )


def test_validate_dataframe_passes_on_clean_data():
    result = validate_dataframe(_good_df(), _base_contract())
    assert result["status"] == "PASS"
    assert result["errors"] == []


def test_validate_dataframe_fails_on_missing_column():
    df = _good_df().drop(columns=["IsHoliday"])
    result = validate_dataframe(df, _base_contract())
    assert result["status"] == "FAIL"
    assert any("IsHoliday" in e for e in result["errors"])


def test_validate_dataframe_fails_on_null_primary_key():
    df = _good_df()
    df.loc[0, "Dept"] = None
    result = validate_dataframe(df, _base_contract())
    assert result["status"] == "FAIL"
    assert any("primary key" in e for e in result["errors"])


def test_validate_dataframe_fails_on_null_target():
    df = _good_df()
    df.loc[0, "Weekly_Sales"] = None
    result = validate_dataframe(df, _base_contract())
    assert result["status"] == "FAIL"
    assert any("Weekly_Sales" in e and "null" in e for e in result["errors"])


def test_validate_dataframe_fails_on_duplicate_primary_key():
    df = _good_df()
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    result = validate_dataframe(df, _base_contract())
    assert result["status"] == "FAIL"
    assert any("duplicate" in e for e in result["errors"])


def test_validate_dataframe_fails_on_out_of_bounds_target():
    df = _good_df()
    df.loc[0, "Weekly_Sales"] = 5_000_000.0
    result = validate_dataframe(df, _base_contract())
    assert result["status"] == "FAIL"
    assert any("maximum" in e for e in result["errors"])


def test_validate_dataframe_fails_on_wrong_dtype():
    df = _good_df()
    df["Store"] = df["Store"].astype(str) + "abc"
    result = validate_dataframe(df, _base_contract())
    assert result["status"] == "FAIL"
    assert any("Store" in e and "dtype" in e for e in result["errors"])


def test_ingestion_joins_and_produces_zero_null_primary_keys(tmp_path):
    df = ingest_data(RAW_DIR, tmp_path / "clean_data.csv")
    assert {"Store", "Dept", "Date", "Weekly_Sales", "Type", "Size", "IsHoliday"}.issubset(df.columns)
    assert df[["Store", "Dept", "Date", "Weekly_Sales"]].isna().sum().sum() == 0
    assert (tmp_path / "clean_data.csv").exists()


def test_ingestion_passes_contract_validation(tmp_path):
    df = ingest_data(RAW_DIR, tmp_path / "clean_data.csv")
    result = validate_dataframe(df, _base_contract())
    assert result["status"] == "PASS", result["errors"]


def test_load_raw_tables_raises_on_missing_dir(tmp_path):
    with pytest.raises(Exception):
        load_raw_tables(tmp_path / "does_not_exist")
