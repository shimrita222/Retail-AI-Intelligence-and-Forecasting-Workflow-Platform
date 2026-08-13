"""Deterministic, pure-Python validation of clean_data.csv against a
dataset_contract.json produced by the Data Contract Architect agent.

This module makes NO decisions with an LLM. It only performs
structural/statistical checks: column presence, dtype compatibility,
null constraints, primary-key uniqueness, and numeric bounds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

VALID_STATUSES = ("PASS", "FAIL")

_DTYPE_CHECKS = {
    "int": lambda s: pd.api.types.is_integer_dtype(s) or _coercible_numeric(s, integer_like=True),
    "float": lambda s: pd.api.types.is_numeric_dtype(s) or _coercible_numeric(s),
    "bool": lambda s: pd.api.types.is_bool_dtype(s) or _coercible_bool(s),
    "datetime": lambda s: pd.api.types.is_datetime64_any_dtype(s) or _coercible_datetime(s),
    "string": lambda s: True,  # any dtype is acceptable as a string-like column
}


def _coercible_numeric(series: pd.Series, integer_like: bool = False) -> bool:
    try:
        coerced = pd.to_numeric(series, errors="coerce")
    except (TypeError, ValueError):
        return False
    if coerced.isna().sum() > series.isna().sum():
        return False
    if integer_like:
        non_null = coerced.dropna()
        return bool((non_null == non_null.astype(int)).all())
    return True


def _coercible_bool(series: pd.Series) -> bool:
    allowed = {True, False, "TRUE", "FALSE", "True", "False", 0, 1}
    return set(series.dropna().unique()).issubset(allowed)


def _coercible_datetime(series: pd.Series) -> bool:
    try:
        coerced = pd.to_datetime(series, errors="coerce")
    except (TypeError, ValueError):
        return False
    return coerced.isna().sum() <= series.isna().sum()


def load_contract(contract_path: str | Path) -> dict[str, Any]:
    contract_path = Path(contract_path)
    with contract_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_clean_data(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def validate_dataframe(df: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    """Validate a DataFrame in-memory against a parsed contract dict.

    Returns {"status": "PASS"|"FAIL", "errors": [...], "warnings": [...]}.
    """
    errors: list[str] = []
    warnings: list[str] = []

    columns_spec: dict[str, Any] = contract.get("columns", {})
    primary_keys: list[str] = contract.get("primary_keys", [])
    target_column: str | None = contract.get("target_column")

    # 1. Column presence
    missing_columns = [c for c in columns_spec if c not in df.columns]
    for col in missing_columns:
        errors.append(f"Missing required column: '{col}'")

    present_spec = {c: spec for c, spec in columns_spec.items() if c not in missing_columns}

    # 2. Dtype compatibility
    for col, spec in present_spec.items():
        expected_dtype = spec.get("dtype")
        checker = _DTYPE_CHECKS.get(expected_dtype)
        if checker is None:
            warnings.append(f"Unknown dtype '{expected_dtype}' declared for column '{col}'; skipped type check")
            continue
        if not checker(df[col]):
            errors.append(
                f"Column '{col}' failed dtype check: expected '{expected_dtype}', "
                f"observed pandas dtype '{df[col].dtype}'"
            )

    # 3. Null constraints (contract-declared non-nullable columns)
    for col, spec in present_spec.items():
        if spec.get("nullable", True) is False:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                errors.append(f"Column '{col}' is declared non-nullable but has {null_count} null value(s)")

    # 4. Primary key nulls + uniqueness
    missing_pk_cols = [c for c in primary_keys if c not in df.columns]
    for col in missing_pk_cols:
        errors.append(f"Primary key column '{col}' not present in data")

    usable_pk = [c for c in primary_keys if c in df.columns]
    if usable_pk:
        pk_null_count = int(df[usable_pk].isna().any(axis=1).sum())
        if pk_null_count > 0:
            errors.append(f"{pk_null_count} row(s) have null value(s) in primary key columns {usable_pk}")

        duplicate_count = int(df.duplicated(subset=usable_pk).sum())
        if duplicate_count > 0:
            errors.append(f"{duplicate_count} duplicate row(s) found for primary key {usable_pk}")

    # 5. Target column checks: zero nulls + numeric bounds
    if target_column:
        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not present in data")
        else:
            target_null_count = int(df[target_column].isna().sum())
            if target_null_count > 0:
                errors.append(f"Target column '{target_column}' has {target_null_count} null value(s)")

            target_spec = columns_spec.get(target_column, {})
            numeric_target = pd.to_numeric(df[target_column], errors="coerce")

            min_bound = target_spec.get("min")
            if min_bound is not None:
                below = int((numeric_target < min_bound).sum())
                if below > 0:
                    errors.append(
                        f"{below} row(s) have '{target_column}' below the contract minimum of {min_bound}"
                    )

            max_bound = target_spec.get("max")
            if max_bound is not None:
                above = int((numeric_target > max_bound).sum())
                if above > 0:
                    errors.append(
                        f"{above} row(s) have '{target_column}' above the contract maximum of {max_bound}"
                    )

    status = "FAIL" if errors else "PASS"
    return {"status": status, "errors": errors, "warnings": warnings}


def validate_contract(data_path: str | Path, contract_path: str | Path) -> dict[str, Any]:
    """Load clean_data.csv and dataset_contract.json from disk and validate."""
    df = load_clean_data(data_path)
    contract = load_contract(contract_path)
    return validate_dataframe(df, contract)
