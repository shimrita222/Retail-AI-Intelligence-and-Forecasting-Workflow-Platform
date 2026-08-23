import pandas as pd

from src.services.feature_pipeline import engineer_features, get_feature_columns
from src.services.modeling_eligibility import (
    DOCUMENTED_WATCHLIST,
    MODELING_EXCLUSIONS,
    get_exclusions,
    modeling_eligible_mask,
    summarize_modeling_population,
)


def _df_with_depts(depts):
    dates = pd.date_range("2010-01-01", periods=6, freq="7D")
    frames = []
    for dept in depts:
        frames.append(
            pd.DataFrame(
                {
                    "Store": 1,
                    "Dept": dept,
                    "Date": dates,
                    "Weekly_Sales": [10.0 + i for i in range(6)],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_dept_47_is_the_only_configured_exclusion():
    assert set(MODELING_EXCLUSIONS.keys()) == {47}
    assert get_exclusions()[47].decision_type == "modeling_eligibility"
    assert get_exclusions()[47].descriptive_status == "included"
    assert get_exclusions()[47].modeling_status == "excluded"


def test_dept_78_18_54_are_not_excluded():
    for dept in (78, 18, 54):
        assert dept not in MODELING_EXCLUSIONS
    # documented, but non-exclusionary
    assert 78 in DOCUMENTED_WATCHLIST
    assert 18 in DOCUMENTED_WATCHLIST
    assert 54 in DOCUMENTED_WATCHLIST


def test_modeling_eligible_mask_excludes_only_dept_47():
    df = _df_with_depts([1, 18, 47, 54, 78])
    mask = modeling_eligible_mask(df)

    eligible_depts = set(df.loc[mask, "Dept"].unique())
    excluded_depts = set(df.loc[~mask, "Dept"].unique())

    assert excluded_depts == {47}
    assert eligible_depts == {1, 18, 54, 78}


def test_dept_47_present_in_descriptive_feature_engineering():
    # Descriptive/EDA paths (engineer_features called directly on the full
    # clean data) must still see Department 47 -- only the Flow's modeling
    # population is filtered, not feature_pipeline.engineer_features itself.
    df = _df_with_depts([1, 47])
    engineered = engineer_features(df)
    assert 47 in engineered["Dept"].unique()
    feature_columns = get_feature_columns(engineered)
    assert "Dept" not in feature_columns  # identifier, excluded from model inputs as before


def test_summarize_modeling_population_reports_dept_47_exclusion():
    df = _df_with_depts([1, 18, 47, 54, 78])
    summary = summarize_modeling_population(df)

    assert summary["total_clean_rows"] == len(df)
    assert summary["excluded_departments"] == [47]
    assert summary["excluded_rows"] == 6  # one dept's worth of synthetic rows
    assert summary["eligible_rows"] == len(df) - 6
    assert 47 in summary["exclusion_reasons"]
    assert "Unresolved business semantics" in summary["exclusion_reasons"][47]
    assert any("negative-sales rate" in e for e in get_exclusions()[47].evidence)
    assert summary["watchlist"] == DOCUMENTED_WATCHLIST


def test_summarize_modeling_population_with_no_excluded_depts_present():
    df = _df_with_depts([1, 18, 54, 78])
    summary = summarize_modeling_population(df)

    assert summary["excluded_departments"] == []
    assert summary["excluded_rows"] == 0
    assert summary["eligible_rows"] == summary["total_clean_rows"]
