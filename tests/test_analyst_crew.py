import numpy as np
import pandas as pd

from src.agents.analyst_crew import compute_micro_inspection
from src.services.feature_pipeline import engineer_features


def _base_row(store, dept, date, sales, **overrides):
    row = {
        "Store": store,
        "Dept": dept,
        "Date": date,
        "Weekly_Sales": sales,
        "IsHoliday": False,
        "MarkDown1": np.nan,
        "MarkDown2": np.nan,
        "MarkDown3": np.nan,
        "MarkDown4": np.nan,
        "MarkDown5": np.nan,
    }
    row.update(overrides)
    return row


def _with_mixed_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """compute_micro_inspection's holiday-spike unstack needs both IsHoliday
    values present somewhere in the frame; add one unrelated holiday row so
    small synthetic fixtures don't hit that (pre-existing, out-of-scope)
    edge case.
    """
    extra = _base_row(999, 999, pd.Timestamp("2010-06-04"), 1.0, IsHoliday=True)
    return pd.concat([df, pd.DataFrame([extra])], ignore_index=True)


# ---------------------------------------------------------------------------
# MarkDown grain: one Store-Date markdown value must not be double-counted
# across departments that share that store/date after the merge.
# ---------------------------------------------------------------------------


def test_markdown_outliers_are_not_duplicated_across_departments():
    date = pd.Timestamp("2012-02-03")
    # Same Store-Date MarkDown1 value (an extreme one), repeated across 5
    # departments -- this is exactly the merge-duplication pattern described
    # in the remediation brief (one Store-Date markdown event repeated once
    # per department row).
    rows = [
        _base_row(2, dept, date, 100.0, MarkDown1=50000.0)
        for dept in (79, 80, 81, 82, 83)
    ]
    # A few ordinary, non-extreme markdown rows so the IQR bound isn't
    # degenerate.
    for i, dept in enumerate((1, 2, 3, 4, 5)):
        rows.append(
            _base_row(2, dept, date - pd.Timedelta(weeks=i + 1), 100.0, MarkDown1=100.0 + i)
        )
    df = _with_mixed_holiday(pd.DataFrame(rows))

    findings = compute_micro_inspection(df)
    md1_outliers = [o for o in findings["extreme_markdown_outliers"] if o["column"] == "MarkDown1"]

    # The extreme value must be reported once (one Store-Date observation),
    # not once per department that happens to share that store/date.
    assert len(md1_outliers) == 1
    assert md1_outliers[0]["value"] == 50000.0
    assert "Dept" not in md1_outliers[0]
    assert md1_outliers[0]["Store"] == 2


def test_markdown_effect_reports_grain_note():
    date = pd.Timestamp("2012-02-03")
    rows = [
        _base_row(1, 1, date, 100.0, MarkDown1=500.0),
        _base_row(1, 2, date, 200.0, MarkDown1=500.0),
        _base_row(1, 1, date - pd.Timedelta(weeks=1), 90.0),
    ]
    df = _with_mixed_holiday(pd.DataFrame(rows))
    from src.agents.analyst_crew import compute_business_intelligence

    micro = compute_micro_inspection(df)
    bi = compute_business_intelligence(df, micro)

    assert "grain_note" in bi["markdown_effect"]
    assert "Store+Date" in bi["markdown_effect"]["grain_note"]


# ---------------------------------------------------------------------------
# MarkDown missingness indicator: NaN must be distinguishable from zero after
# imputation.
# ---------------------------------------------------------------------------


def test_markdown_missing_indicator_added_before_zero_fill():
    dates = pd.date_range("2010-01-01", periods=6, freq="7D")
    df = pd.DataFrame(
        {
            "Store": 1,
            "Dept": 1,
            "Date": dates,
            "Weekly_Sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "IsHoliday": False,
            "MarkDown1": [np.nan, np.nan, np.nan, 5.0, np.nan, 0.0],
        }
    )
    engineered = engineer_features(df)

    assert "MarkDown1_was_missing" in engineered.columns
    # Rows where MarkDown1 was NaN -> was_missing=1, value zero-filled.
    missing_rows = engineered[engineered["MarkDown1_was_missing"] == 1]
    assert (missing_rows["MarkDown1"] == 0.0).all()
    # The genuinely-observed zero must still read was_missing=0.
    observed_zero_rows = engineered[(engineered["MarkDown1"] == 0.0) & (engineered["MarkDown1_was_missing"] == 0)]
    assert len(observed_zero_rows) >= 1


# ---------------------------------------------------------------------------
# Week-over-week: no inf%, calendar-correct 7-day adjacency only.
# ---------------------------------------------------------------------------


def test_wow_zero_previous_is_not_inf_percent():
    dates = pd.date_range("2010-01-01", periods=3, freq="7D")
    df = _with_mixed_holiday(
        pd.DataFrame(
            {
                "Store": 1,
                "Dept": 1,
                "Date": dates,
                "Weekly_Sales": [0.0, 6.0, -498.0],
                "IsHoliday": False,
            }
        )
    )
    findings = compute_micro_inspection(df)
    anomalies = findings["date_level_anomalies"]

    # No inf/-inf may appear anywhere in the pct-based results.
    for row in anomalies["top_by_pct_change"]:
        assert np.isfinite(row["pct_change"])

    zero_prev_dates = {row["Date"] for row in anomalies["zero_previous_transitions"]}
    assert str(dates[1].date()) in zero_prev_dates
    zero_prev_row = next(r for r in anomalies["zero_previous_transitions"] if r["Date"] == str(dates[1].date()))
    assert zero_prev_row["prev_sales"] == 0.0
    assert zero_prev_row["abs_change"] == 6.0
    assert "pct_change" not in zero_prev_row or zero_prev_row.get("pct_change") is None


def test_wow_non_consecutive_week_gap_is_excluded_not_mislabeled():
    df = _with_mixed_holiday(
        pd.DataFrame(
            {
                "Store": 1,
                "Dept": 1,
                "Date": [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-03-01")],  # ~8 weeks apart
                "Weekly_Sales": [100.0, 500.0],
                "IsHoliday": False,
            }
        )
    )
    findings = compute_micro_inspection(df)
    anomalies = findings["date_level_anomalies"]

    all_dates = {r["Date"] for r in anomalies["top_by_pct_change"]} | {
        r["Date"] for r in anomalies["zero_previous_transitions"]
    }
    assert str(pd.Timestamp("2010-03-01").date()) not in all_dates
    assert anomalies["excluded_non_consecutive_week_pairs"] == 1


def test_wow_true_7_day_gap_is_included():
    dates = [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-01-08")]
    df = _with_mixed_holiday(
        pd.DataFrame(
            {
                "Store": 1,
                "Dept": 1,
                "Date": dates,
                "Weekly_Sales": [100.0, 150.0],
                "IsHoliday": False,
            }
        )
    )
    findings = compute_micro_inspection(df)
    anomalies = findings["date_level_anomalies"]

    matched = [r for r in anomalies["top_by_pct_change"] if r["Date"] == str(dates[1].date())]
    assert len(matched) == 1
    assert matched[0]["pct_change"] == 0.5
