"""Explicit, auditable modeling-eligibility policy.

This module is the single source of truth for the difference between
"included in descriptive/EDA data" and "included in the predictive
modeling population". Every department is descriptively included (clean_data.csv,
insights.md, eda_report.html are never filtered by this module); a department
only disappears from the population handed to feature engineering / model
training if it has an entry in MODELING_EXCLUSIONS below.

This is a modeling-eligibility decision, not a data-cleaning decision: an
excluded department's rows are not invalid, corrupted, or deleted -- they are
simply held out of the predictive training population pending resolution of
the open questions recorded in `reason`/`evidence`. See the Department 47 vs
Department 78 audit investigations (2026-08-17) for the full evidence trail;
`evidence` below summarizes the load-bearing findings only.

Do not add a department here based on surface traits alone (round numbers,
symmetric +/-X values, mere presence of negative sales) -- see Department 78,
which shares some of those traits with Department 47 but is NOT excluded,
because its calendar-corrected weekly behavior is much closer to an ordinary
department's (see DOCUMENTED_WATCHLIST).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ModelingExclusion:
    dept: int
    reason: str
    evidence: list[str]
    decision_type: str = "modeling_eligibility"  # never "data_cleaning"
    descriptive_status: str = "included"
    modeling_status: str = "excluded"


MODELING_EXCLUSIONS: dict[int, ModelingExclusion] = {
    47: ModelingExclusion(
        dept=47,
        reason=(
            "Unresolved business semantics (no authoritative source or project documentation "
            "explains what Department 47 represents) combined with verified structural target "
            "behavior materially different from the ordinary department population."
        ),
        evidence=[
            "39.32% negative-sales rate vs ~0.16% for the ordinary-department baseline (>100x)",
            "Anomaly present across 25/37 stores carrying the department -- broad, not a few outlier stores",
            "Calendar-corrected (true 7-day-gap) week-over-week transitions: 32.26% Positive->Positive "
            "vs ~99.7% for ordinary departments",
            "Calendar-corrected Negative->Negative rate 14.52% vs ~0.01% baseline "
            "(negative weeks frequently persist into the next true week)",
            "Department mean Weekly_Sales is negative (-7.68); every comparison department/baseline "
            "has a strongly positive mean",
        ],
    ),
}

# Departments with elevated-but-non-exclusionary anomaly rates. This dict does
# NOT affect modeling_eligible_mask() -- it exists purely so the evidence is
# documented and traceable in model_card.md. A department listed here remains
# fully modeling-eligible.
DOCUMENTED_WATCHLIST: dict[int, str] = {
    78: (
        "Elevated anomaly rate / unusual discrete behavior; retained for modeling with "
        "documentation and monitoring. Calendar-corrected weekly behavior (67.24% "
        "Positive->Positive, 0% Negative->Negative) is close to ordinary-department stability and "
        "materially different from Department 47 (32.26% Positive->Positive, 14.52% "
        "Negative->Negative). Negative-sales rate 14.04% vs Department 47's 39.32%; 51% of its "
        "stores have zero negative observations; per-store sample sizes are small (median ~6 "
        "observations/store), so raw percentages should be read cautiously."
    ),
    18: "Mildly elevated negative-sales rate (~3.58%) vs baseline; retained for modeling, no exclusion evidence.",
    54: "Mildly elevated negative-sales rate (~3.06%) vs baseline; retained for modeling, no exclusion evidence.",
}


def get_exclusions() -> dict[int, ModelingExclusion]:
    return dict(MODELING_EXCLUSIONS)


def modeling_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows eligible for the predictive modeling population."""
    if "Dept" not in df.columns or not MODELING_EXCLUSIONS:
        return pd.Series(True, index=df.index)
    return ~df["Dept"].isin(MODELING_EXCLUSIONS.keys())


def summarize_modeling_population(df: pd.DataFrame) -> dict[str, object]:
    """Traceable before/after summary for run/model metadata."""
    mask = modeling_eligible_mask(df)
    excluded_df = df[~mask]
    excluded_depts = sorted(int(d) for d in excluded_df["Dept"].unique()) if "Dept" in df.columns else []
    return {
        "total_clean_rows": int(len(df)),
        "eligible_rows": int(mask.sum()),
        "excluded_rows": int((~mask).sum()),
        "excluded_departments": excluded_depts,
        "exclusion_reasons": {d: MODELING_EXCLUSIONS[d].reason for d in excluded_depts},
        "watchlist": dict(DOCUMENTED_WATCHLIST),
    }
