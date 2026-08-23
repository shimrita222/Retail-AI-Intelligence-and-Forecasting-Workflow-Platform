"""Data Analyst Crew: Data Profiler, Business Intelligence, and Data Contract
Architect agents, backed entirely by deterministic Python analysis.

Guardrail: the LLM agents never compute statistics themselves. Every number
that appears in insights.md, eda_report.html, or dataset_contract.json is
produced by the pure-pandas functions below; the agents are only given those
precomputed numbers as task context and asked to narrate/explain them. This
keeps validation-relevant facts (contract bounds, null counts, anomaly
counts) reproducible and independent of LLM sampling.

Micro-inspection policy (Karpathy-style "don't trust the aggregate"):
the Data Profiler and Business Intelligence agents are fed record-level
findings -- explicit Store/Dept/Date rows for zero/negative sales, holiday
spikes per store, and extreme markdown outliers -- not just global means.
Those record-level findings are always written into insights.md and
eda_report.html verbatim, whether or not the LLM narrative mentions them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

MICRO_INSPECTION_SAMPLE_SIZE = 15
HOLIDAY_SPIKE_TOP_N = 10
ANOMALY_TOP_N = 15


# ---------------------------------------------------------------------------
# Deterministic analysis (pure pandas/numpy -- no LLM involved)
# ---------------------------------------------------------------------------


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Deterministic data-profiling summary: missingness, cardinality, dtypes."""
    null_counts = df.isna().sum()
    null_pct = (null_counts / max(len(df), 1) * 100).round(3)

    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "date_range": {
            "min": str(pd.to_datetime(df["Date"]).min().date()) if "Date" in df.columns else None,
            "max": str(pd.to_datetime(df["Date"]).max().date()) if "Date" in df.columns else None,
        },
        "store_count": int(df["Store"].nunique()) if "Store" in df.columns else None,
        "dept_count": int(df["Dept"].nunique()) if "Dept" in df.columns else None,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(v) for c, v in null_counts.items() if v > 0},
        "null_pct": {c: float(v) for c, v in null_pct.items() if v > 0},
        "dept_breakdown_top10": (
            df["Dept"].value_counts().head(10).to_dict() if "Dept" in df.columns else {}
        ),
    }


def compute_micro_inspection(df: pd.DataFrame) -> dict[str, Any]:
    """Record-level edge-case inspection: zero/negative sales, holiday spikes
    per individual store, extreme markdown outliers, and date-level anomalies.
    Every finding here is tied to explicit (Store, Dept, Date) rows, not an
    aggregate statistic.
    """
    findings: dict[str, Any] = {}

    # 1. Zero and negative sales instances (record-level)
    zero_sales = df[df["Weekly_Sales"] == 0]
    negative_sales = df[df["Weekly_Sales"] < 0]
    findings["zero_sales"] = {
        "count": int(len(zero_sales)),
        "sample_rows": zero_sales[["Store", "Dept", "Date", "Weekly_Sales"]]
        .head(MICRO_INSPECTION_SAMPLE_SIZE)
        .astype({"Date": str})
        .to_dict(orient="records"),
    }
    findings["negative_sales"] = {
        "count": int(len(negative_sales)),
        "sample_rows": negative_sales[["Store", "Dept", "Date", "Weekly_Sales"]]
        .sort_values("Weekly_Sales")
        .head(MICRO_INSPECTION_SAMPLE_SIZE)
        .astype({"Date": str})
        .to_dict(orient="records"),
    }

    # 2. Holiday-specific spikes per individual store (not a single global average)
    if "IsHoliday" in df.columns:
        per_store = (
            df.groupby(["Store", "IsHoliday"])["Weekly_Sales"]
            .mean()
            .unstack("IsHoliday")
            .rename(columns={True: "holiday_mean", False: "non_holiday_mean"})
        )
        per_store = per_store.dropna()
        per_store["spike_ratio"] = per_store["holiday_mean"] / per_store["non_holiday_mean"].replace(0, pd.NA)
        per_store = per_store.dropna().sort_values("spike_ratio", ascending=False)
        top_spikes = per_store.head(HOLIDAY_SPIKE_TOP_N).reset_index()
        findings["holiday_spikes_per_store"] = top_spikes.round(3).to_dict(orient="records")
    else:
        findings["holiday_spikes_per_store"] = []

    # 3. Extreme markdown outliers (IQR method per markdown column).
    # MarkDown1-5 are Store+Date-level features (see join_keys.features_to_train
    # in data/dataset_manifest.json); after the merge into the Store+Dept+Date
    # sales grain, one Store-Date markdown value is repeated once per
    # department present that week. Deduplicating to (Store, Date) here before
    # computing quantiles/outliers prevents a single markdown observation from
    # being counted or reported as N independent department-level events.
    markdown_outliers: list[dict[str, Any]] = []
    for col in ("MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"):
        if col not in df.columns:
            continue
        store_date_md = df[["Store", "Date", col]].drop_duplicates(subset=["Store", "Date"])
        series = store_date_md[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        outliers = store_date_md[store_date_md[col] > upper_bound].sort_values(col, ascending=False)
        for _, row in outliers.head(5).iterrows():
            markdown_outliers.append(
                {
                    "column": col,
                    "Store": int(row["Store"]),
                    "Date": str(row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]),
                    "value": float(row[col]),
                    "upper_bound": round(float(upper_bound), 2),
                    "note": "Store-Date level markdown observation; applies to every department "
                    "recorded for this store/date, not a single department-specific event.",
                }
            )
    findings["extreme_markdown_outliers"] = markdown_outliers[:ANOMALY_TOP_N]

    # 4. Date-level anomalies: extreme week-over-week swings within a Store/Dept
    # series. Only rows whose date is exactly 7 days after that same series'
    # previous observation are genuine week-over-week comparisons; a larger
    # gap is a real temporal gap in the series, not a one-week change, and is
    # excluded rather than mislabeled as "week-over-week". When the previous
    # week's sales were exactly zero, a percentage change is mathematically
    # undefined (division by zero); it is reported separately with an
    # explicit reason instead of as inf%/-inf%.
    df_sorted = df.sort_values(["Store", "Dept", "Date"]).copy()
    grouped = df_sorted.groupby(["Store", "Dept"])
    df_sorted["prev_sales"] = grouped["Weekly_Sales"].shift(1)
    df_sorted["prev_date"] = grouped["Date"].shift(1)
    df_sorted["days_since_prev"] = (df_sorted["Date"] - df_sorted["prev_date"]).dt.days
    df_sorted["is_consecutive_week"] = df_sorted["days_since_prev"] == 7

    weekly_pairs = df_sorted[df_sorted["is_consecutive_week"]].copy()
    weekly_pairs["abs_change"] = weekly_pairs["Weekly_Sales"] - weekly_pairs["prev_sales"]
    weekly_pairs["pct_change"] = weekly_pairs["abs_change"] / weekly_pairs["prev_sales"].abs().replace(0, pd.NA)

    pct_based = weekly_pairs.dropna(subset=["pct_change"]).copy()
    pct_based = pct_based.reindex(pct_based["pct_change"].abs().sort_values(ascending=False).index)
    top_pct = pct_based.head(ANOMALY_TOP_N)[
        ["Store", "Dept", "Date", "prev_sales", "Weekly_Sales", "pct_change"]
    ].copy()
    top_pct["Date"] = top_pct["Date"].astype(str)

    zero_prev = weekly_pairs[weekly_pairs["prev_sales"] == 0].copy()
    zero_prev = zero_prev.reindex(zero_prev["abs_change"].abs().sort_values(ascending=False).index)
    top_zero_prev = zero_prev.head(ANOMALY_TOP_N)[
        ["Store", "Dept", "Date", "prev_sales", "Weekly_Sales", "abs_change"]
    ].copy()
    top_zero_prev["Date"] = top_zero_prev["Date"].astype(str)
    top_zero_prev["reason"] = "previous value is zero; percentage change is undefined"

    non_consecutive_with_history = (~df_sorted["is_consecutive_week"]) & df_sorted["prev_sales"].notna()

    findings["date_level_anomalies"] = {
        "top_by_pct_change": top_pct.round(3).to_dict(orient="records"),
        "zero_previous_transitions": top_zero_prev.round(3).to_dict(orient="records"),
        "excluded_non_consecutive_week_pairs": int(non_consecutive_with_history.sum()),
        "note": (
            "Only Store/Dept observations exactly 7 days apart are compared as week-over-week; "
            "pairs separated by a larger calendar gap are excluded from these anomaly lists rather "
            "than mislabeled as a one-week change."
        ),
    }

    return findings


def compute_business_intelligence(df: pd.DataFrame, micro_inspection: dict[str, Any]) -> dict[str, Any]:
    """Trend patterns, holiday impact, and markdown effects, grounded in the
    record-level micro-inspection findings rather than a single global mean.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    monthly_trend = df.groupby("Month")["Weekly_Sales"].mean().round(2).to_dict()

    holiday_overall = df.groupby("IsHoliday")["Weekly_Sales"].mean() if "IsHoliday" in df.columns else None
    holiday_impact = {
        "holiday_mean": round(float(holiday_overall.get(True, float("nan"))), 2) if holiday_overall is not None else None,
        "non_holiday_mean": round(float(holiday_overall.get(False, float("nan"))), 2) if holiday_overall is not None else None,
        "stores_with_largest_individual_spikes": micro_inspection.get("holiday_spikes_per_store", [])[:5],
    }

    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    markdown_effect: dict[str, Any] = {}
    if markdown_cols:
        df["has_any_markdown"] = df[markdown_cols].notna().any(axis=1)
        with_md = df.loc[df["has_any_markdown"], "Weekly_Sales"].mean()
        without_md = df.loc[~df["has_any_markdown"], "Weekly_Sales"].mean()
        markdown_effect = {
            "mean_sales_weeks_with_any_markdown": round(float(with_md), 2) if pd.notna(with_md) else None,
            "mean_sales_weeks_without_markdown": round(float(without_md), 2) if pd.notna(without_md) else None,
            "extreme_outlier_count": len(micro_inspection.get("extreme_markdown_outliers", [])),
            "grain_note": (
                "MarkDown1-5 are recorded at Store+Date grain; Weekly_Sales is recorded at "
                "Store+Dept+Date grain. This comparison groups Store+Dept+Date sales rows by "
                "whether their Store+Date shared any recorded markdown value, so it reflects sales "
                "in weeks with recorded markdown information, not a per-department markdown event. "
                "A higher mean here is an observed association, not evidence that markdowns caused "
                "the difference."
            ),
        }

    return {
        "monthly_trend": monthly_trend,
        "holiday_impact": holiday_impact,
        "markdown_effect": markdown_effect,
    }


def build_dataset_contract(df: pd.DataFrame, profiling: dict[str, Any]) -> dict[str, Any]:
    """Deterministically formulate the structural contract that
    contract_validator.py will enforce. Bounds are derived from the observed
    data with a safety margin, never invented by an LLM.
    """
    target = df["Weekly_Sales"].dropna()
    observed_min, observed_max = float(target.min()), float(target.max())
    min_bound = min(-5000.0, round(observed_min * 1.1, 2)) if observed_min < 0 else -5000.0
    max_bound = max(800000.0, round(observed_max * 1.1, 2))

    return {
        "primary_keys": ["Store", "Dept", "Date"],
        "target_column": "Weekly_Sales",
        "columns": {
            "Store": {"dtype": "int", "nullable": False},
            "Dept": {"dtype": "int", "nullable": False},
            "Date": {"dtype": "datetime", "nullable": False},
            "Weekly_Sales": {
                "dtype": "float",
                "nullable": False,
                "min": min_bound,
                "max": max_bound,
            },
            "IsHoliday": {"dtype": "bool", "nullable": False},
            "Type": {"dtype": "string", "nullable": True},
            "Size": {"dtype": "int", "nullable": True},
            "Temperature": {"dtype": "float", "nullable": True},
            "Fuel_Price": {"dtype": "float", "nullable": True},
            "CPI": {"dtype": "float", "nullable": True},
            "Unemployment": {"dtype": "float", "nullable": True},
            "MarkDown1": {"dtype": "float", "nullable": True},
            "MarkDown2": {"dtype": "float", "nullable": True},
            "MarkDown3": {"dtype": "float", "nullable": True},
            "MarkDown4": {"dtype": "float", "nullable": True},
            "MarkDown5": {"dtype": "float", "nullable": True},
        },
        "generated_from_row_count": profiling["row_count"],
        "generation_method": "deterministic (src/agents/analyst_crew.py:build_dataset_contract)",
    }


# ---------------------------------------------------------------------------
# Report rendering (deterministic HTML/Markdown, guarantees micro-inspection
# findings are present regardless of what the LLM narrative says)
# ---------------------------------------------------------------------------


def _records_to_html_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<p><em>No records found.</em></p>"
    df = pd.DataFrame(records)
    return df.to_html(index=False, classes="micro-table", border=0)


def _format_currency(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "N/A"


def _holiday_impact_html(holiday_impact: dict[str, Any]) -> str:
    holiday_mean = holiday_impact.get("holiday_mean")
    non_holiday_mean = holiday_impact.get("non_holiday_mean")
    spikes = holiday_impact.get("stores_with_largest_individual_spikes", [])

    kpis = (
        f"<div class=\"kpi\"><strong>Holiday-week mean sales:</strong> {_format_currency(holiday_mean)}</div>"
        f"<div class=\"kpi\"><strong>Non-holiday-week mean sales:</strong> {_format_currency(non_holiday_mean)}</div>"
    )

    if spikes:
        spikes_df = pd.DataFrame(spikes).rename(
            columns={
                "Store": "Store",
                "non_holiday_mean": "Non-Holiday Mean",
                "holiday_mean": "Holiday Mean",
                "spike_ratio": "Spike Ratio",
            }
        )
        spikes_df = spikes_df[["Store", "Non-Holiday Mean", "Holiday Mean", "Spike Ratio"]]
        table = spikes_df.to_html(index=False, classes="micro-table", border=0)
    else:
        table = "<p><em>No holiday spike data available.</em></p>"

    return f"{kpis}<h4>Top individual store holiday spikes</h4>{table}"


def _markdown_effect_html(markdown_effect: dict[str, Any]) -> str:
    if not markdown_effect:
        return "<p><em>No markdown columns present in this dataset.</em></p>"

    with_md = markdown_effect.get("mean_sales_weeks_with_any_markdown")
    without_md = markdown_effect.get("mean_sales_weeks_without_markdown")
    outlier_count = markdown_effect.get("extreme_outlier_count", 0)

    grain_note = markdown_effect.get("grain_note", "")

    return (
        f"<div class=\"kpi\"><strong>Mean sales (with recorded markdown):</strong> {_format_currency(with_md)}</div>"
        f"<div class=\"kpi\"><strong>Mean sales (no recorded markdown):</strong> {_format_currency(without_md)}</div>"
        f"<div class=\"kpi\"><strong>Extreme markdown outliers (Store-Date level):</strong> {outlier_count}</div>"
        f"<p><em>{grain_note}</em></p>"
    )


def render_eda_report_html(
    profiling: dict[str, Any],
    micro_inspection: dict[str, Any],
    business_intelligence: dict[str, Any],
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Retail EDA Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1, h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
  table.micro-table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  table.micro-table th, table.micro-table td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 0.85rem; }}
  table.micro-table th {{ background: #f2f2f2; }}
  .kpi {{ display: inline-block; margin: 0.5rem 1rem 0.5rem 0; padding: 0.75rem 1rem; background: #f7f7f7; border-radius: 6px; }}
  .section {{ margin-bottom: 2.5rem; }}
</style>
</head>
<body>
<h1>Retail Data Analytics &mdash; EDA Report</h1>

<div class="section">
  <h2>Dataset Profile</h2>
  <div class="kpi"><strong>Rows:</strong> {profiling['row_count']:,}</div>
  <div class="kpi"><strong>Stores:</strong> {profiling['store_count']}</div>
  <div class="kpi"><strong>Departments:</strong> {profiling['dept_count']}</div>
  <div class="kpi"><strong>Date range:</strong> {profiling['date_range']['min']} to {profiling['date_range']['max']}</div>
  <h3>Null counts by column</h3>
  {pd.DataFrame(list(profiling['null_counts'].items()), columns=['Column', 'Nulls']).to_html(index=False, classes='micro-table', border=0) if profiling['null_counts'] else '<p>No nulls detected.</p>'}
</div>

<div class="section">
  <h2>Micro-Inspection Findings (record-level)</h2>
  <p>These findings inspect individual (Store, Dept, Date) records rather than relying only on global aggregates.</p>

  <h3>Zero-sales instances ({micro_inspection['zero_sales']['count']} total)</h3>
  {_records_to_html_table(micro_inspection['zero_sales']['sample_rows'])}

  <h3>Negative-sales instances ({micro_inspection['negative_sales']['count']} total)</h3>
  {_records_to_html_table(micro_inspection['negative_sales']['sample_rows'])}

  <h3>Top holiday sales spikes, per individual store</h3>
  {_records_to_html_table(micro_inspection['holiday_spikes_per_store'])}

  <h3>Extreme markdown outliers (IQR method, Store-Date level -- deduplicated across departments)</h3>
  {_records_to_html_table(micro_inspection['extreme_markdown_outliers'])}

  <h3>Largest week-over-week sales anomalies, per Store/Dept (true 7-day-gap pairs only)</h3>
  <p><em>{micro_inspection['date_level_anomalies'].get('note', '')} Excluded non-consecutive-week pairs: {micro_inspection['date_level_anomalies'].get('excluded_non_consecutive_week_pairs', 0)}.</em></p>
  {_records_to_html_table(micro_inspection['date_level_anomalies'].get('top_by_pct_change', []))}
  <h4>Previous-week-zero transitions (percentage change undefined)</h4>
  {_records_to_html_table(micro_inspection['date_level_anomalies'].get('zero_previous_transitions', []))}
</div>

<div class="section">
  <h2>Business Intelligence Summary</h2>
  <h3>Monthly sales trend</h3>
  {pd.DataFrame(list(business_intelligence['monthly_trend'].items()), columns=['Month', 'Mean Weekly Sales']).to_html(index=False, classes='micro-table', border=0)}
  <h3>Holiday impact</h3>
  {_holiday_impact_html(business_intelligence['holiday_impact'])}
  <h3>Markdown effect</h3>
  {_markdown_effect_html(business_intelligence['markdown_effect'])}
</div>

</body>
</html>
"""


def render_insights_markdown(
    profiling: dict[str, Any],
    micro_inspection: dict[str, Any],
    business_intelligence: dict[str, Any],
    contract: dict[str, Any],
    narrative_sections: dict[str, str] | None = None,
) -> str:
    narrative_sections = narrative_sections or {}
    lines: list[str] = []
    lines.append("# Data Analyst Crew — Insights\n")

    lines.append("## Data Profile\n")
    lines.append(
        f"- **{profiling['row_count']:,} rows**, {profiling['store_count']} stores, "
        f"{profiling['dept_count']} departments, date range "
        f"{profiling['date_range']['min']} to {profiling['date_range']['max']}."
    )
    if profiling["null_counts"]:
        lines.append("- Null counts: " + ", ".join(f"{c}={n}" for c, n in profiling["null_counts"].items()))
    if narrative_sections.get("data_profiler"):
        lines.append("\n" + narrative_sections["data_profiler"])

    lines.append("\n## Micro-Inspection Findings (record-level, not aggregate)\n")
    lines.append(
        f"- **Zero-sales records:** {micro_inspection['zero_sales']['count']} "
        f"(sample: {micro_inspection['zero_sales']['sample_rows'][:3]})"
    )
    lines.append(
        f"- **Negative-sales records:** {micro_inspection['negative_sales']['count']} "
        f"(sample: {micro_inspection['negative_sales']['sample_rows'][:3]})"
    )
    lines.append("- **Top holiday spikes per individual store:**")
    for row in micro_inspection["holiday_spikes_per_store"][:5]:
        lines.append(
            f"  - Store {row['Store']}: holiday mean {row['holiday_mean']:.2f} vs "
            f"non-holiday mean {row['non_holiday_mean']:.2f} (ratio {row['spike_ratio']:.2f}x)"
        )
    lines.append(
        "- **Extreme markdown outliers (Store-Date level; MarkDown1-5 are recorded per "
        "Store+Date, not per department, so each entry below applies to every department "
        "recorded for that store/date, not a single department):**"
    )
    for row in micro_inspection["extreme_markdown_outliers"][:5]:
        lines.append(
            f"  - {row['column']} = {row['value']:.2f} at Store {row['Store']}, "
            f"{row['Date']} (upper bound {row['upper_bound']:.2f})"
        )
    date_anomalies = micro_inspection.get("date_level_anomalies", {})
    lines.append(
        "- **Largest week-over-week anomalies** (only Store/Dept pairs exactly 7 days apart; "
        f"{date_anomalies.get('excluded_non_consecutive_week_pairs', 0)} non-consecutive-week pairs "
        "excluded rather than mislabeled):"
    )
    for row in date_anomalies.get("top_by_pct_change", [])[:5]:
        lines.append(
            f"  - Store {row['Store']}, Dept {row['Dept']}, {row['Date']}: "
            f"{row['prev_sales']:.2f} -> {row['Weekly_Sales']:.2f} ({row['pct_change']*100:.1f}%)"
        )
    zero_prev_rows = date_anomalies.get("zero_previous_transitions", [])[:5]
    if zero_prev_rows:
        lines.append(
            "- **Previous-week-zero transitions** (percentage change is undefined when the prior "
            "week's sales were zero; reported as an absolute change, not as inf%/-inf%):"
        )
        for row in zero_prev_rows:
            lines.append(
                f"  - Store {row['Store']}, Dept {row['Dept']}, {row['Date']}: "
                f"{row['prev_sales']:.2f} -> {row['Weekly_Sales']:.2f} "
                f"(absolute change {row['abs_change']:.2f}, percentage change N/A)"
            )
    lines.append("\n## Business Intelligence\n")
    holiday_impact = business_intelligence.get("holiday_impact", {})
    lines.append(
        f"- Holiday-week mean sales: {holiday_impact.get('holiday_mean')} vs "
        f"non-holiday mean: {holiday_impact.get('non_holiday_mean')}."
    )
    markdown_effect = business_intelligence.get("markdown_effect", {})
    if markdown_effect:
        lines.append(
            f"- Mean sales in weeks with recorded markdown information: "
            f"{markdown_effect.get('mean_sales_weeks_with_any_markdown')} vs weeks with no recorded "
            f"markdown data: {markdown_effect.get('mean_sales_weeks_without_markdown')} "
            f"({markdown_effect.get('extreme_outlier_count')} extreme Store-Date markdown outliers "
            "detected). This reflects data availability, not confirmed promotional activity; "
            f"{markdown_effect.get('grain_note', '')}"
        )
    trend = business_intelligence.get("monthly_trend", {})
    if trend:
        first_month, last_month = next(iter(trend)), list(trend)[-1]
        lines.append(
            f"- Monthly trend spans {first_month} ({trend[first_month]:.2f}) to "
            f"{last_month} ({trend[last_month]:.2f})."
        )
    if narrative_sections.get("business_intelligence"):
        lines.append("\n" + narrative_sections["business_intelligence"])

    lines.append("\n## Data Contract\n")
    lines.append(
        f"- Primary keys: {contract['primary_keys']}; target: `{contract['target_column']}` "
        f"bounded to [{contract['columns']['Weekly_Sales']['min']}, {contract['columns']['Weekly_Sales']['max']}]."
    )
    if narrative_sections.get("data_contract_architect"):
        lines.append("\n" + narrative_sections["data_contract_architect"])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CrewAI wiring: agents narrate the precomputed deterministic findings
# ---------------------------------------------------------------------------


def _build_crew(profiling: dict[str, Any], micro_inspection: dict[str, Any], business_intelligence: dict[str, Any], contract: dict[str, Any]):
    from crewai import Agent, Crew, Process, Task

    data_profiler = Agent(
        role="Data Profiler",
        goal=(
            "Explain the dataset's structural profile and record-level data-quality issues "
            "(missingness, zero/negative sales, anomalies) in plain business language."
        ),
        backstory=(
            "A meticulous data quality analyst who never trusts a global average and always "
            "checks individual records for edge cases before declaring data usable."
        ),
        allow_delegation=False,
        verbose=False,
    )

    business_intel_agent = Agent(
        role="Business Intelligence Analyst",
        goal=(
            "Translate sales trends, holiday effects, and markdown effects -- grounded in "
            "specific store/date evidence, not just averages -- into actionable business insight."
        ),
        backstory=(
            "A retail BI specialist who always backs up trend claims with the specific stores "
            "and dates that drive them."
        ),
        allow_delegation=False,
        verbose=False,
    )

    contract_architect = Agent(
        role="Data Contract Architect",
        goal="Explain why the structural validation rules in dataset_contract.json protect downstream modeling.",
        backstory="A pragmatic data engineer who turns profiling findings into enforceable, minimal contracts.",
        allow_delegation=False,
        verbose=False,
    )

    EVIDENCE_GUARDRAIL = (
        "Evidence discipline (do not violate): the source documentation does not state that negative "
        "Weekly_Sales values represent returns, refunds, corrections, or any other specific business "
        "process -- describe them only as 'negative sales observations with undocumented business "
        "meaning', never assert a cause. The source documentation does not state that a missing "
        "MarkDown value means no promotion occurred -- describe missing MarkDown data only as "
        "'markdown information not recorded/unavailable', never as 'no promotion'. Any markdown-vs-sales "
        "comparison is an observed association across different grains (MarkDown is Store+Date level, "
        "Weekly_Sales is Store+Dept+Date level) -- never state or imply that markdowns caused a sales "
        "difference. Do not fabricate statistics, metrics, or business impact beyond the numbers given."
    )

    profiler_task = Task(
        description=(
            "Using this deterministic profiling and micro-inspection JSON (already computed in Python, "
            "do not recompute or alter any numbers), write a short markdown narrative (5-8 sentences) "
            "summarizing data quality, explicitly calling out the zero/negative sales counts and the "
            f"single most notable week-over-week anomaly.\n\n{EVIDENCE_GUARDRAIL}\n\nPROFILING:\n"
            f"{json.dumps(profiling, indent=2)}\n\nMICRO_INSPECTION:\n{json.dumps(micro_inspection, indent=2)}"
        ),
        expected_output="A short markdown narrative referencing the exact figures given above.",
        agent=data_profiler,
    )

    bi_task = Task(
        description=(
            "Using this deterministic business-intelligence and micro-inspection JSON (already computed "
            "in Python, do not recompute or alter any numbers), write a short markdown narrative (5-8 "
            "sentences) covering monthly trend direction, the holiday impact, and the markdown effect, "
            f"citing at least one specific store from the holiday-spike findings.\n\n{EVIDENCE_GUARDRAIL}"
            f"\n\nBUSINESS_INTELLIGENCE:\n"
            f"{json.dumps(business_intelligence, indent=2)}\n\nMICRO_INSPECTION:\n{json.dumps(micro_inspection, indent=2)}"
        ),
        expected_output="A short markdown narrative referencing the exact figures given above.",
        agent=business_intel_agent,
    )

    contract_task = Task(
        description=(
            "This dataset_contract.json was already generated deterministically in Python from the "
            "profiling output; do not propose different bounds or rules. Write a short markdown "
            "explanation (4-6 sentences) of why these particular rules (primary key uniqueness, "
            "non-null target, numeric bounds) are the right minimal gate for this pipeline.\n\nCONTRACT:\n"
            f"{json.dumps(contract, indent=2)}"
        ),
        expected_output="A short markdown explanation of the contract rules.",
        agent=contract_architect,
    )

    crew = Crew(
        agents=[data_profiler, business_intel_agent, contract_architect],
        tasks=[profiler_task, bi_task, contract_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew, (profiler_task, bi_task, contract_task)


def run_analyst_crew(clean_data_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Full Data Analyst Crew stage: deterministic analysis + LLM narration,
    writing clean_data.csv (already produced by ingestion), eda_report.html,
    insights.md, and dataset_contract.json into run_dir.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(clean_data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    profiling = profile_dataset(df)
    micro_inspection = compute_micro_inspection(df)
    business_intelligence = compute_business_intelligence(df, micro_inspection)
    contract = build_dataset_contract(df, profiling)

    contract_path = run_dir / "dataset_contract.json"
    with contract_path.open("w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    narrative_sections: dict[str, str] = {}
    try:
        crew, (profiler_task, bi_task, contract_task) = _build_crew(
            profiling, micro_inspection, business_intelligence, contract
        )
        crew.kickoff()
        narrative_sections["data_profiler"] = str(profiler_task.output)
        narrative_sections["business_intelligence"] = str(bi_task.output)
        narrative_sections["data_contract_architect"] = str(contract_task.output)
    except Exception as exc:  # noqa: BLE001 - LLM/network calls can fail for many reasons
        narrative_sections["error"] = (
            f"LLM narration unavailable ({exc.__class__.__name__}: {exc}); "
            "insights.md contains deterministic findings only."
        )

    eda_html = render_eda_report_html(profiling, micro_inspection, business_intelligence)
    eda_path = run_dir / "eda_report.html"
    eda_path.write_text(eda_html, encoding="utf-8")

    insights_md = render_insights_markdown(
        profiling, micro_inspection, business_intelligence, contract, narrative_sections
    )
    insights_path = run_dir / "insights.md"
    insights_path.write_text(insights_md, encoding="utf-8")

    return {
        "profiling": profiling,
        "micro_inspection": micro_inspection,
        "business_intelligence": business_intelligence,
        "contract": contract,
        "contract_path": contract_path,
        "eda_report_path": eda_path,
        "insights_path": insights_path,
    }
