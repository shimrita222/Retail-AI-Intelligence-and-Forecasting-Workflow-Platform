"""Streamlit multi-role dashboard for the Retail AI Intelligence &
Forecasting Workflow Platform.

Pages (shown based on the logged-in user's role permissions from config.yaml):
  - Admin: trigger a new CrewAI Flow run, watch live logs, inspect past
    runs and validation-gate failures.
  - Descriptive Analytics ("What Has Happened?"): KPIs, sales trend,
    markdown correlation heatmap, embedded eda_report.html / insights.md.
  - Predictive Intelligence ("What Will Happen?"): candidate model
    comparison, selected model metrics, forecast visualizer, model_card.md.

All numbers shown come straight from files written by the deterministic
services/crews (clean_data.csv, evaluation_report.json, dataset_contract.json,
selected_model.joblib) -- the UI never recomputes statistics itself.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.analyst_crew import compute_business_intelligence, compute_micro_inspection
from src.services.feature_pipeline import engineer_features, get_feature_columns
from src.services.ml_trainer import chronological_split
from src.services.run_registry import is_real_run_id
from src.utils.auth import (
    current_user,
    has_permission,
    load_config,
    render_logout_button,
)

st.set_page_config(page_title="Retail AI Intelligence Platform", layout="wide")

CONFIG = load_config()
ARTIFACTS_DIR = Path(CONFIG["paths"]["artifacts_dir"])
RAW_DIR = Path(CONFIG["paths"]["data_raw_dir"])


# ---------------------------------------------------------------------------
# Login gate (done here, not in auth.py, so app.py owns page flow/config)
# ---------------------------------------------------------------------------

from src.utils.auth import render_login_form  # noqa: E402

if not render_login_form(CONFIG):
    st.stop()

render_logout_button()


# ---------------------------------------------------------------------------
# Background workflow execution (Admin "run the flow" control)
# ---------------------------------------------------------------------------


class _ListLogHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(self.format(record))


def _execute_flow(run_id: str, entry: dict[str, Any]) -> None:
    from src.flows.retail_flow import RetailFlow

    handler = _ListLogHandler(entry["logs"])
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    try:
        entry["logs"].append(f"Starting run {run_id}...")
        flow = RetailFlow(raw_dir=RAW_DIR, artifacts_root=ARTIFACTS_DIR, run_id=run_id)
        flow.kickoff()
        entry["state"] = flow.state.model_dump()
        entry["status"] = flow.state.status
        entry["logs"].append(f"Run finished with status: {flow.state.status}")
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI instead of crashing the thread silently
        entry["status"] = "ERROR"
        entry["error"] = f"{exc.__class__.__name__}: {exc}"
        entry["logs"].append(f"FATAL: {exc.__class__.__name__}: {exc}")
    finally:
        root_logger.removeHandler(handler)
        entry["running"] = False


def start_new_run() -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    entry: dict[str, Any] = {"logs": [], "status": "RUNNING", "running": True, "error": None, "state": None}
    st.session_state.run_registry[run_id] = entry
    thread = threading.Thread(target=_execute_flow, args=(run_id, entry), daemon=True)
    thread.start()
    st.session_state.active_run_id = run_id
    return run_id


if "run_registry" not in st.session_state:
    st.session_state.run_registry = {}
if "active_run_id" not in st.session_state:
    st.session_state.active_run_id = None


# ---------------------------------------------------------------------------
# Run discovery / artifact loading helpers
# ---------------------------------------------------------------------------


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not ARTIFACTS_DIR.exists():
        return runs
    for run_dir in sorted(ARTIFACTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        if not is_real_run_id(run_dir.name):
            continue
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {"run_id": run_dir.name, "status": "UNKNOWN"}
        else:
            meta = {"run_id": run_dir.name, "status": "IN_PROGRESS"}
        meta.setdefault("run_id", run_dir.name)
        runs.append(meta)
    return runs


@st.cache_data(show_spinner="Loading clean data...")
def load_clean_data(run_id: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / run_id / "clean_data.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner="Computing business intelligence...")
def load_business_intelligence(run_id: str) -> dict[str, Any]:
    """Recompute the same deterministic business-intelligence dict the
    Analyst Crew used for eda_report.html, so the UI cards below always
    match the embedded report -- no numbers are invented in app.py.
    """
    df = load_clean_data(run_id)
    micro_inspection = compute_micro_inspection(df)
    return compute_business_intelligence(df, micro_inspection)


@st.cache_data(show_spinner="Loading evaluation report...")
def load_evaluation_report(run_id: str) -> dict[str, Any] | None:
    path = ARTIFACTS_DIR / run_id / "evaluation_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_artifact(run_id: str, filename: str) -> str | None:
    path = ARTIFACTS_DIR / run_id / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_resource(show_spinner="Loading selected model...")
def load_selected_model(run_id: str):
    path = ARTIFACTS_DIR / run_id / "selected_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Sidebar: navigation + run selector
# ---------------------------------------------------------------------------

user = current_user()
st.sidebar.title("Navigation")

available_pages = []
if has_permission("run_workflow"):
    available_pages.append("Admin")
if has_permission("view_descriptive"):
    available_pages.append("Descriptive Analytics")
if has_permission("view_predictive"):
    available_pages.append("Predictive Intelligence")

if not available_pages:
    st.error("Your role has no accessible pages. Contact an administrator.")
    st.stop()

page = st.sidebar.radio("Go to", available_pages)

runs = list_runs()
completed_runs = [r for r in runs if r.get("status") == "COMPLETED"]
run_labels = {f"{r['run_id']}  [{r.get('status', '?')}]": r["run_id"] for r in runs}


# ---------------------------------------------------------------------------
# Admin page
# ---------------------------------------------------------------------------


def render_admin_page() -> None:
    st.title("Admin — Workflow Control")

    col1, col2 = st.columns([1, 3])
    with col1:
        any_running = any(e.get("running") for e in st.session_state.run_registry.values())
        if st.button("Start New Run", type="primary", disabled=any_running):
            start_new_run()
            st.rerun()
        if any_running:
            st.caption("A run is already in progress.")

    active_run_id = st.session_state.active_run_id
    if active_run_id and active_run_id in st.session_state.run_registry:
        st.subheader(f"Live run: `{active_run_id}`")
        _render_live_log_panel(active_run_id)

    st.divider()
    st.subheader("Run History")
    if not runs:
        st.info("No runs yet. Click 'Start New Run' to execute the CrewAI Flow.")
        return

    history_rows = []
    for r in runs:
        history_rows.append(
            {
                "run_id": r.get("run_id"),
                "status": r.get("status"),
                "contract_status": r.get("contract_status"),
                "selected_model": r.get("selected_model_name"),
                "started_at": r.get("started_at"),
                "finished_at": r.get("finished_at"),
            }
        )
    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

    failed_runs = [r for r in runs if r.get("status") == "FAILED"]
    if failed_runs:
        st.subheader("Validation Failure Inspection")
        for r in failed_runs:
            with st.expander(f"Run {r['run_id']} — contract validation FAILED"):
                errors = r.get("validation_errors", [])
                warnings = r.get("validation_warnings", [])
                if errors:
                    st.error("Errors:")
                    for e in errors:
                        st.write(f"- {e}")
                if warnings:
                    st.warning("Warnings:")
                    for w in warnings:
                        st.write(f"- {w}")


@st.fragment(run_every=2)
def _render_live_log_panel(run_id: str) -> None:
    entry = st.session_state.run_registry.get(run_id)
    if entry is None:
        st.info("No active run data.")
        return

    st.write(f"Status: **{entry['status']}**")
    log_text = "\n".join(entry["logs"][-300:]) or "(waiting for log output...)"
    st.code(log_text, language="text")

    if not entry["running"]:
        if entry.get("error"):
            st.error(entry["error"])
        elif entry["status"] == "FAILED":
            st.error("Validation gate FAILED — run halted before the Scientist stage.")
            state = entry.get("state") or {}
            for e in state.get("validation_errors", []):
                st.write(f"- {e}")
        elif entry["status"] == "COMPLETED":
            st.success("Run completed successfully.")
            st.cache_data.clear()


# ---------------------------------------------------------------------------
# Descriptive Analytics page ("What Has Happened?")
# ---------------------------------------------------------------------------


def render_descriptive_page() -> None:
    st.title("Descriptive Analytics — What Has Happened?")

    eligible_runs = [r for r in runs if r.get("status") in ("VALIDATED", "SCIENTIST_COMPLETE", "COMPLETED")]
    if not eligible_runs:
        st.warning("No run has passed the validation gate yet. Ask an Admin to run the workflow.")
        return

    labels = {f"{r['run_id']}  [{r.get('status')}]": r["run_id"] for r in eligible_runs}
    selected_label = st.selectbox("Select run", list(labels.keys()))
    run_id = labels[selected_label]

    df = load_clean_data(run_id)

    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Weekly Sales", f"${df['Weekly_Sales'].sum():,.0f}")
    c2.metric("Avg Weekly Sales / record", f"${df['Weekly_Sales'].mean():,.2f}")
    c3.metric("Stores", f"{df['Store'].nunique()}")
    c4.metric("Departments", f"{df['Dept'].nunique()}")

    st.subheader("Sales Trend Over Time")
    monthly = df.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").dt.to_timestamp()
    trend = monthly.groupby("Month")["Weekly_Sales"].mean().reset_index()
    fig_trend = px.line(trend, x="Month", y="Weekly_Sales", title="Mean Weekly Sales by Month")
    st.plotly_chart(fig_trend, use_container_width=True)

    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    if markdown_cols:
        st.subheader("Markdown Correlation Heatmap")
        corr_cols = markdown_cols + ["Weekly_Sales"]
        corr = df[corr_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Business Intelligence")
    business_intelligence = load_business_intelligence(run_id)

    st.markdown("**Holiday Impact**")
    holiday_impact = business_intelligence.get("holiday_impact", {})
    holiday_mean = holiday_impact.get("holiday_mean")
    non_holiday_mean = holiday_impact.get("non_holiday_mean")

    h1, h2 = st.columns(2)
    h1.metric("Holiday Mean Sales", f"${holiday_mean:,.2f}" if holiday_mean is not None else "N/A")
    h2.metric("Non-Holiday Mean Sales", f"${non_holiday_mean:,.2f}" if non_holiday_mean is not None else "N/A")

    if holiday_mean is not None and non_holiday_mean is not None:
        holiday_chart_df = pd.DataFrame(
            {"Week Type": ["Holiday", "Non-Holiday"], "Mean Weekly Sales": [holiday_mean, non_holiday_mean]}
        )
        fig_holiday = px.bar(
            holiday_chart_df,
            x="Week Type",
            y="Mean Weekly Sales",
            color="Week Type",
            title="Holiday vs Non-Holiday Mean Weekly Sales",
        )
        st.plotly_chart(fig_holiday, use_container_width=True)

    top_spikes = holiday_impact.get("stores_with_largest_individual_spikes", [])
    if top_spikes:
        st.markdown("**Top Individual Store Holiday Spikes**")
        spikes_df = pd.DataFrame(top_spikes).rename(
            columns={
                "non_holiday_mean": "Non-Holiday Mean",
                "holiday_mean": "Holiday Mean",
                "spike_ratio": "Spike Ratio",
            }
        )
        spikes_df = spikes_df[["Store", "Non-Holiday Mean", "Holiday Mean", "Spike Ratio"]]
        st.dataframe(spikes_df, use_container_width=True, hide_index=True)

    st.markdown("**Markdown Effect**")
    markdown_effect = business_intelligence.get("markdown_effect", {})
    if markdown_effect:
        with_md = markdown_effect.get("mean_sales_weeks_with_any_markdown")
        without_md = markdown_effect.get("mean_sales_weeks_without_markdown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Sales (Recorded Markdown)", f"${with_md:,.2f}" if with_md is not None else "N/A")
        m2.metric("Mean Sales (No Recorded Markdown)", f"${without_md:,.2f}" if without_md is not None else "N/A")
        m3.metric("Extreme Outlier Count (Store-Date)", markdown_effect.get("extreme_outlier_count", 0))
        st.caption(markdown_effect.get("grain_note", ""))
    else:
        st.info("No markdown columns present in this run's data.")

    st.subheader("Insights (Data Analyst Crew)")
    insights_md = read_text_artifact(run_id, "insights.md")
    if insights_md:
        with st.expander("View insights.md", expanded=False):
            st.markdown(insights_md)
    else:
        st.info("insights.md not found for this run.")

    st.subheader("Full EDA Report")
    eda_html = read_text_artifact(run_id, "eda_report.html")
    if eda_html:
        st.components.v1.html(eda_html, height=800, scrolling=True)
    else:
        st.info("eda_report.html not found for this run.")


# ---------------------------------------------------------------------------
# Predictive Intelligence page ("What Will Happen?")
# ---------------------------------------------------------------------------


def render_predictive_page() -> None:
    st.title("Predictive Intelligence — What Will Happen?")

    eligible_runs = [r for r in runs if r.get("status") == "COMPLETED"]
    if not eligible_runs:
        st.warning("No run has completed the Scientist stage yet. Ask an Admin to run the workflow.")
        return

    labels = {f"{r['run_id']}  [{r.get('status')}]": r["run_id"] for r in eligible_runs}
    selected_label = st.selectbox("Select run", list(labels.keys()))
    run_id = labels[selected_label]

    report = load_evaluation_report(run_id)
    if report is None:
        st.error("evaluation_report.json not found for this run.")
        return

    st.subheader("Candidate Model Comparison")
    metrics_rows = [
        {"model": name, **metrics, "selected": name == report["selected_model_name"]}
        for name, metrics in report["candidate_metrics"].items()
    ]
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    st.subheader("Selected Model")
    selected_metrics = report["candidate_metrics"][report["selected_model_name"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", report["selected_model_name"])
    c2.metric("MAE", f"{selected_metrics['MAE']:,.2f}")
    c3.metric("RMSE", f"{selected_metrics['RMSE']:,.2f}")
    c4.metric("R2", f"{selected_metrics['R2']:.4f}")

    st.subheader("Forecast Visualizer")
    with st.spinner("Scoring the chronological hold-out set..."):
        df = load_clean_data(run_id)
        engineered = engineer_features(df)
        feature_columns = report["feature_columns"]
        _, test_df, cutoff = chronological_split(engineered)
        model = load_selected_model(run_id)
        if model is not None:
            predictions = model.predict(test_df[feature_columns])
            plot_df = test_df[["Date", "Weekly_Sales"]].copy()
            plot_df["Predicted_Sales"] = predictions
            agg = plot_df.groupby("Date")[["Weekly_Sales", "Predicted_Sales"]].mean().reset_index()
            agg = agg.melt(id_vars="Date", var_name="Series", value_name="Sales")
            fig_forecast = px.line(
                agg, x="Date", y="Sales", color="Series",
                title=f"Actual vs Predicted Mean Weekly Sales (test set, after {cutoff.date()})",
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.error("selected_model.joblib not found for this run.")

    st.subheader("Model Card")
    model_card_md = read_text_artifact(run_id, "model_card.md")
    if model_card_md:
        st.markdown(model_card_md)
    else:
        st.info("model_card.md not found for this run.")

    st.subheader("Evaluation Report (narrative)")
    eval_report_md = read_text_artifact(run_id, "evaluation_report.md")
    if eval_report_md:
        with st.expander("View evaluation_report.md", expanded=False):
            st.markdown(eval_report_md)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Admin":
    render_admin_page()
elif page == "Descriptive Analytics":
    render_descriptive_page()
elif page == "Predictive Intelligence":
    render_predictive_page()
