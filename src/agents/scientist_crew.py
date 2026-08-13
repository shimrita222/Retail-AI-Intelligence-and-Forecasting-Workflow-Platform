"""Data Scientist Crew: Feature Strategy, Model Evaluation, and Model
Governance agents, backed entirely by deterministic ML results.

Guardrail: the LLM agents never train models, compute metrics, or decide
which candidate wins. src/services/feature_pipeline.py and
src/services/ml_trainer.py already did all of that deterministically before
this module is called. The agents only narrate the fixed evaluation_report
dict into evaluation_report.md and model_card.md; the metrics table is
injected verbatim by Python regardless of what the LLM writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _metrics_table_markdown(candidate_metrics: dict[str, dict[str, float]], selected_model_name: str) -> str:
    lines = ["| Model | MAE | RMSE | R2 | Selected |", "|---|---|---|---|---|"]
    for name, metrics in candidate_metrics.items():
        marker = "selected" if name == selected_model_name else ""
        lines.append(
            f"| {name} | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} | {metrics['R2']:.4f} | {marker} |"
        )
    return "\n".join(lines)


def render_evaluation_report_markdown(
    training_result: dict[str, Any], narrative: str | None = None
) -> str:
    metrics_table = _metrics_table_markdown(
        training_result["candidate_metrics"], training_result["selected_model_name"]
    )
    lines = [
        "# Model Evaluation Report\n",
        f"- **Train rows:** {training_result['train_rows']:,}",
        f"- **Test rows:** {training_result['test_rows']:,}",
        f"- **Chronological split cutoff date:** {training_result['split_cutoff_date']}",
        f"- **Feature columns ({len(training_result['feature_columns'])}):** "
        f"{', '.join(training_result['feature_columns'])}",
        "",
        "## Candidate Comparison (chronological hold-out test set)\n",
        metrics_table,
        "",
        f"**Selected model:** `{training_result['selected_model_name']}` "
        "(deterministically chosen: lowest RMSE on the chronological test split).",
    ]
    if narrative:
        lines += ["", "## Model Evaluation Agent Narrative\n", narrative]
    return "\n".join(lines) + "\n"


def render_model_card_markdown(
    training_result: dict[str, Any], contract: dict[str, Any] | None = None, narrative: str | None = None
) -> str:
    selected = training_result["selected_model_name"]
    metrics = training_result["candidate_metrics"][selected]
    lines = [
        f"# Model Card — {selected}\n",
        "## Intended Use",
        "Weekly sales forecasting per (Store, Dept) for the Retail Data Analytics workflow. "
        "Not intended for financial reporting or any decision outside retail demand planning.",
        "",
        "## Training Data",
        f"- Source: Kaggle 'Retail Data Analytics' (stores.csv, features.csv, train.csv).",
        f"- Chronological split cutoff: {training_result['split_cutoff_date']} "
        f"({training_result['train_rows']:,} train rows / {training_result['test_rows']:,} test rows).",
    ]
    if contract:
        lines.append(
            f"- Target `{contract['target_column']}` contract bounds: "
            f"[{contract['columns']['Weekly_Sales']['min']}, {contract['columns']['Weekly_Sales']['max']}]."
        )
    lines += [
        "",
        "## Selected Model & Performance",
        f"- **Model:** {selected}",
        f"- **MAE:** {metrics['MAE']:.2f}",
        f"- **RMSE:** {metrics['RMSE']:.2f}",
        f"- **R2:** {metrics['R2']:.4f}",
        "- Selection rule: lowest RMSE on the chronological hold-out test set, computed deterministically "
        "in src/services/ml_trainer.py (no LLM involved in model choice or metric computation).",
        "",
        "## Candidate Models Considered",
        "Exactly two candidates were trained and compared: `Ridge(alpha=1.0)` and "
        "`RandomForestRegressor(n_estimators=100, random_state=42)`. No other model families were evaluated.",
        "",
        "## Limitations",
        "- Trained on historical US retail data (2010-2012); may not generalize to other markets or eras.",
        "- Chronological hold-out approximates production drift but does not guarantee future accuracy.",
        "- Lag/rolling features require at least 4 prior weeks of history per (Store, Dept); "
        "cold-start series are dropped during feature engineering.",
    ]
    if narrative:
        lines += ["", "## Model Governance Agent Narrative\n", narrative]
    return "\n".join(lines) + "\n"


def _build_crew(training_result: dict[str, Any], feature_columns: list[str]):
    from crewai import Agent, Crew, Process, Task

    feature_strategy_agent = Agent(
        role="Feature Strategy Agent",
        goal="Explain the deterministic feature engineering choices (lags, rolling stats, encodings) and their rationale.",
        backstory="A forecasting specialist who reasons about which historical signals a retail sales model should see.",
        allow_delegation=False,
        verbose=False,
    )

    model_eval_agent = Agent(
        role="Model Evaluation Agent",
        goal="Summarize the deterministic candidate comparison (Ridge vs RandomForest) without altering any metric.",
        backstory="A rigorous ML evaluator who reports exactly what the metrics say, no more and no less.",
        allow_delegation=False,
        verbose=False,
    )

    governance_agent = Agent(
        role="Model Governance Agent",
        goal="Write a model card documenting intended use, limitations, and the deterministic selection rule.",
        backstory="An ML governance reviewer focused on responsible, well-documented model deployment.",
        allow_delegation=False,
        verbose=False,
    )

    feature_task = Task(
        description=(
            "These feature columns were already engineered deterministically in "
            "src/services/feature_pipeline.py (lag1/lag4 sales, rolling 4-week mean/std, "
            "zero-filled markdowns, one-hot store type). Do not propose new features or change any "
            "of these; write a short markdown paragraph (4-6 sentences) explaining why this feature "
            f"set is appropriate for weekly retail forecasting.\n\nFEATURE_COLUMNS:\n{feature_columns}"
        ),
        expected_output="A short markdown paragraph.",
        agent=feature_strategy_agent,
    )

    eval_task = Task(
        description=(
            "This candidate comparison was already computed deterministically; do not recompute or "
            "alter any numbers. Write a short markdown paragraph (4-6 sentences) summarizing the "
            f"comparison and confirming the selection is justified by the metrics.\n\nRESULT:\n"
            f"{json.dumps({'candidate_metrics': training_result['candidate_metrics'], 'selected_model_name': training_result['selected_model_name']}, indent=2)}"
        ),
        expected_output="A short markdown paragraph.",
        agent=model_eval_agent,
    )

    governance_task = Task(
        description=(
            "Using this deterministic training result, write a short markdown paragraph (4-6 sentences) "
            "on responsible-use considerations and limitations for deploying this model in a retail "
            f"forecasting context.\n\nRESULT:\n{json.dumps({'selected_model_name': training_result['selected_model_name'], 'train_rows': training_result['train_rows'], 'test_rows': training_result['test_rows']}, indent=2)}"
        ),
        expected_output="A short markdown paragraph.",
        agent=governance_agent,
    )

    crew = Crew(
        agents=[feature_strategy_agent, model_eval_agent, governance_agent],
        tasks=[feature_task, eval_task, governance_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew, (feature_task, eval_task, governance_task)


def run_scientist_crew(
    training_result: dict[str, Any],
    run_dir: str | Path,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full Data Scientist Crew stage: consumes the already-computed
    deterministic training_result (from ml_trainer.train_and_select_model)
    and writes evaluation_report.md and model_card.md into run_dir.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    narratives: dict[str, str] = {}
    try:
        crew, (feature_task, eval_task, governance_task) = _build_crew(
            training_result, training_result["feature_columns"]
        )
        crew.kickoff()
        narratives["feature_strategy"] = str(feature_task.output)
        narratives["model_evaluation"] = str(eval_task.output)
        narratives["model_governance"] = str(governance_task.output)
    except Exception as exc:  # noqa: BLE001 - LLM/network calls can fail for many reasons
        narratives["error"] = (
            f"LLM narration unavailable ({exc.__class__.__name__}: {exc}); "
            "reports contain deterministic results only."
        )

    combined_eval_narrative = "\n\n".join(
        filter(None, [narratives.get("feature_strategy"), narratives.get("model_evaluation")])
    )
    eval_report_md = render_evaluation_report_markdown(training_result, combined_eval_narrative or None)
    eval_report_path = run_dir / "evaluation_report.md"
    eval_report_path.write_text(eval_report_md, encoding="utf-8")

    model_card_md = render_model_card_markdown(training_result, contract, narratives.get("model_governance"))
    model_card_path = run_dir / "model_card.md"
    model_card_path.write_text(model_card_md, encoding="utf-8")

    return {
        "evaluation_report_path": eval_report_path,
        "model_card_path": model_card_path,
        "narratives": narratives,
    }
