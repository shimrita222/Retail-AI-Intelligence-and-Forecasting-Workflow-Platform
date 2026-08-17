"""CrewAI Flow orchestrating the full retail intelligence & forecasting run.

Lifecycle: initiate_flow -> run_analyst_stage -> validate_contract_gate
  -> (PASS) run_scientist_stage -> finalize_flow
  -> (FAIL) handle_validation_failure (halts; no scientist stage runs)

The validation gate itself is pure Python (contract_validator.validate_contract);
the Flow only branches on its PASS/FAIL result. No LLM decides whether the
run continues.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from src.agents.analyst_crew import run_analyst_crew
from src.agents.scientist_crew import run_scientist_crew
from src.services.contract_validator import validate_contract
from src.services.data_ingestion import ingest_data
from src.services.feature_pipeline import engineer_features, get_feature_columns
from src.services.ml_trainer import save_artifacts, train_and_select_model
from src.services.modeling_eligibility import (
    get_exclusions,
    modeling_eligible_mask,
    summarize_modeling_population,
)


class RetailFlowState(BaseModel):
    run_id: str = ""
    run_dir: str = ""
    status: str = "PENDING"
    clean_data_path: str = ""
    contract_path: str = ""
    contract_status: str = ""
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    selected_model_name: str = ""
    candidate_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    modeling_population: dict[str, Any] = Field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""


class RetailFlow(Flow[RetailFlowState]):
    """Stateful orchestration of ingestion -> Analyst Crew -> validation gate
    -> Scientist Crew -> finalize, with one artifact directory per run.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        artifacts_root: str | Path = "artifacts",
        run_id: str | None = None,
    ) -> None:
        super().__init__()
        self._raw_dir = Path(raw_dir)
        self._artifacts_root = Path(artifacts_root)
        self._run_id_override = run_id

    @start()
    def initiate_flow(self) -> str:
        run_id = self._run_id_override or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        run_dir = self._artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self.state.run_id = run_id
        self.state.run_dir = str(run_dir)
        self.state.status = "INITIATED"
        self.state.started_at = datetime.now(timezone.utc).isoformat()
        return run_id

    @listen(initiate_flow)
    def run_analyst_stage(self, _run_id: str) -> dict[str, Any]:
        run_dir = Path(self.state.run_dir)
        clean_data_path = run_dir / "clean_data.csv"

        ingest_data(self._raw_dir, clean_data_path)
        self.state.clean_data_path = str(clean_data_path)

        analyst_result = run_analyst_crew(clean_data_path, run_dir)
        self.state.contract_path = str(analyst_result["contract_path"])
        self.state.status = "ANALYST_COMPLETE"
        return analyst_result

    @router(run_analyst_stage)
    def validate_contract_gate(self, _analyst_result: dict[str, Any]) -> str:
        validation = validate_contract(self.state.clean_data_path, self.state.contract_path)

        # Modeling-eligibility concerns are a policy decision (src/services/
        # modeling_eligibility.py), not a structural data-validity rule, so they
        # are appended here as warnings rather than encoded into
        # dataset_contract.json's structural bounds.
        clean_df = pd.read_csv(self.state.clean_data_path)
        for dept, exclusion in get_exclusions().items():
            if "Dept" in clean_df.columns and (clean_df["Dept"] == dept).any():
                validation["warnings"].append(
                    f"Department {dept} is present and will be excluded from the predictive "
                    f"modeling population (modeling-eligibility decision, not data cleaning): "
                    f"{exclusion.reason}"
                )

        self.state.contract_status = validation["status"]
        self.state.validation_errors = validation["errors"]
        self.state.validation_warnings = validation["warnings"]

        validation_path = Path(self.state.run_dir) / "validation_result.json"
        with validation_path.open("w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2)

        if validation["status"] == "FAIL":
            self.state.status = "FAILED"
            return "FAIL"

        self.state.status = "VALIDATED"
        return "PASS"

    @listen("FAIL")
    def handle_validation_failure(self) -> dict[str, Any]:
        self._write_run_metadata()
        return {
            "run_id": self.state.run_id,
            "status": self.state.status,
            "errors": self.state.validation_errors,
        }

    @listen("PASS")
    def run_scientist_stage(self) -> dict[str, Any]:
        run_dir = Path(self.state.run_dir)

        clean_df = pd.read_csv(self.state.clean_data_path)

        # Descriptive data (clean_data.csv on disk, insights.md, eda_report.html)
        # already reflects the FULL department population and is never filtered
        # here. Only the in-memory population handed to feature engineering /
        # model training is restricted to the modeling-eligible departments --
        # see src/services/modeling_eligibility.py for the policy and evidence.
        modeling_population = summarize_modeling_population(clean_df)
        eligible_df = clean_df[modeling_eligible_mask(clean_df)].copy()

        engineered = engineer_features(eligible_df)
        feature_columns = get_feature_columns(engineered)

        training_result = train_and_select_model(engineered, feature_columns)
        training_result["modeling_population"] = modeling_population
        save_artifacts(training_result, run_dir)

        contract = json.loads(Path(self.state.contract_path).read_text(encoding="utf-8"))
        run_scientist_crew(training_result, run_dir, contract=contract)

        self.state.selected_model_name = training_result["selected_model_name"]
        self.state.candidate_metrics = training_result["candidate_metrics"]
        self.state.modeling_population = modeling_population
        self.state.status = "SCIENTIST_COMPLETE"
        return training_result

    @listen(run_scientist_stage)
    def finalize_flow(self, _training_result: dict[str, Any]) -> dict[str, Any]:
        self.state.status = "COMPLETED"
        return self._write_run_metadata()

    def _write_run_metadata(self) -> dict[str, Any]:
        self.state.finished_at = datetime.now(timezone.utc).isoformat()
        metadata = self.state.model_dump()
        metadata_path = Path(self.state.run_dir) / "run_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return metadata


def run_retail_flow(
    raw_dir: str | Path = "data/raw",
    artifacts_root: str | Path = "artifacts",
    run_id: str | None = None,
) -> RetailFlowState:
    """Convenience entrypoint: build and kick off a RetailFlow, return final state."""
    flow = RetailFlow(raw_dir=raw_dir, artifacts_root=artifacts_root, run_id=run_id)
    flow.kickoff()
    return flow.state
