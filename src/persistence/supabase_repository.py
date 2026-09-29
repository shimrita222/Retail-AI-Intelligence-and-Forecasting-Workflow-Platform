"""Supabase-backed implementation of RunRepository.

Not wired into app.py yet -- that integration is later Phase 4A/4B work.
This is the Step 1 foundation: a working, independently testable
implementation of the persistence contract, exercised in tests against a
fake/mock client rather than a live Supabase instance (local Supabase
requires Docker, which was unavailable when this step was implemented --
see the Phase 4A Step 1 report).

Every public method catches the underlying client's exceptions and re-raises
PersistenceError, so app.py/agents never see a raw Supabase/Postgres error
type.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.persistence.interfaces import (
    ArtifactMetadata,
    BusinessInsight,
    CandidateOutcome,
    PersistenceError,
    RunRecord,
    RunRepository,
    validate_run_id,
    validate_status,
)


class SupabaseRunRepository(RunRepository):
    """RunRepository implementation backed by a Supabase Postgres schema
    (see supabase/migrations/20260903005542_phase4a_base_schema.sql).

    `client` is expected to already be scoped correctly by the caller --
    see src/persistence/supabase_client.py. This class never decides which
    credential to use; it only issues queries through whatever client it is
    given, so RLS (or its deliberate bypass for the rare service-role case)
    is entirely determined by the caller.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def create_run(self, run: RunRecord) -> str:
        try:
            response = (
                self._client.table("workflow_runs")
                .insert(_run_record_to_row(run))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001 - normalize to PersistenceError
            raise PersistenceError(f"create_run failed for {run.run_id}: {exc}") from exc
        rows = getattr(response, "data", None) or []
        if not rows:
            raise PersistenceError(f"create_run for {run.run_id} returned no row")
        return rows[0]["id"]

    def update_run_status(self, run_id: str, status: str, **fields: Any) -> None:
        validate_run_id(run_id)
        validate_status(status)
        payload = {"status": status, **fields}
        try:
            self._client.table("workflow_runs").update(payload).eq("run_id", run_id).execute()
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(f"update_run_status failed for {run_id}: {exc}") from exc

    def get_run(self, run_id: str) -> RunRecord | None:
        validate_run_id(run_id)
        try:
            response = (
                self._client.table("workflow_runs").select("*").eq("run_id", run_id).execute()
            )
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(f"get_run failed for {run_id}: {exc}") from exc
        rows = getattr(response, "data", None) or []
        if not rows:
            return None
        return _row_to_run_record(rows[0])

    def list_runs(self) -> list[RunRecord]:
        try:
            response = self._client.table("workflow_runs").select("*").execute()
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(f"list_runs failed: {exc}") from exc
        rows = getattr(response, "data", None) or []
        return [_row_to_run_record(row) for row in rows]

    def save_candidate_outcomes(self, run_id: str, outcomes: list[CandidateOutcome]) -> None:
        run_uuid = self._resolve_run_uuid(run_id)
        rows = [
            {
                "run_id": run_uuid,
                "model_id": outcome.model_id,
                "family": outcome.family,
                "mae": outcome.mae,
                "rmse": outcome.rmse,
                "r2": outcome.r2,
                "usable": outcome.usable,
                "reason_code": outcome.reason_code,
                "reason_params": outcome.reason_params,
            }
            for outcome in outcomes
        ]
        if not rows:
            return
        try:
            self._client.table("candidate_outcomes").upsert(rows, on_conflict="run_id,model_id").execute()
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(f"save_candidate_outcomes failed for {run_id}: {exc}") from exc

    def save_selected_model(
        self,
        run_id: str,
        model_id: str,
        artifact_path: str,
        artifact_storage: str,
        tie_break_applied: bool = False,
        tie_break_reason_params: dict[str, Any] | None = None,
    ) -> None:
        run_uuid = self._resolve_run_uuid(run_id)
        row = {
            "run_id": run_uuid,
            "model_id": model_id,
            "artifact_path": artifact_path,
            "artifact_storage": artifact_storage,
            "tie_break_applied": tie_break_applied,
            "tie_break_reason_params": tie_break_reason_params,
        }
        try:
            self._client.table("selected_models").upsert(row, on_conflict="run_id").execute()
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(f"save_selected_model failed for {run_id}: {exc}") from exc

    def save_artifact_metadata(self, run_id: str, artifact: ArtifactMetadata) -> None:
        run_uuid = self._resolve_run_uuid(run_id)
        row = {"run_id": run_uuid, **asdict(artifact)}
        try:
            self._client.table("artifacts").upsert(row, on_conflict="run_id,artifact_type").execute()
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(f"save_artifact_metadata failed for {run_id}: {exc}") from exc

    def save_business_insights(self, run_id: str, insights: list[BusinessInsight]) -> None:
        run_uuid = self._resolve_run_uuid(run_id)
        rows = [{"run_id": run_uuid, **asdict(insight)} for insight in insights]
        if not rows:
            return
        try:
            self._client.table("business_insights").insert(rows).execute()
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(f"save_business_insights failed for {run_id}: {exc}") from exc

    def _resolve_run_uuid(self, run_id: str) -> str:
        validate_run_id(run_id)
        try:
            response = (
                self._client.table("workflow_runs").select("id").eq("run_id", run_id).execute()
            )
        except Exception as exc:  # noqa: BLE001
            raise PersistenceError(f"could not resolve run uuid for {run_id}: {exc}") from exc
        rows = getattr(response, "data", None) or []
        if not rows:
            raise PersistenceError(f"no workflow_runs row found for run_id {run_id}")
        return rows[0]["id"]


def _run_record_to_row(run: RunRecord) -> dict[str, Any]:
    row = asdict(run)
    return row


def _row_to_run_record(row: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        status=row["status"],
        owner_id=row.get("owner_id"),
        contract_status=row.get("contract_status"),
        is_demo=row.get("is_demo", False),
        source_environment=row.get("source_environment", "local"),
        dataset_scope=row.get("dataset_scope"),
        train_row_count=row.get("train_row_count"),
        test_row_count=row.get("test_row_count"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )
