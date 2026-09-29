"""Phase 4A Step 1 tests for SupabaseRunRepository, exercised against a
fake in-memory client (no live Supabase / no Docker required -- local
Supabase was unavailable when this step was implemented, see the Step 1
report). These tests verify the repository's query shape and error-wrapping
behavior, not live Postgres/RLS enforcement (that requires an actual local
Supabase instance and is deferred to when Docker is available).
"""

from __future__ import annotations

import pytest

from src.persistence.interfaces import (
    ArtifactMetadata,
    BusinessInsight,
    CandidateOutcome,
    PersistenceError,
    RunRecord,
)
from src.persistence.supabase_repository import SupabaseRunRepository

VALID_RUN_ID = "20260814T060614Z-7ddf5d40"
RUN_UUID = "80b00f95-8607-44fd-994c-afc3d99812c6"


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """Minimal stand-in for postgrest-py's fluent query builder."""

    def __init__(self, table: "_FakeTable", op: str, payload=None):
        self._table = table
        self._op = op
        self._payload = payload
        self._filters: dict[str, object] = {}

    def eq(self, column, value):
        self._filters[column] = value
        return self

    def execute(self):
        return self._table._execute(self._op, self._payload, self._filters)


class _FakeTable:
    def __init__(self, name: str, store: dict[str, list[dict]], fail: bool = False):
        self.name = name
        self._store = store.setdefault(name, [])
        self._fail = fail
        self.calls: list[tuple] = []

    def insert(self, payload):
        self.calls.append(("insert", payload))
        return _FakeQuery(self, "insert", payload)

    def update(self, payload):
        self.calls.append(("update", payload))
        return _FakeQuery(self, "update", payload)

    def upsert(self, payload, on_conflict=None):
        self.calls.append(("upsert", payload, on_conflict))
        return _FakeQuery(self, "upsert", payload)

    def select(self, *_args):
        self.calls.append(("select",))
        return _FakeQuery(self, "select")

    def _execute(self, op, payload, filters):
        if self._fail:
            raise RuntimeError("simulated Supabase failure")

        if op == "insert":
            rows = payload if isinstance(payload, list) else [payload]
            stored_rows = []
            for row in rows:
                row = dict(row)
                row.setdefault("id", RUN_UUID)
                self._store.append(row)
                stored_rows.append(row)
            return _FakeResponse(stored_rows)

        if op == "upsert":
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                self._store.append(dict(row))
            return _FakeResponse(rows)

        if op in ("select", "update"):
            matched = [
                row for row in self._store if all(row.get(k) == v for k, v in filters.items())
            ]
            if op == "update":
                for row in matched:
                    row.update(payload)
            return _FakeResponse(matched)

        raise AssertionError(f"unexpected op {op}")


class _FakeClient:
    def __init__(self, fail: bool = False):
        self._store: dict[str, list[dict]] = {}
        self._fail = fail

    def table(self, name: str) -> _FakeTable:
        return _FakeTable(name, self._store, fail=self._fail)


def _repo_with_existing_run() -> SupabaseRunRepository:
    client = _FakeClient()
    repo = SupabaseRunRepository(client)
    run = RunRecord(run_id=VALID_RUN_ID, status="INITIATED")
    repo.create_run(run)
    return repo


def test_create_run_inserts_and_returns_uuid():
    client = _FakeClient()
    repo = SupabaseRunRepository(client)
    run = RunRecord(run_id=VALID_RUN_ID, status="INITIATED")
    run_uuid = repo.create_run(run)
    assert run_uuid == RUN_UUID


def test_get_run_returns_none_when_missing():
    client = _FakeClient()
    repo = SupabaseRunRepository(client)
    assert repo.get_run(VALID_RUN_ID) is None


def test_get_run_round_trips_after_create():
    repo = _repo_with_existing_run()
    fetched = repo.get_run(VALID_RUN_ID)
    assert fetched is not None
    assert fetched.run_id == VALID_RUN_ID
    assert fetched.status == "INITIATED"


def test_update_run_status_is_idempotent():
    repo = _repo_with_existing_run()
    repo.update_run_status(VALID_RUN_ID, "COMPLETED")
    repo.update_run_status(VALID_RUN_ID, "COMPLETED")  # re-applying must not raise
    fetched = repo.get_run(VALID_RUN_ID)
    assert fetched.status == "COMPLETED"


def test_update_run_status_rejects_invalid_status():
    repo = _repo_with_existing_run()
    with pytest.raises(PersistenceError):
        repo.update_run_status(VALID_RUN_ID, "NOT_A_REAL_STATUS")


def test_list_runs_returns_created_run():
    repo = _repo_with_existing_run()
    runs = repo.list_runs()
    assert any(r.run_id == VALID_RUN_ID for r in runs)


def test_save_candidate_outcomes_upserts():
    repo = _repo_with_existing_run()
    outcomes = [
        CandidateOutcome(model_id="Ridge", usable=True, rmse=3193.8),
        CandidateOutcome(model_id="RandomForestRegressor", usable=True, rmse=3138.5),
    ]
    repo.save_candidate_outcomes(VALID_RUN_ID, outcomes)  # does not raise


def test_save_selected_model_upserts():
    repo = _repo_with_existing_run()
    repo.save_selected_model(
        VALID_RUN_ID,
        model_id="RandomForestRegressor",
        artifact_path="artifacts/20260814T060614Z-7ddf5d40/selected_model.joblib",
        artifact_storage="local",
    )  # does not raise


def test_save_artifact_metadata_upserts():
    repo = _repo_with_existing_run()
    artifact = ArtifactMetadata(
        artifact_type="model_card.md",
        storage_backend="local",
        path_or_key="artifacts/20260814T060614Z-7ddf5d40/model_card.md",
    )
    repo.save_artifact_metadata(VALID_RUN_ID, artifact)  # does not raise


def test_save_business_insights_inserts():
    repo = _repo_with_existing_run()
    insights = [BusinessInsight(reason_code="holiday_spike_detected", kind="holiday_impact")]
    repo.save_business_insights(VALID_RUN_ID, insights)  # does not raise


def test_save_candidate_outcomes_raises_for_unknown_run():
    client = _FakeClient()
    repo = SupabaseRunRepository(client)
    with pytest.raises(PersistenceError):
        repo.save_candidate_outcomes(VALID_RUN_ID, [CandidateOutcome(model_id="Ridge", usable=True)])


def test_client_failure_is_wrapped_as_persistence_error():
    client = _FakeClient(fail=True)
    repo = SupabaseRunRepository(client)
    run = RunRecord(run_id=VALID_RUN_ID, status="INITIATED")
    with pytest.raises(PersistenceError):
        repo.create_run(run)
