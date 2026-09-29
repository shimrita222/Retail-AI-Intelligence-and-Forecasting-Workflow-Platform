"""Phase 4A Step 1 tests -- interface/dataclass validation only.

No live Supabase connection required (local Supabase needs Docker, which
was unavailable when this step was implemented -- see the Phase 4A Step 1
report). These tests exercise the persistence contract in isolation.
"""

from __future__ import annotations

import pytest

from src.persistence.interfaces import (
    ARTIFACT_TYPES,
    TERMINAL_WORKFLOW_RUN_STATUSES,
    WORKFLOW_RUN_STATUSES,
    ArtifactMetadata,
    PersistenceError,
    RunRecord,
    RunRepository,
    validate_run_id,
    validate_status,
)

VALID_RUN_ID = "20260814T060614Z-7ddf5d40"


def test_valid_run_id_passes():
    validate_run_id(VALID_RUN_ID)  # does not raise


@pytest.mark.parametrize(
    "bad_run_id",
    ["not-a-run-id", "20260814-060614Z-7ddf5d40", "20260814T060614Z-tooLONGhex123", ""],
)
def test_invalid_run_id_raises(bad_run_id):
    with pytest.raises(PersistenceError):
        validate_run_id(bad_run_id)


def test_valid_status_passes():
    for status in WORKFLOW_RUN_STATUSES:
        validate_status(status)  # does not raise


def test_invalid_status_raises():
    with pytest.raises(PersistenceError):
        validate_status("NOT_A_REAL_STATUS")


def test_terminal_statuses_are_subset_of_all_statuses():
    assert TERMINAL_WORKFLOW_RUN_STATUSES <= WORKFLOW_RUN_STATUSES


def test_terminal_statuses_match_retail_flow_contract():
    # Mirrors src/flows/retail_flow.py's real terminal statuses exactly --
    # this test breaks loudly if the DB/persistence status vocabulary ever
    # drifts from the Python contract.
    assert TERMINAL_WORKFLOW_RUN_STATUSES == {
        "COMPLETED",
        "FAILED",
        "GATE_FAILED",
        "VIABILITY_FAILED",
        "ARTIFACT_FAILED",
    }


def test_run_record_accepts_valid_fields():
    run = RunRecord(run_id=VALID_RUN_ID, status="COMPLETED", is_demo=True, source_environment="local")
    assert run.run_id == VALID_RUN_ID
    assert run.is_demo is True


def test_run_record_rejects_invalid_run_id():
    with pytest.raises(PersistenceError):
        RunRecord(run_id="bad", status="COMPLETED")


def test_run_record_rejects_invalid_status():
    with pytest.raises(PersistenceError):
        RunRecord(run_id=VALID_RUN_ID, status="NOT_REAL")


def test_run_record_rejects_invalid_source_environment():
    with pytest.raises(PersistenceError):
        RunRecord(run_id=VALID_RUN_ID, status="COMPLETED", source_environment="not-an-env")


def test_run_record_defaults_are_safe():
    run = RunRecord(run_id=VALID_RUN_ID, status="INITIATED")
    assert run.is_demo is False
    assert run.source_environment == "local"


def test_artifact_metadata_accepts_known_type():
    artifact = ArtifactMetadata(
        artifact_type="selected_model.joblib",
        storage_backend="local",
        path_or_key="artifacts/20260814T060614Z-7ddf5d40/selected_model.joblib",
    )
    assert artifact.artifact_type in ARTIFACT_TYPES


def test_artifact_metadata_rejects_unknown_type():
    with pytest.raises(PersistenceError):
        ArtifactMetadata(
            artifact_type="not_a_real_artifact.bin",
            storage_backend="local",
            path_or_key="x",
        )


def test_artifact_metadata_rejects_unknown_storage_backend():
    with pytest.raises(PersistenceError):
        ArtifactMetadata(
            artifact_type="model_card.md",
            storage_backend="s3",
            path_or_key="x",
        )


def test_artifact_metadata_never_stores_large_binaries_by_design():
    # Structural guardrail: ArtifactMetadata has no field capable of holding
    # binary content -- only a path/key reference plus size/hash metadata.
    field_names = set(ArtifactMetadata.__dataclass_fields__.keys())
    assert field_names == {
        "artifact_type",
        "storage_backend",
        "path_or_key",
        "content_type",
        "size_bytes",
        "sha256",
    }


def test_run_repository_is_abstract():
    with pytest.raises(TypeError):
        RunRepository()  # type: ignore[abstract]


def test_run_repository_defines_all_required_methods():
    required = {
        "create_run",
        "update_run_status",
        "get_run",
        "list_runs",
        "save_candidate_outcomes",
        "save_selected_model",
        "save_artifact_metadata",
        "save_business_insights",
    }
    assert required <= set(RunRepository.__abstractmethods__)
