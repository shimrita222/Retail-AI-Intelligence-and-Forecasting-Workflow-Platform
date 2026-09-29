"""Repository interface for Phase 4A persistence.

This module defines the *contract* only -- src/persistence/supabase_repository.py
provides the concrete Supabase-backed implementation. Phase 4B's future
backend service is expected to call the same RunRepository interface, not a
rewritten one, so nothing here should assume it runs inside a Streamlit
process.

Status values intentionally mirror RetailFlowState.status
(src/flows/retail_flow.py) exactly -- do not add a status here that Python
does not also produce, and do not let this list drift from the database
CHECK constraint in supabase/migrations/20260903005542_phase4a_base_schema.sql.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.services.run_registry import REAL_RUN_ID_PATTERN

# Mirrors RetailFlowState.status (src/flows/retail_flow.py) and the
# workflow_runs.status CHECK constraint in the Phase 4A migration.
WORKFLOW_RUN_STATUSES: frozenset[str] = frozenset(
    {
        "INITIATED",
        "ANALYST_COMPLETE",
        "VALIDATED",
        "SCIENTIST_COMPLETE",
        "COMPLETED",
        "FAILED",
        "GATE_FAILED",
        "VIABILITY_FAILED",
        "ARTIFACT_FAILED",
    }
)

TERMINAL_WORKFLOW_RUN_STATUSES: frozenset[str] = frozenset(
    {"COMPLETED", "FAILED", "GATE_FAILED", "VIABILITY_FAILED", "ARTIFACT_FAILED"}
)

SOURCE_ENVIRONMENTS: frozenset[str] = frozenset({"local", "staging", "production"})

ARTIFACT_STORAGE_BACKENDS: frozenset[str] = frozenset({"local", "supabase_storage"})

# Mirrors artifacts.artifact_type CHECK constraint in the Phase 4A migration
# -- the real artifact filenames confirmed present under artifacts/<run_id>/.
ARTIFACT_TYPES: frozenset[str] = frozenset(
    {
        "run_metadata.json",
        "evaluation_report.json",
        "evaluation_report.md",
        "model_card.md",
        "selected_model.joblib",
        "clean_data.csv",
        "dataset_contract.json",
        "eda_report.html",
        "insights.md",
        "validation_result.json",
    }
)


class PersistenceError(Exception):
    """Raised for any persistence failure. Never lets a raw Supabase/Postgres
    exception reach app.py or a CrewAI agent -- callers only ever see this
    type (or a subclass) out of this package.
    """


def validate_run_id(run_id: str) -> None:
    """Raise PersistenceError if run_id doesn't match the canonical app
    format (src/services/run_registry.py REAL_RUN_ID_PATTERN)."""
    if not REAL_RUN_ID_PATTERN.match(run_id):
        raise PersistenceError(f"run_id {run_id!r} does not match the canonical run_id format")


def validate_status(status: str) -> None:
    if status not in WORKFLOW_RUN_STATUSES:
        raise PersistenceError(f"status {status!r} is not a recognized workflow run status")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: str
    owner_id: str | None = None
    contract_status: str | None = None
    is_demo: bool = False
    source_environment: str = "local"
    dataset_scope: str | None = None
    train_row_count: int | None = None
    test_row_count: int | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        validate_status(self.status)
        if self.source_environment not in SOURCE_ENVIRONMENTS:
            raise PersistenceError(f"source_environment {self.source_environment!r} is invalid")


@dataclass(frozen=True)
class CandidateOutcome:
    model_id: str
    usable: bool
    family: str | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    reason_code: str | None = None
    reason_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_type: str
    storage_backend: str
    path_or_key: str
    content_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.artifact_type not in ARTIFACT_TYPES:
            raise PersistenceError(f"artifact_type {self.artifact_type!r} is not a recognized artifact type")
        if self.storage_backend not in ARTIFACT_STORAGE_BACKENDS:
            raise PersistenceError(f"storage_backend {self.storage_backend!r} is invalid")


@dataclass(frozen=True)
class BusinessInsight:
    reason_code: str
    reason_params: dict[str, Any] = field(default_factory=dict)
    kind: str | None = None


class RunRepository(ABC):
    """Persistence abstraction for workflow-run metadata.

    Every method is idempotent where the underlying operation is naturally
    re-appliable (status updates, candidate/artifact/insight upserts) -- a
    caller retrying after a transient failure must not create duplicate
    rows or corrupt state. Implementations raise PersistenceError (never a
    raw client/library exception) on failure.
    """

    @abstractmethod
    def create_run(self, run: RunRecord) -> str:
        """Create a new workflow_runs row. Returns the DB-internal run uuid
        (str). Raises PersistenceError if run.run_id already exists."""

    @abstractmethod
    def update_run_status(
        self,
        run_id: str,
        status: str,
        **fields: Any,
    ) -> None:
        """Idempotently update a run's status (and optionally other
        RunRecord fields, e.g. finished_at, contract_status, reason codes).
        Re-applying the same status is a no-op, not an error."""

    @abstractmethod
    def get_run(self, run_id: str) -> RunRecord | None:
        """Return the run, or None if it does not exist."""

    @abstractmethod
    def list_runs(self) -> list[RunRecord]:
        """Return all runs visible under the caller's current role-scoped
        access (shared role-based access model -- see Phase 4A spec).
        Ordering/filtering beyond RLS is a caller concern, not this
        interface's."""

    @abstractmethod
    def save_candidate_outcomes(self, run_id: str, outcomes: list[CandidateOutcome]) -> None:
        """Idempotent upsert on (run_id, model_id)."""

    @abstractmethod
    def save_selected_model(
        self,
        run_id: str,
        model_id: str,
        artifact_path: str,
        artifact_storage: str,
        tie_break_applied: bool = False,
        tie_break_reason_params: dict[str, Any] | None = None,
    ) -> None:
        """Idempotent upsert -- selected_models is 1:1 with a run."""

    @abstractmethod
    def save_artifact_metadata(self, run_id: str, artifact: ArtifactMetadata) -> None:
        """Idempotent upsert on (run_id, artifact_type). Stores a
        path/size/hash reference only -- never the binary content."""

    @abstractmethod
    def save_business_insights(self, run_id: str, insights: list[BusinessInsight]) -> None:
        """Appends structured, language-neutral evidence rows. Never stores
        rendered en/he narration -- that is generated at presentation time
        (Phase 4C) from this evidence."""
