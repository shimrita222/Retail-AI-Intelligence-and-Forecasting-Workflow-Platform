"""Persistence/repository abstraction for Phase 4A (Supabase foundation).

Nothing in this package is wired into app.py yet -- that integration is
later Phase 4A/4B work. This package exists so Streamlit pages and CrewAI
agents never need to know Supabase table names, raw SQL, or hold privileged
credentials: all business-data access goes through the RunRepository
interface defined in interfaces.py.
"""

from __future__ import annotations

from src.persistence.interfaces import (
    ArtifactMetadata,
    CandidateOutcome,
    PersistenceError,
    RunRecord,
    RunRepository,
)

__all__ = [
    "ArtifactMetadata",
    "CandidateOutcome",
    "PersistenceError",
    "RunRecord",
    "RunRepository",
]
