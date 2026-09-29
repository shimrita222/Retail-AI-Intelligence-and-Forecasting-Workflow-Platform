"""Phase 4A Step 1: structural lint of the base-schema migration SQL.

This does NOT apply the migration or connect to a database (local Supabase
needs Docker, which was unavailable when this step was implemented -- see
the Step 1 report). It only verifies the committed SQL file contains the
expected tables, RLS enablement, and no anon-role business-data policy --
a cheap regression guard against someone silently weakening the schema
later, independent of having a live Postgres available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = REPO_ROOT / "supabase" / "migrations"

EXPECTED_TABLES = [
    "profiles",
    "workflow_runs",
    "candidate_outcomes",
    "selected_models",
    "artifacts",
    "business_insights",
    "external_data_sources",
]

EXPECTED_STATUSES = [
    "INITIATED",
    "ANALYST_COMPLETE",
    "VALIDATED",
    "SCIENTIST_COMPLETE",
    "COMPLETED",
    "FAILED",
    "GATE_FAILED",
    "VIABILITY_FAILED",
    "ARTIFACT_FAILED",
]


@pytest.fixture(scope="module")
def migration_sql() -> str:
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    assert sql_files, "expected at least one migration file under supabase/migrations/"
    return "\n".join(f.read_text(encoding="utf-8") for f in sql_files)


def test_migrations_directory_exists():
    assert MIGRATIONS_DIR.is_dir()


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_every_expected_table_is_created(migration_sql, table):
    assert f"create table public.{table}" in migration_sql


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_rls_is_enabled_on_every_table(migration_sql, table):
    assert f"alter table public.{table} enable row level security" in migration_sql


def test_status_check_constraint_matches_retail_flow_contract(migration_sql):
    for status in EXPECTED_STATUSES:
        assert f"'{status}'" in migration_sql


def test_run_id_pattern_matches_run_registry_contract(migration_sql):
    from src.services.run_registry import REAL_RUN_ID_PATTERN

    assert REAL_RUN_ID_PATTERN.pattern in migration_sql


def test_no_anon_role_policy_grants_business_data_access(migration_sql):
    # Every `to authenticated` policy is fine; nothing should grant `anon`
    # access to business tables (deny-by-default via absence of a policy).
    assert "to anon" not in migration_sql.lower()


def test_role_column_has_no_client_writable_policy(migration_sql):
    # profiles.role must never be reachable via a client INSERT/UPDATE
    # policy -- only the SECURITY DEFINER trigger (owner-run, bypasses RLS)
    # may set it, and role elevation is a privileged backend-only path.
    assert "create policy profiles_update" not in migration_sql
    assert "create policy profiles_insert" not in migration_sql


def test_selected_model_binary_is_never_stored_as_a_column(migration_sql):
    # selected_models must only ever hold a path reference, never bytea/binary.
    selected_models_section = migration_sql.split("create table public.selected_models")[1].split(
        ";"
    )[0]
    assert "bytea" not in selected_models_section.lower()


def test_no_hardcoded_secret_looking_literal_in_migration(migration_sql):
    assert "eyJhbGciOi" not in migration_sql  # JWT-shaped literal
    assert "service_role_key" not in migration_sql.lower() or "SUPABASE_SERVICE_ROLE_KEY" not in migration_sql
