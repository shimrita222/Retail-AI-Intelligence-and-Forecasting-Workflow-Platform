"""Phase 4A Step 1D -- live RLS + persistence acceptance harness.

This module is SKIPPED IN ITS ENTIRETY unless every variable in
REQUIRED_ENV_VARS is present -- by default (no live credentials configured)
it performs zero network calls and zero mutation, same as the rest of the
suite. It is not run as part of the normal `pytest tests/ -v` baseline in
any environment where those variables are unset.

Scope, by owner decision (Phase 4A Step 1D pre-mutation gate):

- Does NOT create or delete Supabase Auth identities. The four disposable
  STAGING test users (Admin/Analyst/Scientist/Business) are created MANUALLY
  via the Supabase Dashboard -- see the Step 1D manual setup instructions.
  This harness only signs in with credentials supplied locally through
  environment variables.
- Does NOT use SUPABASE_SERVICE_ROLE_KEY anywhere. Every request against an
  application table goes through one of the four real user JWTs, so RLS is
  exercised exactly as a real client would experience it.
- Creates EXACTLY ONE external_data_sources row, deliberately, as the sole
  exception to "no residue": that table currently has no DELETE
  policy/grant for any role, so a row created here cannot be removed
  through any of the four test identities. This is intentional temporary
  STAGING residue for full Admin-write acceptance coverage (owner-approved,
  Phase 4A Step 1D harness-gap closure) -- see
  test_admin_external_data_source_insert_then_update_lifecycle. The row is
  uniquely marked (PHASE4A-STEP1D-ADMIN-SUCCESS-<uuid8>) and its id/name are
  surfaced via a UserWarning (visible in pytest's default warnings summary,
  no -s flag needed) so the owner can delete it manually afterward via the
  Supabase Dashboard SQL Editor. No service-role client is used anywhere in
  this module. Write-DENIAL paths for non-admin roles create nothing.
- workflow_runs test rows are created and deleted per-test via the
  `test_run` fixture (Admin JWT for both), which cascades to
  candidate_outcomes / selected_models / artifacts / business_insights
  automatically (ON DELETE CASCADE) -- no separate cleanup step is needed
  for those four child tables.
"""

from __future__ import annotations

import os
import uuid
import warnings
from datetime import datetime, timezone

import pytest

from src.persistence.interfaces import (
    ArtifactMetadata,
    BusinessInsight,
    CandidateOutcome,
    PersistenceError,
    RunRecord,
)
from src.persistence.staging_guard import assert_staging_target
from src.persistence.supabase_repository import SupabaseRunRepository

REQUIRED_ENV_VARS = (
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "APP_ENV",
    "PHASE4A_ADMIN_EMAIL",
    "PHASE4A_ADMIN_PASSWORD",
    "PHASE4A_ANALYST_EMAIL",
    "PHASE4A_ANALYST_PASSWORD",
    "PHASE4A_SCIENTIST_EMAIL",
    "PHASE4A_SCIENTIST_PASSWORD",
    "PHASE4A_BUSINESS_EMAIL",
    "PHASE4A_BUSINESS_PASSWORD",
)

_missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]

pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "Phase 4A Step 1D live harness requires all Phase 4A staging test "
        f"credentials to be set locally -- missing: {_missing}. Not run by "
        "default; this is expected until the owner creates the four "
        "disposable STAGING test identities and populates local env vars."
    ),
)

TEST_DATASET_SCOPE = "phase4a-step1d-rls-test"
NON_ADMIN_ROLES = ("analyst", "scientist", "business")


def classify_supabase_error(exc: BaseException) -> str:
    """Best-effort classification of a Supabase/PostgREST/GoTrue exception
    into one of: 'rls_denied', 'privilege_denied', 'auth_failed',
    'connectivity_error', 'unknown'. Used so a live-test assertion never
    treats an unrelated failure (bad credentials, network blip, wrong host)
    as if it were a positive authorization-denial result.
    """
    type_name = type(exc).__name__.lower()
    message = str(exc).lower()

    if "authapierror" in type_name or "invalid login credentials" in message or "invalid_grant" in message:
        return "auth_failed"
    if "row-level security" in message or "row level security" in message:
        return "rls_denied"
    if "permission denied for" in message or ("insufficient" in message and "privilege" in message):
        return "privilege_denied"
    if any(
        token in message
        for token in ("connection", "timed out", "timeout", "name or service not known", "getaddrinfo", "network")
    ):
        return "connectivity_error"
    return "unknown"


@pytest.fixture(scope="session", autouse=True)
def _require_verified_staging_target():
    """Hard gate, not a skip: if credentials are present but the target
    cannot be proven to be the verified STAGING project, every test in this
    module fails loudly rather than silently mutating an unverified target.
    Runs before any other fixture/client in this module."""
    assert_staging_target()


def _sign_in(email_env: str, password_env: str):
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    anon_key = os.environ["SUPABASE_ANON_KEY"]
    email = os.environ[email_env]
    password = os.environ[password_env]
    client = create_client(url, anon_key)
    auth_response = client.auth.sign_in_with_password({"email": email, "password": password})
    access_token = auth_response.session.access_token
    client.postgrest.auth(access_token)
    # Authoritative auth.uid() for this session, taken directly from the
    # GoTrue sign-in response -- never from a profiles-table query (Admin's
    # SELECT policy returns every profile, so row order there is not a
    # reliable way to identify "this session's own" row).
    client.phase4a_user_id = auth_response.user.id
    return client


@pytest.fixture(scope="session")
def admin_client():
    return _sign_in("PHASE4A_ADMIN_EMAIL", "PHASE4A_ADMIN_PASSWORD")


@pytest.fixture(scope="session")
def analyst_client():
    return _sign_in("PHASE4A_ANALYST_EMAIL", "PHASE4A_ANALYST_PASSWORD")


@pytest.fixture(scope="session")
def scientist_client():
    return _sign_in("PHASE4A_SCIENTIST_EMAIL", "PHASE4A_SCIENTIST_PASSWORD")


@pytest.fixture(scope="session")
def business_client():
    return _sign_in("PHASE4A_BUSINESS_EMAIL", "PHASE4A_BUSINESS_PASSWORD")


@pytest.fixture
def admin_repository(admin_client):
    return SupabaseRunRepository(admin_client)


def _new_test_run_id() -> str:
    # Matches REAL_RUN_ID_PATTERN: ^\d{8}T\d{6}Z-[0-9a-f]{8}$
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{now}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_run(admin_repository, admin_client):
    """Creates one workflow_runs row as Admin, tagged with the Step 1D test
    markers, and guarantees cleanup via the Admin identity's own DELETE
    privilege (workflow_runs_delete_admin) even if the test body fails --
    the delete cascades to every child table automatically."""
    run_id = _new_test_run_id()
    record = RunRecord(
        run_id=run_id,
        status="INITIATED",
        owner_id=admin_client.phase4a_user_id,
        is_demo=True,
        source_environment="staging",
        dataset_scope=TEST_DATASET_SCOPE,
    )
    admin_repository.create_run(record)
    try:
        yield run_id
    finally:
        admin_repository._client.table("workflow_runs").delete().eq("run_id", run_id).execute()


@pytest.fixture
def test_run_uuid(admin_repository, test_run):
    return admin_repository._resolve_run_uuid(test_run)


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


def test_each_role_can_read_own_profile(admin_client, analyst_client, scientist_client, business_client):
    for client in (admin_client, analyst_client, scientist_client, business_client):
        response = client.table("profiles").select("*").execute()
        assert len(response.data) >= 1, "expected caller's own profile row to be readable"


def test_admin_sees_all_four_test_profiles(admin_client):
    # Admin's SELECT policy is `id = auth.uid() OR is_admin()` -- Admin
    # should see every profile in the project, not just their own.
    response = admin_client.table("profiles").select("id").execute()
    assert len(response.data) >= 4


@pytest.mark.parametrize("role", NON_ADMIN_ROLES)
def test_non_admin_cannot_self_promote_role(role, request):
    client = request.getfixturevalue(f"{role}_client")
    own = client.table("profiles").select("id,role").execute().data[0]
    try:
        client.table("profiles").update({"role": "Admin"}).eq("id", own["id"]).execute()
    except Exception as exc:  # noqa: BLE001 -- expected path
        assert classify_supabase_error(exc) in ("rls_denied", "privilege_denied")
        return
    # No UPDATE policy exists for profiles at all -- PostgREST may instead
    # silently affect zero rows rather than raising. Confirm no change.
    refreshed = client.table("profiles").select("role").eq("id", own["id"]).execute()
    assert refreshed.data[0]["role"] != "Admin", "non-admin must never self-promote to Admin"


# ---------------------------------------------------------------------------
# workflow_runs
# ---------------------------------------------------------------------------


def test_all_four_roles_can_read_shared_workflow_run(
    test_run, admin_client, analyst_client, scientist_client, business_client
):
    for client in (admin_client, analyst_client, scientist_client, business_client):
        response = client.table("workflow_runs").select("run_id").eq("run_id", test_run).execute()
        assert len(response.data) == 1


@pytest.mark.parametrize("role", NON_ADMIN_ROLES)
def test_non_admin_cannot_create_workflow_run(role, request):
    client = request.getfixturevalue(f"{role}_client")
    repository = SupabaseRunRepository(client)
    record = RunRecord(
        run_id=_new_test_run_id(),
        status="INITIATED",
        is_demo=True,
        source_environment="staging",
        dataset_scope=TEST_DATASET_SCOPE,
    )
    with pytest.raises(PersistenceError) as excinfo:
        repository.create_run(record)
    underlying = excinfo.value.__cause__ or excinfo.value
    assert classify_supabase_error(underlying) in ("rls_denied", "privilege_denied")


def test_admin_can_update_own_run_status(admin_repository, test_run):
    admin_repository.update_run_status(test_run, "COMPLETED")
    refreshed = admin_repository.get_run(test_run)
    assert refreshed is not None
    assert refreshed.status == "COMPLETED"


@pytest.mark.parametrize("role", NON_ADMIN_ROLES)
def test_non_admin_cannot_update_workflow_run_status(role, request, test_run, admin_repository):
    client = request.getfixturevalue(f"{role}_client")
    repository = SupabaseRunRepository(client)
    try:
        repository.update_run_status(test_run, "COMPLETED")
    except PersistenceError:
        pass  # expected: raised via the RLS-denial -> exception path
    # Whether or not it raised, confirm via Admin that the row was NOT
    # actually changed (an UPDATE with a RLS-filtered WHERE can silently
    # affect zero rows instead of raising).
    refreshed = admin_repository.get_run(test_run)
    assert refreshed is not None
    assert refreshed.status == "INITIATED", f"{role} must not be able to change workflow_runs.status"


@pytest.mark.parametrize("role", NON_ADMIN_ROLES)
def test_non_admin_cannot_delete_workflow_run(role, request, test_run, admin_repository):
    client = request.getfixturevalue(f"{role}_client")
    response = client.table("workflow_runs").delete().eq("run_id", test_run).execute()
    assert response.data == [], "non-admin DELETE must affect zero rows (RLS-filtered)"
    assert admin_repository.get_run(test_run) is not None, "row must still exist after a denied delete attempt"


# ---------------------------------------------------------------------------
# candidate_outcomes / selected_models / artifacts / business_insights
# (identical RLS shape: shared read, Admin-owned insert only)
# ---------------------------------------------------------------------------

CHILD_TABLE_WRITE_CASES = [
    (
        "candidate_outcomes",
        lambda repo, run_id: repo.save_candidate_outcomes(
            run_id,
            [CandidateOutcome(model_id="ridge", usable=True, family="linear", mae=1.0, rmse=1.2, r2=0.5)],
        ),
        "model_id",
    ),
    (
        "selected_models",
        lambda repo, run_id: repo.save_selected_model(
            run_id,
            model_id="ridge",
            artifact_path="artifacts/phase4a-step1d-test/selected_model.joblib",
            artifact_storage="local",
        ),
        "model_id",
    ),
    (
        "artifacts",
        lambda repo, run_id: repo.save_artifact_metadata(
            run_id,
            ArtifactMetadata(
                artifact_type="run_metadata.json",
                storage_backend="local",
                path_or_key="artifacts/phase4a-step1d-test/run_metadata.json",
            ),
        ),
        "artifact_type",
    ),
    (
        "business_insights",
        lambda repo, run_id: repo.save_business_insights(
            run_id, [BusinessInsight(reason_code="PHASE4A_STEP1D_TEST_EVIDENCE", kind="test")]
        ),
        "reason_code",
    ),
]


@pytest.mark.parametrize(
    "table_name, write_fn, marker_col", CHILD_TABLE_WRITE_CASES, ids=[c[0] for c in CHILD_TABLE_WRITE_CASES]
)
def test_admin_write_then_all_roles_can_read(
    table_name,
    write_fn,
    marker_col,
    admin_repository,
    test_run,
    test_run_uuid,
    admin_client,
    analyst_client,
    scientist_client,
    business_client,
):
    write_fn(admin_repository, test_run)
    for client in (admin_client, analyst_client, scientist_client, business_client):
        response = client.table(table_name).select(marker_col).eq("run_id", test_run_uuid).execute()
        assert len(response.data) >= 1


@pytest.mark.parametrize(
    "table_name, write_fn, marker_col", CHILD_TABLE_WRITE_CASES, ids=[c[0] for c in CHILD_TABLE_WRITE_CASES]
)
@pytest.mark.parametrize("role", NON_ADMIN_ROLES)
def test_non_admin_cannot_write_child_table(table_name, write_fn, marker_col, role, request, test_run):
    client = request.getfixturevalue(f"{role}_client")
    repository = SupabaseRunRepository(client)
    with pytest.raises(PersistenceError):
        write_fn(repository, test_run)


# ---------------------------------------------------------------------------
# external_data_sources -- no repository coverage yet (documented follow-up
# item). Read and write-DENIAL only, via direct client.table() calls.
# Deliberately does NOT insert a row that could succeed: external_data_
# sources has no DELETE policy/grant for any role, so a row created here
# could not be cleaned up through any of the four test identities.
# ---------------------------------------------------------------------------


def test_all_four_roles_can_read_external_data_sources(
    admin_client, analyst_client, scientist_client, business_client
):
    for client in (admin_client, analyst_client, scientist_client, business_client):
        response = client.table("external_data_sources").select("id").execute()
        assert isinstance(response.data, list)  # shared-read must not error, regardless of row count


@pytest.mark.parametrize("role", NON_ADMIN_ROLES)
def test_non_admin_cannot_insert_external_data_source(role, request):
    client = request.getfixturevalue(f"{role}_client")
    marker_name = f"PHASE4A-STEP1D-DENIAL-PROBE-{uuid.uuid4().hex[:8]}"
    try:
        client.table("external_data_sources").insert({"name": marker_name, "is_active": True}).execute()
    except Exception as exc:  # noqa: BLE001 -- expected path
        assert classify_supabase_error(exc) in (
            "rls_denied",
            "privilege_denied",
        ), f"insert failed for an unrelated reason: {classify_supabase_error(exc)}"
        return
    # Reaching here means RLS/grants unexpectedly allowed the insert -- a
    # security regression, and the probe row CANNOT be deleted by any of
    # the four test identities (no DELETE policy exists for anyone on this
    # table). Surface this loudly rather than leaving silent test debris.
    pytest.fail(
        f"SECURITY REGRESSION: {role} was able to INSERT into "
        f"external_data_sources (marker name {marker_name!r}). This row "
        "cannot be cleaned up by any of the four test identities -- it "
        "requires manual owner deletion via the Supabase Dashboard SQL "
        "editor before this test is re-run."
    )


@pytest.mark.parametrize("role", NON_ADMIN_ROLES)
def test_non_admin_cannot_update_external_data_source(role, request):
    client = request.getfixturevalue(f"{role}_client")
    # No row needs to exist for this to be a meaningful denial probe: the
    # UPDATE policy is `using(is_admin())`, so a non-admin's UPDATE matches
    # zero rows regardless of whether any row exists.
    response = client.table("external_data_sources").update({"is_active": False}).eq("name", "does-not-exist").execute()
    assert response.data == []


def test_admin_external_data_source_insert_then_update_lifecycle(admin_client):
    """Full-coverage Admin-success acceptance test for external_data_sources
    (Phase 4A Step 1D harness-gap closure, owner-authorized).

    external_data_sources has no DELETE policy/grant for any role, so this
    is a DELIBERATE, SOLE exception to the "no residue" rule elsewhere in
    this module: exactly one clearly-marked row is created and is expected
    to remain in STAGING after a successful live run, pending manual owner
    deletion via the Supabase Dashboard SQL Editor. Uses only the normal
    Admin Auth/JWT client -- never service-role -- and only schema fields
    that actually exist (name, kind, owner_id, is_active; see
    supabase/migrations/20260903005542_phase4a_base_schema.sql).
    """
    # Authoritative auth.uid() for this session -- from the GoTrue sign-in
    # response (see _sign_in), never from profiles-table ordering: Admin's
    # SELECT policy returns every profile, so data[0] there is not
    # guaranteed to be this session's own row.
    admin_id = admin_client.phase4a_user_id
    marker_name = f"PHASE4A-STEP1D-ADMIN-SUCCESS-{uuid.uuid4().hex[:8]}"

    insert_response = (
        admin_client.table("external_data_sources")
        .insert({"name": marker_name, "kind": "phase4a-step1d-test", "owner_id": admin_id, "is_active": True})
        .execute()
    )
    assert len(insert_response.data) == 1, "Admin INSERT into external_data_sources must succeed (ALLOW expected)"
    row_id = insert_response.data[0]["id"]

    read_after_insert = admin_client.table("external_data_sources").select("id,name,is_active").eq("id", row_id).execute()
    assert len(read_after_insert.data) == 1, "inserted row must be immediately readable by Admin"
    assert read_after_insert.data[0]["name"] == marker_name

    update_response = (
        admin_client.table("external_data_sources").update({"is_active": False}).eq("id", row_id).execute()
    )
    assert len(update_response.data) == 1, "Admin UPDATE of own external_data_sources row must succeed (ALLOW expected)"

    read_after_update = admin_client.table("external_data_sources").select("is_active,name").eq("id", row_id).execute()
    assert len(read_after_update.data) == 1
    assert read_after_update.data[0]["is_active"] is False, "updated value must be readable back"
    assert read_after_update.data[0]["name"] == marker_name

    # Deterministic residue identifier, surfaced via pytest's default
    # warnings summary (shown after the run even without -s/-rA) -- no
    # credential/JWT/key value is included, only the non-secret row id and
    # marker name the owner needs for manual cleanup.
    warnings.warn(
        "PHASE4A STEP 1D RESIDUE: external_data_sources row NOT auto-deleted "
        "(no DELETE policy/grant exists for any role) -- owner must delete "
        f"manually via Supabase Dashboard SQL Editor: id={row_id!r}, name={marker_name!r}",
        UserWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# persistence-abstraction specific checks
# ---------------------------------------------------------------------------


def test_get_run_round_trips_canonical_run_id_and_provenance(admin_repository, test_run):
    record = admin_repository.get_run(test_run)
    assert record is not None
    assert record.run_id == test_run
    assert record.is_demo is True
    assert record.source_environment == "staging"
    assert record.dataset_scope == TEST_DATASET_SCOPE


def test_list_runs_visible_identically_to_all_four_roles(
    test_run, admin_client, analyst_client, scientist_client, business_client
):
    for client in (admin_client, analyst_client, scientist_client, business_client):
        repository = SupabaseRunRepository(client)
        run_ids = {run.run_id for run in repository.list_runs()}
        assert test_run in run_ids


def test_anonymous_client_cannot_read_workflow_runs(test_run):
    from supabase import create_client

    anon_client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    # No sign-in performed -- this client is the `anon` role, not
    # `authenticated`. `anon` has no table-level SELECT grant on
    # workflow_runs (deliberate, least-privilege: see
    # 20260914205942_phase4a_data_api_grants.sql -- "grants nothing to
    # anon"), so PostgREST rejects the request with a Postgres
    # permission-denied error (42501) before RLS is ever evaluated. This is
    # a stricter, not weaker, form of "anon gets zero business-data access"
    # than an RLS-filtered empty result would be -- do not grant anon SELECT
    # to make this look like a plain empty read.
    try:
        response = anon_client.table("workflow_runs").select("run_id").eq("run_id", test_run).execute()
    except Exception as exc:  # noqa: BLE001 -- expected path: anon has no grant on this table
        assert classify_supabase_error(exc) == "privilege_denied", (
            f"expected a privilege-denied error for anon, got {classify_supabase_error(exc)}: {exc}"
        )
        return
    pytest.fail(
        "SECURITY REGRESSION: anon SELECT on workflow_runs unexpectedly succeeded "
        f"with data={response.data!r} -- anon may have gained a SELECT grant/policy"
    )
