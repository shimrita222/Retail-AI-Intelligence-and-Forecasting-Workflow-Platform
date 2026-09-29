"""Phase 4A Step 1D: unit tests for the fail-closed STAGING target guard.

Pure local logic -- no network, no Supabase client, no credentials. Proves
the guard rejects every unsafe input and accepts only the one independently
verified STAGING project identity.
"""

from __future__ import annotations

import inspect

import pytest

from src.persistence import staging_guard
from src.persistence.staging_guard import (
    VERIFIED_STAGING_PROJECT_REF,
    StagingGuardError,
    assert_staging_target,
)

VALID_URL = f"https://{VERIFIED_STAGING_PROJECT_REF}.supabase.co"


def test_accepts_verified_staging_target():
    assert assert_staging_target(app_env="staging", supabase_url=VALID_URL) == VERIFIED_STAGING_PROJECT_REF


def test_rejects_missing_app_env():
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env=None, supabase_url=VALID_URL)


def test_rejects_empty_app_env():
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env="", supabase_url=VALID_URL)


@pytest.mark.parametrize("bad_env", ["local", "production", "Staging", "STAGING", "dev", "prod"])
def test_rejects_non_staging_app_env(bad_env):
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env=bad_env, supabase_url=VALID_URL)


def test_rejects_missing_supabase_url():
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env="staging", supabase_url=None)


def test_rejects_empty_supabase_url():
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env="staging", supabase_url="")


@pytest.mark.parametrize(
    "bad_url",
    [
        "not-a-url",
        "http://" + VERIFIED_STAGING_PROJECT_REF + ".supabase.co",  # http, not https
        "https://" + VERIFIED_STAGING_PROJECT_REF + ".supabase.co/rest/v1",  # extra path
        "https://" + VERIFIED_STAGING_PROJECT_REF + ".supabase.co/",  # trailing slash
        "https://" + VERIFIED_STAGING_PROJECT_REF + ".supabase.com",  # wrong TLD
        "https://short.supabase.co",  # ref too short
        "https://" + VERIFIED_STAGING_PROJECT_REF.upper() + ".supabase.co",  # uppercase ref
        "https://" + VERIFIED_STAGING_PROJECT_REF + "x.supabase.co",  # ref too long
    ],
)
def test_rejects_malformed_supabase_url(bad_url):
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env="staging", supabase_url=bad_url)


def test_rejects_wrong_but_well_formed_project_ref():
    other_ref = "a" * 20
    assert other_ref != VERIFIED_STAGING_PROJECT_REF
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env="staging", supabase_url=f"https://{other_ref}.supabase.co")


def test_rejects_production_like_configuration():
    # A hypothetical future Production project: well-formed URL, but neither
    # the verified staging ref nor a staging-declared APP_ENV.
    prod_ref = "p" * 20
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env="production", supabase_url=f"https://{prod_ref}.supabase.co")


def test_rejects_production_app_env_even_with_correct_staging_url():
    # The URL alone must never be trusted -- APP_ENV must also explicitly
    # declare staging, so a misconfigured APP_ENV cannot silently run
    # against the right project under the wrong declared intent.
    with pytest.raises(StagingGuardError):
        assert_staging_target(app_env="production", supabase_url=VALID_URL)


def test_reads_from_environment_when_arguments_omitted(monkeypatch):
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("SUPABASE_URL", VALID_URL)
    assert assert_staging_target() == VERIFIED_STAGING_PROJECT_REF


def test_environment_read_also_rejects_unsafe_target(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    with pytest.raises(StagingGuardError):
        assert_staging_target()


def test_guard_module_never_reads_key_material():
    # Not a secrecy requirement in itself (SUPABASE_URL is public/safe
    # config) -- but the guard's decision must never depend on
    # SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY. Check the actual
    # env-var read calls, not prose -- the module's own docstring names
    # both keys to document that they are deliberately NOT read.
    source = inspect.getsource(staging_guard)
    for key_name in ("SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
        assert f'os.environ.get("{key_name}")' not in source
        assert f'os.environ["{key_name}"]' not in source


def test_guard_module_docstring_only_mentions_keys_to_disclaim_them():
    source = inspect.getsource(staging_guard)
    assert "never" in source.lower() and "SUPABASE_ANON_KEY" in source, (
        "expected the module docstring to explicitly disclaim reading key material"
    )
