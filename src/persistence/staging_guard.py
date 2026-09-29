"""Fail-closed target-identity guard for Phase 4A live/integration tests.

Mutating integration tests must never run against anything except the
independently verified STAGING project (Phase 4A Step 1C):

    name:   Retail AI Intelligence - Staging
    ref:    kzdiojsxlxsplyfrtclv
    region: eu-central-1

This guard exists so that safety does not depend only on a human
remembering which URL happens to be configured locally (CLAUDE.md /
Phase 4A Step 1C Production Guard requirement, §22). Every mutating
integration test must call assert_staging_target() -- directly or via the
session-scoped autouse fixture in the live test harness -- before touching
any Supabase client. On anything short of an exact match it raises
StagingGuardError and refuses to proceed.

This module never reads, logs, or returns SUPABASE_ANON_KEY or any
privileged backend secret -- it only inspects APP_ENV and the (public,
non-secret) SUPABASE_URL.
"""

from __future__ import annotations

import os
import re

# The one project independently verified as STAGING in Phase 4A Step 1C
# (supabase projects list --output json, cross-checked against `linked`).
# This is a non-secret project identifier, safe to hardcode.
VERIFIED_STAGING_PROJECT_REF = "kzdiojsxlxsplyfrtclv"

# Supabase project URLs are always https://<20-char-lowercase-ref>.supabase.co
# with no path/query/fragment -- anything else is treated as malformed.
_SUPABASE_URL_PATTERN = re.compile(r"^https://(?P<ref>[a-z0-9]{20})\.supabase\.co$")


class StagingGuardError(Exception):
    """Raised when the live-test target cannot be proven to be the verified
    STAGING project. Callers must treat this as fatal -- never catch it and
    continue, never fall back to a default target."""


def _extract_project_ref(supabase_url: str) -> str:
    match = _SUPABASE_URL_PATTERN.match(supabase_url.strip())
    if not match:
        raise StagingGuardError(
            "SUPABASE_URL is not a well-formed Supabase project URL "
            "(expected exactly https://<20-char-ref>.supabase.co with no "
            "path/query) -- refusing to run a mutating integration test"
        )
    return match.group("ref")


def assert_staging_target(
    app_env: str | None = None,
    supabase_url: str | None = None,
    expected_ref: str = VERIFIED_STAGING_PROJECT_REF,
) -> str:
    """Raise StagingGuardError unless the target is unambiguously the
    verified STAGING project. Returns the verified project ref on success.

    Reads from the given arguments if provided, else from the environment
    (APP_ENV, SUPABASE_URL).
    """
    resolved_app_env = app_env if app_env is not None else os.environ.get("APP_ENV")
    resolved_url = supabase_url if supabase_url is not None else os.environ.get("SUPABASE_URL")

    if not resolved_app_env:
        raise StagingGuardError(
            "APP_ENV is not set -- refusing to run a mutating integration test"
        )
    if resolved_app_env != "staging":
        raise StagingGuardError(
            f"APP_ENV={resolved_app_env!r} is not 'staging' -- refusing to run a "
            "mutating integration test against a non-staging-declared environment "
            "(this also rejects APP_ENV='production' by construction)"
        )
    if not resolved_url:
        raise StagingGuardError(
            "SUPABASE_URL is not set -- refusing to run a mutating integration test"
        )

    ref = _extract_project_ref(resolved_url)
    if ref != expected_ref:
        raise StagingGuardError(
            "SUPABASE_URL does not resolve to the independently verified STAGING "
            f"project ref ({expected_ref!r}) -- refusing to run a mutating "
            "integration test against an unverified/unexpected project"
        )
    return ref
