"""Supabase client factory.

Reads connection details from environment variables only -- never hardcoded,
never logged, never returned to a caller. This is the ONLY place in the
codebase permitted to read SUPABASE_SERVICE_ROLE_KEY; every other caller
must go through RunRepository instead of touching a client directly.

least-privilege principle: build_user_scoped_client() (an authenticated
user's own JWT) is the default for every interactive read/write. build_
service_role_client() exists only for the narrow backend-only operations
documented in the Phase 4A specification (e.g. the auth.users provisioning
trigger, which actually runs inside Postgres and doesn't need this client at
all) -- callers should prefer the user-scoped client unless they have a
specific, documented reason not to.
"""

from __future__ import annotations

import os

from src.persistence.interfaces import PersistenceError


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise PersistenceError(
            f"required environment variable {name} is not set -- see .env.example"
        )
    return value


def build_user_scoped_client(access_token: str):
    """Build a Supabase client authenticated as a specific end user, so
    Postgres RLS is enforced using that user's own JWT. This is the
    preferred client for all interactive reads/writes (see the approved
    Phase 4A owner decision: background-thread writes carry the initiating
    user's identity, not the service-role key).
    """
    from supabase import Client, create_client

    url = _require_env("SUPABASE_URL")
    anon_key = _require_env("SUPABASE_ANON_KEY")
    client: Client = create_client(url, anon_key)
    client.postgrest.auth(access_token)
    return client


def build_service_role_client():
    """Build a Supabase client using the service-role key, which BYPASSES
    Row Level Security entirely. Reserved for the narrow set of operations
    documented in the Phase 4A specification as genuinely requiring
    privileged backend authority. Never expose this client, or the key it
    wraps, to Streamlit UI code, a CrewAI agent, a log line, a test
    fixture printout, or any artifact.
    """
    from supabase import Client, create_client

    url = _require_env("SUPABASE_URL")
    service_role_key = _require_env("SUPABASE_SERVICE_ROLE_KEY")
    client: Client = create_client(url, service_role_key)
    return client
