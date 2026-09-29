"""Reusable Supabase Auth session/identity foundation (Phase 4B.1).

This module is deliberately Streamlit-independent -- it never reads or writes
st.session_state, and is safe to call from any context, including a later
background thread that was handed a captured token explicitly (see the
Phase 4B Obsidian note's background-thread invariant: background workflow
threads must never read st.session_state directly for credentials). The
actual Streamlit login-form cutover is later, separate, gated work
(Phase 4B.2) -- this module only builds the pieces that cutover will call.

Every function here operates exclusively through the anon-key (sign-in) or
user-scoped (build_user_scoped_client, reused from supabase_client.py) client
paths already established in Phase 4A. Nothing here ever reads, requires, or
constructs a client from the privileged backend-only service-role secret --
see supabase_client.build_service_role_client for that narrow, unrelated
path (this module never imports or calls it).

The application role (Admin/Analyst/Scientist/Business) is always resolved
from the authenticated user's own `profiles` row via RLS (`id = auth.uid()`
filtered explicitly -- see the module docstring note below for why the
filter is mandatory), never trusted from client/browser input and never
read from the GoTrue User object's own `role`/`app_metadata`/`user_metadata`
fields, which describe the Postgres/provider identity, not this
application's role.
"""

from __future__ import annotations

from dataclasses import dataclass

from supabase import Client, create_client

from src.persistence.interfaces import PersistenceError
from src.persistence.supabase_client import _require_env

# Mirrors the exact CHECK constraint in
# supabase/migrations/20260903005542_phase4a_base_schema.sql -- do not accept
# any role value not in this set, regardless of what a profile row claims.
VALID_APPLICATION_ROLES = frozenset({"Admin", "Analyst", "Scientist", "Business"})


class AuthenticationError(PersistenceError):
    """Raised for any Supabase Auth / profile-resolution failure.

    Fail-closed: callers must treat this as a denial and must never catch it
    to silently fall back to a different authentication mechanism (see the
    Phase 4B owner decision: a Supabase Auth denial must remain a denial).
    Messages here only ever carry a safe classification label (see
    classify_auth_error), never raw provider text, tokens, or passwords.
    """


def classify_auth_error(exc: BaseException) -> str:
    """Best-effort classification of a Supabase Auth/PostgREST exception into
    one of: 'auth_failed', 'rls_denied', 'privilege_denied',
    'connectivity_error', 'unknown'. Used so callers can distinguish "wrong
    credentials" from "network blip" from "RLS denied the profile lookup"
    without ever needing to inspect/print the raw exception."""
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


@dataclass(frozen=True)
class AuthenticatedIdentity:
    """Minimal, Streamlit-independent representation of a signed-in
    application user. Never carries a password. Carries both tokens because
    a later caller (Phase 4B.2, or a background thread via explicit
    capture) needs access_token to build a user-scoped Supabase client
    (supabase_client.build_user_scoped_client) so RLS enforces auth.uid()
    correctly, and refresh_token to call refresh()/sign_out() later without
    re-prompting for credentials.
    """

    user_id: str
    email: str | None
    role: str
    access_token: str
    refresh_token: str
    expires_at: int | None

    def __repr__(self) -> str:  # pragma: no cover - defensive redaction only
        return (
            f"AuthenticatedIdentity(user_id={self.user_id!r}, email={self.email!r}, "
            f"role={self.role!r}, access_token='<redacted>', refresh_token='<redacted>', "
            f"expires_at={self.expires_at!r})"
        )

    __str__ = __repr__


def _anon_client() -> Client:
    """Build a plain anon-key client with no session yet -- the only client
    shape allowed for an initial sign-in (there is no token to scope to
    until sign-in succeeds). Reuses the same env-var contract as
    supabase_client.py's factories (_require_env) rather than duplicating
    the missing-variable error message."""
    url = _require_env("SUPABASE_URL")
    anon_key = _require_env("SUPABASE_ANON_KEY")
    return create_client(url, anon_key)


def _resolve_role(client: Client, user_id: str) -> str:
    """Fail-closed application-role resolution. Always filters by the
    caller's own user_id explicitly -- an Admin's `profiles` SELECT policy
    (`id = auth.uid() OR is_admin()`) returns every profile row when
    unfiltered, so relying on "the first row" would be unsafe (this exact
    mistake was found and fixed during Phase 4A Step 1D live acceptance)."""
    try:
        response = client.table("profiles").select("role").eq("id", user_id).execute()
    except Exception as exc:  # noqa: BLE001 - normalize to AuthenticationError, never leak provider internals
        raise AuthenticationError(f"profile lookup failed ({classify_auth_error(exc)})") from exc

    rows = getattr(response, "data", None) or []
    if len(rows) != 1:
        raise AuthenticationError(
            f"expected exactly one profile row for the authenticated user, found {len(rows)} "
            "-- refusing to authenticate"
        )
    role = rows[0].get("role")
    if role not in VALID_APPLICATION_ROLES:
        raise AuthenticationError(
            f"profile role {role!r} is not a recognized application role -- refusing to authenticate"
        )
    return role


def _identity_from_auth_response(client: Client, auth_response) -> AuthenticatedIdentity:
    session = getattr(auth_response, "session", None)
    user = getattr(auth_response, "user", None)
    if session is None or user is None:
        raise AuthenticationError("authentication response had no session/user -- refusing to authenticate")

    role = _resolve_role(client, user.id)
    return AuthenticatedIdentity(
        user_id=user.id,
        email=getattr(user, "email", None),
        role=role,
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_at=getattr(session, "expires_at", None),
    )


def sign_in(email: str, password: str) -> AuthenticatedIdentity:
    """Authenticate with Supabase Auth (email/password) and resolve the
    caller's application role. Raises AuthenticationError on any failure --
    wrong credentials, malformed response, missing/invalid profile, or an
    unsupported role. Never falls back to any other authentication path."""
    client = _anon_client()
    try:
        auth_response = client.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as exc:  # noqa: BLE001 - normalize to AuthenticationError, never leak provider internals
        raise AuthenticationError(f"sign-in failed ({classify_auth_error(exc)})") from exc

    # The same client that just signed in is already correctly scoped for
    # the profile lookup below -- Client wires an on_auth_state_change
    # listener (see supabase/_sync/client.py) that re-authenticates
    # postgrest on SIGNED_IN, so no separate build_user_scoped_client call
    # is needed here.
    return _identity_from_auth_response(client, auth_response)


def refresh(refresh_token: str) -> AuthenticatedIdentity:
    """Refresh an expired/expiring session and re-resolve the application
    role (in case it changed since the original sign-in). Raises
    AuthenticationError on any failure."""
    client = _anon_client()
    try:
        auth_response = client.auth.refresh_session(refresh_token)
    except Exception as exc:  # noqa: BLE001
        raise AuthenticationError(f"session refresh failed ({classify_auth_error(exc)})") from exc
    return _identity_from_auth_response(client, auth_response)


def resolve_identity_from_tokens(access_token: str, refresh_token: str) -> AuthenticatedIdentity:
    """Rehydrate an AuthenticatedIdentity from an already-known token pair
    (e.g. restoring a session across a later caller/rerun without
    re-prompting for credentials) and re-resolve the application role.
    Raises AuthenticationError if the tokens are invalid/expired or the
    profile cannot be resolved."""
    client = _anon_client()
    try:
        auth_response = client.auth.set_session(access_token, refresh_token)
    except Exception as exc:  # noqa: BLE001
        raise AuthenticationError(f"session restoration failed ({classify_auth_error(exc)})") from exc
    return _identity_from_auth_response(client, auth_response)


def sign_out(access_token: str, refresh_token: str) -> None:
    """Invalidate the given session with Supabase Auth. Raises
    AuthenticationError on failure -- callers decide separately whether to
    still clear local session state on a failed remote sign-out (a
    Streamlit-layer decision, not this module's)."""
    client = _anon_client()
    try:
        client.auth.set_session(access_token, refresh_token)
        client.auth.sign_out()
    except Exception as exc:  # noqa: BLE001
        raise AuthenticationError(f"sign-out failed ({classify_auth_error(exc)})") from exc
