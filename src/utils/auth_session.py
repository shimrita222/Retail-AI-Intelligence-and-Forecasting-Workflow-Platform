"""Streamlit session/mode helpers for the Supabase Auth cutover (Phase 4B.2).

Flat-key compatibility adapter (Phase 4B.2 plan §6): Supabase mode writes the
SAME five session_state keys src/utils/auth.py's legacy login already uses
(authenticated, username, name, role, permissions), plus a small set of
Supabase-only supplemental keys legacy mode never writes/reads. This means
current_user()/has_permission() in src/utils/auth.py are reused unmodified
by both modes -- this module never duplicates that logic, and never
introduces a nested st.session_state["auth"] structure (an earlier, rejected
design).

Permissions are ALWAYS derived from the caller-supplied config["roles"]
mapping and NEVER trusted from a stored value -- not just at login/refresh,
but on every successful restoration of an already-fresh session (no network
call needed for that: the role is already trusted for this rerun under the
existing staleness bound, §8a of the plan; only the permissions list is
re-derived, which is pure local computation). A role with no configured
permissions entry fails closed (AuthenticationError) and clears state
rather than continuing with a stale or placeholder list.

Every replacement authentication record is fully built and validated
BEFORE any session_state key is written (see _build_supabase_record /
_apply_record). Once construction succeeds, applying it is a sequence of
plain dict writes that cannot themselves raise -- so a partial/half-updated
record cannot result from a successful replacement. This is correctness
atomicity (no failure can occur between the first and last key write), not
multi-threaded transactional atomicity; it relies on Streamlit's
single-threaded-per-rerun script execution model, which already governs
every other session_state mutation in this app.

This module intentionally does not render any Streamlit UI (no login form,
no logout button) -- that is Phase 4B.2B/C, separate gated work. It only
provides the pure session-state helpers: resolving which auth mode is
active, writing a successful Supabase identity into session state,
restoring/refreshing an existing session on rerun, and clearing all
auth-sensitive state. None of these functions call into src.utils.auth's
credential-verification path (verify_credentials/hash_password), and none
import supabase_client.build_service_role_client.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import streamlit as st

from src.persistence.interfaces import PersistenceError
from src.persistence.supabase_auth import AuthenticatedIdentity, AuthenticationError, refresh, sign_in, sign_out
from src.utils.auth import SESSION_KEYS, current_user, logout

VALID_AUTH_MODES = ("supabase", "legacy")

# Supabase-only supplemental keys (never written/read by legacy mode).
SUPPLEMENTAL_KEYS = (
    "auth_mode",
    "supabase_user_id",
    "supabase_email",
    "access_token",
    "refresh_token",
    "token_expires_at",
)

# Session-state clock-skew allowance before a Supabase access token is
# treated as needing refresh (plan §7b).
_EXPIRY_SKEW_SECONDS = 60


class AuthModeError(Exception):
    """Raised by resolve_auth_mode() for a missing/invalid AUTH_MODE. Callers
    must fail closed -- never guess a mode or fall back to legacy."""


def resolve_auth_mode(env: dict[str, str] | None = None) -> str:
    """Resolve the explicit, configuration-driven auth mode. Reads AUTH_MODE
    from the given mapping (defaults to os.environ). Missing/empty ->
    'supabase' (the target system). Any value outside VALID_AUTH_MODES ->
    AuthModeError (fail closed; no login form should be rendered by the
    caller in that case)."""
    source = env if env is not None else os.environ
    raw = source.get("AUTH_MODE")
    if raw is None or raw == "":
        return "supabase"
    if raw not in VALID_AUTH_MODES:
        raise AuthModeError(
            f"AUTH_MODE={raw!r} is not one of {VALID_AUTH_MODES!r} -- refusing to resolve an auth mode"
        )
    return raw


def _is_finite_number(value: Any) -> bool:
    """True only for a real, finite int/float -- rejects bool (int subclass),
    NaN, +/-Infinity, and any non-numeric type, without raising."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _resolve_permissions(config: dict, role: str) -> list:
    """Derive the permissions list for `role` from config["roles"], the
    same non-secret mapping src.utils.auth.verify_credentials() already
    uses for legacy logins. Fails closed -- never returns a placeholder or
    partial list -- if the role is unrecognized or config is malformed."""
    roles_map = config.get("roles") if isinstance(config, dict) else None
    if not isinstance(roles_map, dict):
        raise AuthenticationError("config['roles'] is missing or malformed -- refusing to authenticate")
    permissions = roles_map.get(role)
    if not isinstance(permissions, list):
        raise AuthenticationError(
            f"role {role!r} has no configured permissions -- refusing to authenticate"
        )
    return list(permissions)  # defensive copy -- never share config's own list object


def _build_supabase_record(identity: AuthenticatedIdentity, config: dict) -> dict:
    """Build and fully validate the complete replacement session record
    BEFORE anything is written to session_state. Raises AuthenticationError
    -- with session_state completely untouched -- if any identity field is
    missing/malformed or the role has no configured permissions."""
    if not isinstance(identity.user_id, str) or identity.user_id == "":
        raise AuthenticationError("identity.user_id missing/invalid -- refusing to authenticate")
    if not isinstance(identity.role, str) or identity.role == "":
        raise AuthenticationError("identity.role missing/invalid -- refusing to authenticate")
    if not isinstance(identity.access_token, str) or identity.access_token == "":
        raise AuthenticationError("identity.access_token missing/invalid -- refusing to authenticate")
    if not isinstance(identity.refresh_token, str) or identity.refresh_token == "":
        raise AuthenticationError("identity.refresh_token missing/invalid -- refusing to authenticate")
    if not _is_finite_number(identity.expires_at):
        raise AuthenticationError("identity.expires_at missing/invalid (non-finite) -- refusing to authenticate")

    permissions = _resolve_permissions(config, identity.role)  # raises on an unsupported/misconfigured role

    return {
        "authenticated": True,
        "username": identity.user_id,
        "name": identity.email or identity.user_id,
        "role": identity.role,
        "permissions": permissions,
        "auth_mode": "supabase",
        "supabase_user_id": identity.user_id,
        "supabase_email": identity.email,
        "access_token": identity.access_token,
        "refresh_token": identity.refresh_token,
        "token_expires_at": identity.expires_at,
    }


def _apply_record(state: Any, record: dict) -> None:
    """Write an already-fully-validated record. Pure assignments only --
    nothing here can raise, which is what makes 'no stale field survives a
    successful replacement' actually true: every key in SUPPLEMENTAL_KEYS
    plus the five shared keys is present in `record` and gets overwritten,
    so no key from a previous session can be left behind by this call."""
    for key, value in record.items():
        state[key] = value


def supabase_login_success(identity: AuthenticatedIdentity, config: dict, session_state: Any = None) -> None:
    """Write a successful Supabase sign-in (or a successful refresh -- see
    restore_session) into session state using the flat-key compatibility
    adapter. Builds and validates the complete record first
    (_build_supabase_record); only then writes it (_apply_record). If
    validation fails, session_state is left exactly as it was -- never
    stores the password (it was never passed to this function) or the raw
    AuthenticatedIdentity object."""
    record = _build_supabase_record(identity, config)  # raises before any write
    state = session_state if session_state is not None else st.session_state
    _apply_record(state, record)


def clear_all_auth_state(session_state: Any = None) -> None:
    """Clear every auth-sensitive key for both modes.

    When called with no argument (the real app path), reuses the existing,
    unmodified src.utils.auth.logout() against the real st.session_state --
    preserving today's legacy logout behaviour exactly.

    When called with an explicit `session_state` mapping (used by every
    pure-function test in this module, since logout() only ever operates on
    the real st.session_state singleton and would silently no-op against an
    injected test dict), this replicates logout()'s exact effect on that
    mapping: pop the five SESSION_KEYS, then restore the identical defaults
    auth.py's _init_session_state() sets (authenticated=False, username/
    name/role=None, permissions=[]).

    Either way, the Supabase-only supplemental keys and active_run_id are
    then also popped. No auth-sensitive key survives this call in either
    mode, on either state object."""
    if session_state is None:
        logout()  # unmodified src.utils.auth.logout() on the real st.session_state
        state = st.session_state
    else:
        state = session_state
        for key in SESSION_KEYS:
            state.pop(key, None)
        # Exactly mirrors auth.py's _init_session_state() defaults.
        state["authenticated"] = False
        state["username"] = None
        state["name"] = None
        state["role"] = None
        state["permissions"] = []

    for key in SUPPLEMENTAL_KEYS:
        state.pop(key, None)
    state.pop("active_run_id", None)


def _legacy_fields_valid(state: Any) -> bool:
    """Type/value validation for a legacy-mode session, so restore_session
    never raises on a malformed/tampered record -- it fails closed instead.
    (Legacy semantics are owned by the unmodified src/utils/auth.py; this
    is only a defensive pre-check before trusting the stored values.)"""
    username = state.get("username")
    name = state.get("name")
    role = state.get("role")
    permissions = state.get("permissions")
    return (
        isinstance(username, str) and username != ""
        and isinstance(name, str)
        and isinstance(role, str) and role != ""
        and isinstance(permissions, list)
    )


def _supabase_fields_valid(state: Any) -> bool:
    """Type/value/consistency validation for a Supabase-mode session before
    trusting it. Rejects malformed types, a non-finite token_expires_at
    (including NaN/+-Infinity), and a username/supabase_user_id mismatch
    (the two are supposed to be the same identifier under this adapter --
    see _build_supabase_record -- so divergence means the record was
    tampered with or corrupted). Does NOT validate `permissions` here:
    restore_session always re-derives permissions from config before
    trusting them, so a stale/wrong stored value is corrected rather than
    merely rejected."""
    username = state.get("username")
    role = state.get("role")
    supabase_user_id = state.get("supabase_user_id")
    access_token = state.get("access_token")
    refresh_token = state.get("refresh_token")
    token_expires_at = state.get("token_expires_at")

    if not (isinstance(username, str) and username != ""):
        return False
    if not (isinstance(role, str) and role != ""):
        return False
    if not (isinstance(supabase_user_id, str) and supabase_user_id != ""):
        return False
    if username != supabase_user_id:
        return False
    if not (isinstance(access_token, str) and access_token != ""):
        return False
    if not (isinstance(refresh_token, str) and refresh_token != ""):
        return False
    if not _is_finite_number(token_expires_at):
        return False
    return True


def restore_session(auth_mode_resolved: str, config: dict, session_state: Any = None) -> bool:
    """Decision tree from Phase 4B.2 plan §7b (as strengthened by prior
    owner corrections). Returns True if the session is valid and the
    caller may proceed past the login gate, False if the caller must
    render a login form -- session_state has already been cleared in
    every False-returning branch except the plain "no session yet" case.

    Never raises on malformed/tampered session content -- every field is
    validated by type, value, and cross-field consistency before use;
    anything unexpected clears state and returns False. Does not render
    any UI."""
    state = session_state if session_state is not None else st.session_state

    authenticated = state.get("authenticated", False)
    if not isinstance(authenticated, bool):
        # A malformed non-boolean value (e.g. "True", 1, a list) is treated
        # as a corrupted/tampered record, not as "logged out" -- clearing
        # prevents any stale credential fields that might accompany it from
        # surviving under a falsy-but-truthy read.
        clear_all_auth_state(state)
        return False
    if not authenticated:
        return False  # legitimate "no session yet" -- nothing to clear

    stored_mode = state.get("auth_mode")
    if not isinstance(stored_mode, str) or stored_mode not in VALID_AUTH_MODES or stored_mode != auth_mode_resolved:
        clear_all_auth_state(state)
        return False

    if auth_mode_resolved == "legacy":
        if not _legacy_fields_valid(state):
            clear_all_auth_state(state)
            return False
        return True

    # auth_mode_resolved == "supabase"
    if not _supabase_fields_valid(state):
        clear_all_auth_state(state)
        return False

    now = time.time()
    still_fresh = now < state["token_expires_at"] - _EXPIRY_SKEW_SECONDS  # safe: already validated finite numeric

    if still_fresh:
        # No network call needed -- the role is already trusted for this
        # rerun under the existing staleness bound (plan §8a). But
        # permissions must still reflect the CURRENT config["roles"]
        # mapping, not whatever was last written, in case config changed
        # since login/last refresh. Pure local computation.
        try:
            permissions = _resolve_permissions(config, state["role"])
        except AuthenticationError:
            clear_all_auth_state(state)
            return False
        state["permissions"] = permissions
        return True

    try:
        identity = refresh(state["refresh_token"])
        supabase_login_success(identity, config, state)  # build-then-apply; replaces every field, no leftovers
    except AuthenticationError:
        clear_all_auth_state(state)
        return False

    return True


def session_mode_is_consistent(auth_mode_resolved: str, session_state: Any = None) -> bool:
    """Narrow, network-free consistency gate run once at the top of the
    Streamlit login gate, before either login branch (Phase 4B.2B plan
    §16.1a). Two independent fail-closed cases:

    1. Authenticated session: returns False (caller must clear) if
       `authenticated` is a malformed non-bool, or if authenticated is
       True and `auth_mode` is missing, not one of VALID_AUTH_MODES, or
       does not equal auth_mode_resolved. True only if an existing
       authenticated session's auth_mode already matches.
    2. Unauthenticated session: returns False (caller must clear) if
       `authenticated` is falsy but any SUPPLEMENTAL_KEYS entry is still
       present -- a contaminated, not-genuinely-fresh session (e.g.
       leftover Supabase tokens from a session cleared through some path
       other than clear_all_auth_state()). True only when authenticated
       is falsy AND no supplemental key is present at all -- the one
       case this plan treats as genuinely fresh.

    Deliberately a small subset of restore_session()'s own branches A/B
    logic, reimplemented standalone rather than by calling
    restore_session() itself, which would also pull in its expiry/
    refresh branch (D) -- explicitly deferred to Phase 4B.2C. Performs
    no network call; inspects only key presence/type."""
    state = session_state if session_state is not None else st.session_state
    authenticated = state.get("authenticated", False)
    if not isinstance(authenticated, bool):
        return False
    if not authenticated:
        return not any(key in state for key in SUPPLEMENTAL_KEYS)
    stored_mode = state.get("auth_mode")
    return isinstance(stored_mode, str) and stored_mode in VALID_AUTH_MODES and stored_mode == auth_mode_resolved


def mark_legacy_login_if_fresh(was_authenticated_before: bool, session_state: Any = None) -> None:
    """Tag the session as legacy-mode ONLY when this call just transitioned
    the session from unauthenticated to authenticated -- a genuine, fresh
    legacy login via src.utils.auth.render_login_form()/login() -- never
    merely because the session was already authenticated going into this
    call. Idempotent no-op otherwise. Never writes to src.utils.auth's own
    state; only ever sets this module's own auth_mode key."""
    state = session_state if session_state is not None else st.session_state
    just_authenticated = (not was_authenticated_before) and state.get("authenticated") is True
    if just_authenticated:
        state["auth_mode"] = "legacy"


def process_supabase_login(email: str, password: str, config: dict, session_state: Any = None) -> bool:
    """Pure, UI-free Supabase login attempt.

    Idempotent short-circuit (mirrors render_login_form()'s own
    "already authenticated" contract for legacy mode): if the session is
    already authenticated AND consistent with Supabase mode
    (session_mode_is_consistent() returns True while `authenticated` is
    exactly True), returns True immediately WITHOUT calling sign_in()
    again and WITHOUT touching any existing identity/token field -- a
    second call with different (or even wrong) credentials never
    re-authenticates, overwrites, or clears a valid existing session
    merely because this function ran again.

    Otherwise: any inconsistent/contaminated session (malformed
    `authenticated`, an authenticated session with a missing/invalid/
    mismatched auth_mode, or an unauthenticated session still carrying a
    leftover Supabase artifact) is cleared via clear_all_auth_state()
    BEFORE a real sign_in() attempt is made -- so this function remains
    safe even if a future caller forgets to run the gate-level check
    first. A genuinely fresh, non-contaminated unauthenticated session is
    left untouched and a real login is attempted.

    Returns False on any AuthenticationError (bad credentials, missing/
    invalid profile, unsupported role, connectivity failure, etc.)
    without re-raising. Never imports or calls into src.utils.auth's
    credential-verification path (verify_credentials/hash_password/
    login) -- a failed Supabase login must never fall back to legacy
    authentication (plan §5)."""
    state = session_state if session_state is not None else st.session_state
    consistent = session_mode_is_consistent("supabase", state)

    if consistent and state.get("authenticated") is True:
        return True  # already a valid, mode-consistent Supabase session -- no re-login, nothing touched

    if not consistent:
        clear_all_auth_state(state)

    try:
        identity = sign_in(email, password)
        supabase_login_success(identity, config, state)
    except AuthenticationError:
        return False
    return True


def render_supabase_login_form(config: dict) -> bool:
    """Render the Supabase-mode login form in the current Streamlit
    context. Mirrors src.utils.auth.render_login_form()'s exact contract:
    returns True once authenticated (either just now or from a prior
    rerun), False while still showing the form.

    Uses session_mode_is_consistent() (already implemented, Phase
    4B.2B-i) rather than a bare bool(authenticated) check -- so an
    authenticated session with a missing/invalid/mismatched auth_mode,
    or an unauthenticated session still carrying a leftover Supabase
    artifact, is cleared via clear_all_auth_state() BEFORE the form
    renders, never silently trusted or shown stale (plan §16.1a).
    Short-circuits (returns True without rendering the form) only when
    the session is both authenticated is exactly True AND
    mode-consistent.

    Delegates the actual login attempt to process_supabase_login(),
    which already short-circuits a valid, mode-consistent existing
    session without calling sign_in() again. This function adds nothing
    but rendering: it never calls supabase_auth.sign_in()/
    supabase_login_success() directly, and never imports anything from
    src.utils.auth's credential-verification path (verify_credentials/
    hash_password/login) -- a failed Supabase login must never fall
    back to legacy authentication (plan §5).

    On failure, shows one fixed, generic error string -- never the
    underlying AuthenticationError's classification label, never
    provider text, never a credential-existence hint, never the entered
    password."""
    consistent = session_mode_is_consistent("supabase", st.session_state)
    if consistent and st.session_state.get("authenticated") is True:
        return True
    if not consistent:
        clear_all_auth_state()

    if st.session_state.pop("logout_notice", False):
        st.info("Signed out locally; the server session may still be active.")

    st.title("Retail AI Intelligence & Forecasting Workflow Platform")
    st.subheader("Sign in")

    with st.form("supabase_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if not email or not password:
            st.error("Enter your email and password.")
        elif process_supabase_login(email, password, config):
            st.rerun()
        else:
            st.error("Sign-in failed. Check your credentials and try again.")

    return bool(st.session_state.get("authenticated", False))


def render_supabase_logout_button() -> None:
    """Render the Supabase-mode logout control in the current Streamlit
    context. Mirrors src.utils.auth.render_logout_button()'s shape
    exactly (sidebar, shows the signed-in user's name/role via the
    shared, unmodified current_user(), a single "Log out" button), but
    clears the session via clear_all_auth_state() instead of
    src.utils.auth.logout() -- the latter only clears the five shared
    keys, not the six Supabase-only supplemental keys (plan §16.4).

    Attempts a remote supabase_auth.sign_out() first, while the tokens
    are still present -- a failure (AuthenticationError) is swallowed
    without being re-raised, logged, or displayed (never exposes the raw
    provider message or any token), and NEVER prevents the local clear
    below, which always runs regardless of the remote outcome (plan
    §7c). The optional one-shot "signed out locally" notice for a failed
    remote sign-out is Phase 4B.2C-iii's work, not this function's yet."""
    user = current_user()
    if user is None:
        return
    with st.sidebar:
        st.markdown(f"**{user['name']}**  \nRole: `{user['role']}`")
        if st.button("Log out"):
            access_token = st.session_state.get("access_token")
            refresh_token = st.session_state.get("refresh_token")
            if access_token and refresh_token:
                try:
                    sign_out(access_token, refresh_token)
                except PersistenceError:
                    # remote sign-out failed -- local clear below still proceeds regardless
                    # (plan §7c); set the one-shot notice flag render_supabase_login_form()
                    # will show exactly once on the next render. Never log or expose the
                    # underlying exception's message or any token.
                    st.session_state["logout_notice"] = True
            clear_all_auth_state()
            st.rerun()
