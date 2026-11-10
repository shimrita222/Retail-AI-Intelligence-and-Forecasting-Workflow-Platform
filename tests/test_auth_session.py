"""Phase 4B.2A/4B.2B tests for src/utils/auth_session.py -- the flat-key
compatibility adapter and session helpers (resolve_auth_mode,
supabase_login_success, restore_session, clear_all_auth_state,
session_mode_is_consistent, mark_legacy_login_if_fresh,
process_supabase_login, render_supabase_login_form,
render_supabase_logout_button), plus the app.py authentication gate.

Fakes only: AuthenticatedIdentity is instantiated directly (it is a plain
frozen dataclass); sign_in()/refresh() are monkeypatched at their import
site inside auth_session.py. No live Supabase connection, no STAGING.

Two testing styles are used:
- Pure-function tests (session_mode_is_consistent, mark_legacy_login_if_fresh,
  process_supabase_login, and everything from 4B.2A): session_state is an
  explicit injected dict, exactly as every pure function in auth_session.py
  supports -- no Streamlit runtime needed.
- streamlit.testing.v1.AppTest (render_supabase_login_form,
  render_supabase_logout_button, and the app.py gate): these functions call
  st.* directly with no injectable session_state, so they are driven
  end-to-end via AppTest, which runs the script in-process (no server, no
  network) -- monkeypatches applied before .run()/.click() correctly affect
  the script. The app.py gate tests extract the gate block verbatim from the
  real app.py (see _extract_gate_source()) rather than maintaining a
  hand-typed duplicate.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.utils.auth_session as auth_session
from src.persistence.interfaces import PersistenceError
from src.persistence.supabase_auth import AuthenticatedIdentity, AuthenticationError
from src.utils.auth import SESSION_KEYS
from src.utils.auth_session import (
    SUPPLEMENTAL_KEYS,
    AuthModeError,
    _is_finite_number,
    clear_all_auth_state,
    mark_legacy_login_if_fresh,
    process_supabase_login,
    resolve_auth_mode,
    restore_session,
    session_mode_is_consistent,
    supabase_login_success,
)

VALID_ACCESS_TOKEN = "fake-access-token"
VALID_REFRESH_TOKEN = "fake-refresh-token"
VALID_USER_ID = "11111111-1111-1111-1111-111111111111"

CONFIG = {
    "roles": {
        "Admin": ["run_workflow", "view_descriptive", "view_predictive", "view_logs"],
        "Analyst": ["view_descriptive"],
        "Scientist": ["view_predictive"],
        "Business": ["view_descriptive", "view_predictive"],
    }
}


def make_identity(
    user_id=VALID_USER_ID,
    email="admin@example.com",
    role="Admin",
    access_token=VALID_ACCESS_TOKEN,
    refresh_token=VALID_REFRESH_TOKEN,
    expires_at=9999999999,
):
    return AuthenticatedIdentity(
        user_id=user_id,
        email=email,
        role=role,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
    )


def make_supabase_session(role="Admin", expires_at=9999999999, **overrides):
    """A session_state dict shaped exactly like supabase_login_success()
    would produce, for tests that start from an already-authenticated
    state (restoration / refresh tests)."""
    state = {
        "authenticated": True,
        "username": VALID_USER_ID,
        "name": "admin@example.com",
        "role": role,
        "permissions": CONFIG["roles"][role],
        "auth_mode": "supabase",
        "supabase_user_id": VALID_USER_ID,
        "supabase_email": "admin@example.com",
        "access_token": VALID_ACCESS_TOKEN,
        "refresh_token": VALID_REFRESH_TOKEN,
        "token_expires_at": expires_at,
    }
    state.update(overrides)
    return state


def make_legacy_session(role="Admin", **overrides):
    """Shaped exactly like src.utils.auth.login() would produce -- the
    five shared keys only, no supplemental keys."""
    state = {
        "authenticated": True,
        "username": "admin",
        "name": "Admin User",
        "role": role,
        "permissions": CONFIG["roles"][role],
    }
    state.update(overrides)
    return state


def _refresh_must_not_be_called(*_args, **_kwargs):
    raise AssertionError("supabase_auth.refresh() must not be called for a still-fresh session")


def _assert_fully_cleared(state):
    """Shared assertion for every fail-closed path: no Supabase-only key
    survives (tokens, supabase_user_id/email, auth_mode, token_expires_at)
    and the five shared keys are reset to auth.py's own logged-out
    defaults -- never left holding a stale credential, role, or
    permissions list from the rejected session."""
    assert state["authenticated"] is False
    assert state["username"] is None
    assert state["name"] is None
    assert state["role"] is None
    assert state["permissions"] == []
    for key in SUPPLEMENTAL_KEYS:
        assert key not in state, f"{key} must not survive a fail-closed clear"
    assert "access_token" not in state
    assert "refresh_token" not in state
    assert "active_run_id" not in state


def _assert_no_leaked_auth_details(at, *forbidden_substrings):
    """Shared assertion for logout error paths: none of the given forbidden
    substrings (tokens, raw provider error text) appear in any user-visible
    Streamlit element rendered by this AppTest run -- error, warning, info,
    or sidebar markdown. A vacuous pass (nothing rendered at all) is itself
    the correct, intended outcome for a failed remote sign-out, which shows
    nothing beyond setting the one-shot notice flag for the next render."""
    rendered_texts = (
        [e.value for e in at.error]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
        + [m.value for m in at.sidebar.markdown]
    )
    for forbidden in forbidden_substrings:
        assert not any(forbidden in text for text in rendered_texts), f"{forbidden!r} leaked into UI output"


def _session_state_value(at, key, default=None):
    """Safe read of a possibly-unset AppTest session_state value.
    at.session_state does NOT support .get() -- verified empirically
    (it raises AttributeError, since this test-harness-facing object
    only supports `in` / `[...]` indexing). The real st.session_state
    used by application code DOES support .get() (also verified) -- this
    helper works around a test-side accessor limitation only, not a
    defect in the already-approved auth_session.py code."""
    return at.session_state[key] if key in at.session_state else default


# ---------------------------------------------------------------------------
# Phase 4B.2B-ii/iii fixtures: AppTest scripts for render_supabase_login_form(),
# render_supabase_logout_button(), and the app.py authentication gate.
# ---------------------------------------------------------------------------

_LOGIN_FORM_SCRIPT = (
    "import streamlit as st\n"
    "from src.utils.auth_session import render_supabase_login_form\n"
    "CONFIG = {'roles': {'Admin': ['run_workflow', 'view_descriptive', 'view_predictive', 'view_logs'],\n"
    "                     'Analyst': ['view_descriptive']}}\n"
    "render_supabase_login_form(CONFIG)\n"
)

_LOGOUT_BUTTON_SCRIPT = (
    "import streamlit as st\n"
    "from src.utils.auth_session import render_supabase_logout_button\n"
    "render_supabase_logout_button()\n"
)


def _extract_gate_source() -> str:
    """Extract the authentication-gate block verbatim from the real
    app.py, bounded by its own section-header comment and the next
    section's header comment -- never hand-duplicated, never importing
    or executing the rest of app.py (whose module-level code below the
    gate touches ARTIFACTS_DIR/RAW_DIR/run_registry filesystem state
    unrelated to authentication). If app.py's gate section is ever
    restructured such that these two header comments no longer exist,
    this fails loudly at collection time rather than silently testing
    stale/wrong code."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    lines = app_path.read_text(encoding="utf-8").splitlines()

    header_idx = next(i for i, l in enumerate(lines) if "Login gate (done here" in l)
    next_section_idx = next(i for i, l in enumerate(lines) if "Background workflow execution" in l)

    start = header_idx + 1
    while start < len(lines) and (lines[start].strip() == "" or lines[start].strip().startswith("#")):
        start += 1

    end = next_section_idx
    while end > start and (lines[end - 1].strip() == "" or lines[end - 1].strip().startswith("#")):
        end -= 1

    gate_source = "\n".join(lines[start:end]).strip() + "\n"
    assert "resolve_auth_mode" in gate_source, "gate extraction markers no longer match app.py's structure"
    return gate_source


_EXTRACTED_GATE_SOURCE = _extract_gate_source()

_GATE_SCRIPT = (
    "import streamlit as st\n"
    "from src.utils.auth import load_config, render_logout_button\n"
    "CONFIG = load_config()\n"
) + _EXTRACTED_GATE_SOURCE

_TEST_LEGACY_PASSWORD_HASH = hashlib.sha256(b"test-only-password").hexdigest()
_GATE_SCRIPT_WITH_TEST_LEGACY_USER = (
    "import streamlit as st\n"
    "from src.utils.auth import render_logout_button\n"
    "CONFIG = {\n"
    "    'roles': {'Admin': ['run_workflow']},\n"
    "    'credentials': {'usernames': {'testuser': {\n"
    f"        'name': 'Test User', 'role': 'Admin', 'password_hash': '{_TEST_LEGACY_PASSWORD_HASH}'\n"
    "    }}},\n"
    "}\n"
) + _EXTRACTED_GATE_SOURCE


# ---------------------------------------------------------------------------
# resolve_auth_mode() -- defaults and invalid values
# ---------------------------------------------------------------------------


def test_resolve_auth_mode_defaults_to_supabase_when_unset():
    assert resolve_auth_mode(env={}) == "supabase"


def test_resolve_auth_mode_defaults_to_supabase_when_empty_string():
    assert resolve_auth_mode(env={"AUTH_MODE": ""}) == "supabase"


def test_resolve_auth_mode_accepts_legacy():
    assert resolve_auth_mode(env={"AUTH_MODE": "legacy"}) == "legacy"


def test_resolve_auth_mode_accepts_supabase_explicitly():
    assert resolve_auth_mode(env={"AUTH_MODE": "supabase"}) == "supabase"


@pytest.mark.parametrize("bad_value", ["Legacy", "SUPABASE", "both", "none", " supabase"])
def test_resolve_auth_mode_invalid_value_fails_closed(bad_value):
    with pytest.raises(AuthModeError):
        resolve_auth_mode(env={"AUTH_MODE": bad_value})


# ---------------------------------------------------------------------------
# supabase_login_success() -- successful construction + permissions
# ---------------------------------------------------------------------------


def test_supabase_login_success_writes_flat_keys_and_derives_permissions():
    state = {}
    identity = make_identity(role="Analyst")
    supabase_login_success(identity, CONFIG, state)

    assert state["authenticated"] is True
    assert state["username"] == VALID_USER_ID
    assert state["name"] == "admin@example.com"
    assert state["role"] == "Analyst"
    assert state["permissions"] == CONFIG["roles"]["Analyst"]
    assert state["auth_mode"] == "supabase"
    assert state["supabase_user_id"] == VALID_USER_ID
    assert state["supabase_email"] == "admin@example.com"
    assert state["access_token"] == VALID_ACCESS_TOKEN
    assert state["refresh_token"] == VALID_REFRESH_TOKEN
    assert state["token_expires_at"] == 9999999999
    # Exactly the 5 shared + 6 supplemental keys -- no nested "auth" dict.
    assert set(state.keys()) == set(SESSION_KEYS) | set(SUPPLEMENTAL_KEYS)


def test_supabase_login_success_email_none_falls_back_to_user_id_for_name():
    state = {}
    identity = make_identity(email=None)
    supabase_login_success(identity, CONFIG, state)
    assert state["name"] == VALID_USER_ID
    assert state["supabase_email"] is None


def test_supabase_login_success_never_stores_a_password_field():
    state = {}
    supabase_login_success(make_identity(), CONFIG, state)
    assert "password" not in state
    assert not any("password" in str(k).lower() for k in state)


# ---------------------------------------------------------------------------
# Unsupported roles / malformed identities fail closed, no partial write
# ---------------------------------------------------------------------------


def test_supabase_login_success_unsupported_role_fails_closed_with_no_write():
    state = {}
    identity = make_identity(role="SuperAdmin")  # not in CONFIG["roles"]
    with pytest.raises(AuthenticationError):
        supabase_login_success(identity, CONFIG, state)
    assert state == {}  # nothing was written -- build-then-apply held


def test_supabase_login_success_malformed_config_fails_closed():
    state = {}
    with pytest.raises(AuthenticationError):
        supabase_login_success(make_identity(), {"roles": "not-a-dict"}, state)
    assert state == {}


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("user_id", ""),
        ("user_id", None),
        ("role", ""),
        ("role", None),
        ("access_token", ""),
        ("access_token", None),
        ("refresh_token", ""),
        ("refresh_token", None),
    ],
)
def test_supabase_login_success_malformed_identity_field_fails_closed_with_no_write(field, bad_value):
    state = {"preexisting": "value"}
    identity = make_identity(**{field: bad_value})
    with pytest.raises(AuthenticationError):
        supabase_login_success(identity, CONFIG, state)
    assert state == {"preexisting": "value"}  # untouched, not even partially


@pytest.mark.parametrize("bad_expiry", [float("nan"), float("inf"), float("-inf"), "9999999999", None])
def test_supabase_login_success_rejects_non_finite_or_non_numeric_expiry(bad_expiry):
    state = {}
    identity = make_identity(expires_at=bad_expiry)
    with pytest.raises(AuthenticationError):
        supabase_login_success(identity, CONFIG, state)
    assert state == {}


def test_is_finite_number_rejects_bool_and_nan_and_infinity():
    assert _is_finite_number(True) is False
    assert _is_finite_number(False) is False
    assert _is_finite_number(float("nan")) is False
    assert _is_finite_number(float("inf")) is False
    assert _is_finite_number(float("-inf")) is False
    assert _is_finite_number("123") is False
    assert _is_finite_number(123) is True
    assert _is_finite_number(123.5) is True


# ---------------------------------------------------------------------------
# restore_session() -- malformed `authenticated`
# ---------------------------------------------------------------------------


def test_restore_session_no_session_shows_login_without_clearing():
    state = {}
    assert restore_session("supabase", CONFIG, state) is False
    assert state == {}  # nothing to clear, nothing added


def test_restore_session_authenticated_false_is_the_legitimate_not_logged_in_case():
    state = {"authenticated": False}
    assert restore_session("supabase", CONFIG, state) is False
    assert state == {"authenticated": False}  # untouched -- not treated as malformed


@pytest.mark.parametrize("malformed_authenticated", ["true", 1, 0, None, [], {}])
def test_restore_session_malformed_authenticated_clears_state(malformed_authenticated):
    state = make_supabase_session()
    state["authenticated"] = malformed_authenticated
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


# ---------------------------------------------------------------------------
# restore_session() -- missing / malformed / mismatched auth_mode
# ---------------------------------------------------------------------------


def test_restore_session_missing_auth_mode_on_authenticated_session_fails_closed():
    state = make_supabase_session()
    del state["auth_mode"]
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


@pytest.mark.parametrize("bad_mode", ["Supabase", "both", "", 1, None])
def test_restore_session_malformed_auth_mode_fails_closed(bad_mode):
    state = make_supabase_session()
    state["auth_mode"] = bad_mode
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


def test_restore_session_mode_mismatch_supabase_session_under_legacy_resolved_mode():
    state = make_supabase_session()
    assert restore_session("legacy", CONFIG, state) is False
    _assert_fully_cleared(state)


def test_restore_session_mode_mismatch_legacy_session_under_supabase_resolved_mode():
    state = make_legacy_session()
    state["auth_mode"] = "legacy"
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


# ---------------------------------------------------------------------------
# restore_session() -- legacy mode (compatibility preserved)
# ---------------------------------------------------------------------------


def test_legacy_session_keys_unchanged():
    # Proxy for "src/utils/auth.py is untouched": the exact shared-key
    # contract auth_session.py relies on has not drifted.
    assert SESSION_KEYS == ("authenticated", "username", "name", "role", "permissions")


def test_restore_session_valid_legacy_session_continues_unmodified():
    state = make_legacy_session(role="Business")
    state["auth_mode"] = "legacy"
    before = dict(state)
    assert restore_session("legacy", CONFIG, state) is True
    assert state == before  # legacy path makes no changes -- auth.py owns this behavior


@pytest.mark.parametrize(
    "field,bad_value",
    [("username", ""), ("username", None), ("role", ""), ("permissions", "not-a-list"), ("permissions", None)],
)
def test_restore_session_malformed_legacy_fields_fail_closed(field, bad_value):
    state = make_legacy_session()
    state["auth_mode"] = "legacy"
    state[field] = bad_value
    assert restore_session("legacy", CONFIG, state) is False
    _assert_fully_cleared(state)


# ---------------------------------------------------------------------------
# restore_session() -- supabase: username / supabase_user_id consistency
# ---------------------------------------------------------------------------


def test_restore_session_username_supabase_user_id_mismatch_fails_closed():
    state = make_supabase_session()
    state["supabase_user_id"] = "a-different-user-id"
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


# ---------------------------------------------------------------------------
# restore_session() -- supabase: malformed / non-finite token_expires_at
# and other malformed fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_expiry", [float("nan"), float("inf"), float("-inf"), "9999999999", None, True])
def test_restore_session_rejects_non_finite_token_expires_at(bad_expiry):
    state = make_supabase_session()
    state["token_expires_at"] = bad_expiry
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("access_token", ""),
        ("access_token", None),
        ("refresh_token", ""),
        ("refresh_token", None),
        ("role", ""),
        ("role", None),
    ],
)
def test_restore_session_malformed_supabase_fields_fail_closed(field, bad_value):
    state = make_supabase_session()
    state[field] = bad_value
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


# ---------------------------------------------------------------------------
# restore_session() -- valid unexpired session: no provider network call
# ---------------------------------------------------------------------------


def test_restore_session_still_fresh_makes_no_refresh_call(monkeypatch):
    call_count = {"n": 0}

    def spy(*_args, **_kwargs):
        call_count["n"] += 1
        raise AssertionError("refresh() must not be called for a still-fresh session")

    monkeypatch.setattr("src.utils.auth_session.refresh", spy)
    state = make_supabase_session(expires_at=9999999999)  # far future
    assert restore_session("supabase", CONFIG, state) is True
    assert call_count["n"] == 0


def test_restore_session_still_fresh_rederives_permissions_from_config(monkeypatch):
    monkeypatch.setattr("src.utils.auth_session.refresh", _refresh_must_not_be_called)
    state = make_supabase_session(role="Analyst", expires_at=9999999999)
    state["permissions"] = ["stale-permission-that-should-be-overwritten"]
    assert restore_session("supabase", CONFIG, state) is True
    assert state["permissions"] == CONFIG["roles"]["Analyst"]


def test_restore_session_still_fresh_role_no_longer_in_config_fails_closed(monkeypatch):
    monkeypatch.setattr("src.utils.auth_session.refresh", _refresh_must_not_be_called)
    state = make_supabase_session(role="Analyst", expires_at=9999999999)
    config_without_analyst = {"roles": {"Admin": CONFIG["roles"]["Admin"]}}
    assert restore_session("supabase", config_without_analyst, state) is False
    _assert_fully_cleared(state)


# ---------------------------------------------------------------------------
# restore_session() -- expired access token: refresh success / failure
# ---------------------------------------------------------------------------


def test_restore_session_expired_refresh_success_replaces_fields_and_updates_role(monkeypatch):
    new_identity = make_identity(
        role="Scientist",
        access_token="new-access-token",
        refresh_token="new-refresh-token",
        expires_at=9999999999,
    )
    monkeypatch.setattr("src.utils.auth_session.refresh", lambda token: new_identity)

    state = make_supabase_session(role="Admin", expires_at=1)  # already expired
    assert restore_session("supabase", CONFIG, state) is True

    assert state["role"] == "Scientist"
    assert state["permissions"] == CONFIG["roles"]["Scientist"]
    assert state["access_token"] == "new-access-token"
    assert state["refresh_token"] == "new-refresh-token"
    assert state["token_expires_at"] == 9999999999
    assert state["access_token"] != VALID_ACCESS_TOKEN  # no leftover from the old record
    assert state["refresh_token"] != VALID_REFRESH_TOKEN


def test_restore_session_expired_refresh_failure_clears_state(monkeypatch):
    def raise_auth_error(token):
        raise AuthenticationError("refresh failed (auth_failed)")

    monkeypatch.setattr("src.utils.auth_session.refresh", raise_auth_error)
    state = make_supabase_session(expires_at=1)
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


def test_restore_session_expired_refresh_returns_unsupported_role_clears_state(monkeypatch):
    identity_with_bad_role = make_identity(role="SuperAdmin")
    monkeypatch.setattr("src.utils.auth_session.refresh", lambda token: identity_with_bad_role)
    state = make_supabase_session(expires_at=1)
    assert restore_session("supabase", CONFIG, state) is False
    _assert_fully_cleared(state)


# ---------------------------------------------------------------------------
# clear_all_auth_state() -- shared + supplemental keys
# ---------------------------------------------------------------------------


def test_clear_all_auth_state_clears_supabase_session_completely():
    state = make_supabase_session()
    clear_all_auth_state(state)
    _assert_fully_cleared(state)


def test_clear_all_auth_state_clears_active_run_id():
    state = make_supabase_session()
    state["active_run_id"] = "20260101T000000Z-deadbeef"
    clear_all_auth_state(state)
    assert "active_run_id" not in state


def test_clear_all_auth_state_on_legacy_session_matches_auth_py_defaults():
    state = make_legacy_session()
    clear_all_auth_state(state)
    assert state == {
        "authenticated": False,
        "username": None,
        "name": None,
        "role": None,
        "permissions": [],
    }


def test_clear_all_auth_state_is_idempotent_on_already_clear_state():
    state = {}
    clear_all_auth_state(state)  # must not raise
    assert state["authenticated"] is False


# ---------------------------------------------------------------------------
# No service-role dependency, no token exposure
# ---------------------------------------------------------------------------


def test_module_never_references_service_role_credential():
    import ast

    import src.utils.auth_session as module

    # A whole-source substring/line-prefix check is unreliable here: the
    # module's own docstring deliberately *mentions*
    # build_service_role_client in prose (to document that it is never
    # imported), and that sentence happens to wrap with the word "import"
    # starting its own line -- which would false-positive both a raw
    # substring check and a "line startswith 'import '" heuristic. Parse
    # the actual AST and look only at real ast.Import/ast.ImportFrom
    # nodes, which cannot appear inside a string literal/docstring.
    assert not hasattr(module, "build_service_role_client")
    assert "SUPABASE_SERVICE_ROLE_KEY" not in vars(module)

    tree = ast.parse(inspect.getsource(module))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported_names.update(alias.asname or alias.name for alias in node.names)
    assert "build_service_role_client" not in imported_names
    assert "SUPABASE_SERVICE_ROLE_KEY" not in imported_names


def test_failure_messages_never_contain_token_values():
    state = {}
    identity = make_identity(role="SuperAdmin")  # triggers AuthenticationError
    try:
        supabase_login_success(identity, CONFIG, state)
    except AuthenticationError as exc:
        message = str(exc)
        assert VALID_ACCESS_TOKEN not in message
        assert VALID_REFRESH_TOKEN not in message
    else:
        pytest.fail("expected AuthenticationError")


# ===========================================================================
# Phase 4B.2B-i: session_mode_is_consistent()
# ===========================================================================


@pytest.mark.parametrize("state", [{}, {"authenticated": False}])
def test_session_mode_is_consistent_true_for_fresh_unauthenticated_session(state):
    assert session_mode_is_consistent("supabase", state) is True
    assert session_mode_is_consistent("legacy", state) is True


def test_session_mode_is_consistent_true_for_valid_authenticated_supabase_session():
    state = make_supabase_session()
    assert session_mode_is_consistent("supabase", state) is True


def test_session_mode_is_consistent_true_for_valid_authenticated_legacy_session():
    state = make_legacy_session()
    state["auth_mode"] = "legacy"
    assert session_mode_is_consistent("legacy", state) is True


@pytest.mark.parametrize(
    "state_factory,resolved",
    [
        (lambda: make_supabase_session(), "legacy"),
        (lambda: {**make_legacy_session(), "auth_mode": "legacy"}, "supabase"),
    ],
)
def test_session_mode_is_consistent_false_for_mode_mismatch(state_factory, resolved):
    assert session_mode_is_consistent(resolved, state_factory()) is False


@pytest.mark.parametrize("malformed_authenticated", ["true", 1, 0, None, [], {}])
def test_session_mode_is_consistent_false_for_malformed_authenticated(malformed_authenticated):
    state = make_supabase_session()
    state["authenticated"] = malformed_authenticated
    assert session_mode_is_consistent("supabase", state) is False


@pytest.mark.parametrize("key", list(SUPPLEMENTAL_KEYS) + ["__all__"])
def test_session_mode_is_consistent_false_for_contaminated_unauthenticated_session(key):
    state = {"authenticated": False}
    if key == "__all__":
        for k in SUPPLEMENTAL_KEYS:
            state[k] = "leftover"
    else:
        state[key] = "leftover"
    assert session_mode_is_consistent("supabase", state) is False
    assert session_mode_is_consistent("legacy", state) is False  # rejected regardless of target mode


# ===========================================================================
# Phase 4B.2B-i: mark_legacy_login_if_fresh()
# ===========================================================================


def test_mark_legacy_login_if_fresh_tags_on_genuine_fresh_login():
    state = make_legacy_session()
    assert "auth_mode" not in state
    mark_legacy_login_if_fresh(was_authenticated_before=False, session_state=state)
    assert state["auth_mode"] == "legacy"


def test_mark_legacy_login_if_fresh_does_not_retag_existing_session():
    state = make_legacy_session()  # authenticated=True, no auth_mode key
    mark_legacy_login_if_fresh(was_authenticated_before=True, session_state=state)
    assert "auth_mode" not in state  # never added merely because the session was already authenticated


def test_mark_legacy_login_if_fresh_failed_login_does_not_mark_auth_mode():
    state = {"authenticated": False}  # login attempt failed, still unauthenticated
    mark_legacy_login_if_fresh(was_authenticated_before=False, session_state=state)
    assert "auth_mode" not in state


# ===========================================================================
# Phase 4B.2B-i: process_supabase_login()
# ===========================================================================


def test_process_supabase_login_success_writes_record(monkeypatch):
    fake_identity = make_identity(role="Analyst")
    monkeypatch.setattr("src.utils.auth_session.sign_in", lambda email, password: fake_identity)
    state = {}
    result = process_supabase_login("a@example.com", "pw", CONFIG, state)
    assert result is True
    assert state["role"] == "Analyst"
    assert state["permissions"] == CONFIG["roles"]["Analyst"]


def test_process_supabase_login_failure_returns_false_state_unchanged(monkeypatch):
    def _raise(*_a, **_k):
        raise AuthenticationError("sign-in failed (auth_failed)")

    monkeypatch.setattr("src.utils.auth_session.sign_in", _raise)
    state = {}
    result = process_supabase_login("a@example.com", "wrong", CONFIG, state)
    assert result is False
    assert state == {}


def test_process_supabase_login_contaminated_session_cleared_even_on_failure(monkeypatch):
    def _raise(*_a, **_k):
        raise AuthenticationError("sign-in failed (auth_failed)")

    monkeypatch.setattr("src.utils.auth_session.sign_in", _raise)
    state = {"authenticated": False, "access_token": "stale-token"}  # contaminated
    result = process_supabase_login("a@example.com", "wrong", CONFIG, state)
    assert result is False
    _assert_fully_cleared(state)  # cleared despite the subsequent sign_in failure


def test_process_supabase_login_mode_mismatch_cleared_before_attempt(monkeypatch):
    fake_identity = make_identity(role="Admin")
    monkeypatch.setattr("src.utils.auth_session.sign_in", lambda email, password: fake_identity)
    state = make_legacy_session()
    state["auth_mode"] = "legacy"  # authenticated under the wrong mode for this call
    result = process_supabase_login("a@example.com", "pw", CONFIG, state)
    assert result is True
    assert state["auth_mode"] == "supabase"
    assert state["username"] == VALID_USER_ID  # the new supabase identity, not the old legacy one


def test_process_supabase_login_existing_valid_session_short_circuits_without_sign_in(monkeypatch):
    def _must_not_be_called(*_a, **_k):
        raise AssertionError("sign_in() must not be called for an already-authenticated, mode-consistent session")

    monkeypatch.setattr("src.utils.auth_session.sign_in", _must_not_be_called)
    state = make_supabase_session(role="Admin")
    before = dict(state)
    result = process_supabase_login("someone-else@example.com", "wrong-password", CONFIG, state)
    assert result is True
    assert state == before


# ===========================================================================
# Phase 4B.2B-ii: render_supabase_login_form()
# ===========================================================================


def test_render_supabase_login_form_existing_valid_session_short_circuits(monkeypatch):
    def _must_not_be_called(*_a, **_k):
        raise AssertionError("sign_in() must not be called for an already-authenticated session")

    monkeypatch.setattr("src.utils.auth_session.sign_in", _must_not_be_called)

    at = AppTest.from_string(_LOGIN_FORM_SCRIPT)
    for key, value in make_supabase_session().items():
        at.session_state[key] = value
    at.run()

    assert len(at.text_input) == 0  # form never rendered
    assert at.session_state["access_token"] == VALID_ACCESS_TOKEN  # untouched


def test_render_supabase_login_form_fresh_successful_login(monkeypatch):
    fake_identity = make_identity(role="Admin")
    monkeypatch.setattr("src.utils.auth_session.sign_in", lambda email, password: fake_identity)

    at = AppTest.from_string(_LOGIN_FORM_SCRIPT)
    at.run()
    at.text_input[0].input("admin@example.com")
    at.text_input[1].input("correct-password")
    at.button[0].click()
    at.run()

    assert at.session_state["authenticated"] is True
    assert at.session_state["role"] == "Admin"
    assert len(at.error) == 0


def test_render_supabase_login_form_fresh_failed_login(monkeypatch):
    def _raise(*_a, **_k):
        raise AuthenticationError("sign-in failed (auth_failed)")

    monkeypatch.setattr("src.utils.auth_session.sign_in", _raise)

    at = AppTest.from_string(_LOGIN_FORM_SCRIPT)
    at.run()
    at.text_input[0].input("admin@example.com")
    at.text_input[1].input("wrong-password")
    at.button[0].click()
    at.run()

    assert at.error[0].value == "Sign-in failed. Check your credentials and try again."
    assert _session_state_value(at, "authenticated", False) is False


def test_render_supabase_login_form_blank_credentials_rejected_without_sign_in_call(monkeypatch):
    def _must_not_be_called(*_a, **_k):
        raise AssertionError("sign_in() must not be called for a blank submission")

    monkeypatch.setattr("src.utils.auth_session.sign_in", _must_not_be_called)

    at = AppTest.from_string(_LOGIN_FORM_SCRIPT)
    at.run()
    at.button[0].click()  # submit with both fields left blank
    at.run()

    assert at.error[0].value == "Enter your email and password."


def test_render_supabase_login_form_inconsistent_session_cleared_before_rendering():
    at = AppTest.from_string(_LOGIN_FORM_SCRIPT)
    legacy_shaped = make_legacy_session()
    legacy_shaped["auth_mode"] = "legacy"  # authenticated, but wrong mode for this (supabase) form
    for key, value in legacy_shaped.items():
        at.session_state[key] = value
    at.run()

    assert len(at.text_input) == 2  # cleared -> form renders fresh
    assert at.session_state["authenticated"] is False


@pytest.mark.parametrize(
    "failure_message",
    ["sign-in failed (auth_failed)", "profile lookup failed (connectivity_error)", "sign-in failed (unknown)"],
)
def test_render_supabase_login_form_error_message_never_leaks_provider_details(monkeypatch, failure_message):
    def _raise(*_a, **_k):
        raise AuthenticationError(failure_message)

    monkeypatch.setattr("src.utils.auth_session.sign_in", _raise)

    at = AppTest.from_string(_LOGIN_FORM_SCRIPT)
    at.run()
    at.text_input[0].input("a@example.com")
    at.text_input[1].input("pw")
    at.button[0].click()
    at.run()

    assert at.error[0].value == "Sign-in failed. Check your credentials and try again."
    assert failure_message not in at.error[0].value


# ===========================================================================
# Phase 4B.2B-ii: render_supabase_logout_button()
# ===========================================================================


def test_render_supabase_logout_button_displays_authenticated_user():
    at = AppTest.from_string(_LOGOUT_BUTTON_SCRIPT)
    for key, value in make_supabase_session(role="Admin").items():
        at.session_state[key] = value
    at.run()

    assert len(at.sidebar.markdown) == 1
    assert at.sidebar.markdown[0].value == "**admin@example.com**  \nRole: `Admin`"


def test_render_supabase_logout_button_click_clears_all_auth_keys_and_active_run_id():
    at = AppTest.from_string(_LOGOUT_BUTTON_SCRIPT)
    seed = make_supabase_session()
    seed["active_run_id"] = "20260101T000000Z-deadbeef"
    for key, value in seed.items():
        at.session_state[key] = value
    at.run()
    at.sidebar.button[0].click()
    at.run()

    assert at.session_state["authenticated"] is False
    for key in SUPPLEMENTAL_KEYS:
        assert key not in at.session_state
    assert "active_run_id" not in at.session_state


def test_render_supabase_logout_button_module_imports_sign_out_from_supabase_auth():
    # Structural proof (AST-based, same approach as the service-role-credential
    # check) that sign_out IS now imported, from the correct module -- updated
    # for the approved Phase 4B.2C-ii remote sign-out wiring; supersedes the
    # earlier 4B.2B-era assertion that it must NOT be imported.
    import ast

    tree = ast.parse(inspect.getsource(auth_session))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.persistence.supabase_auth":
            names = {alias.asname or alias.name for alias in node.names}
            if "sign_out" in names:
                return
    pytest.fail("sign_out must be imported from src.persistence.supabase_auth")


def test_render_supabase_logout_button_unauthenticated_session_renders_nothing():
    at = AppTest.from_string(_LOGOUT_BUTTON_SCRIPT)
    at.run()  # no session seeded -- unauthenticated

    assert len(at.sidebar.markdown) == 0
    assert len(at.sidebar.button) == 0


# ===========================================================================
# Phase 4B.2B-iii: app.py authentication gate (extracted source, not duplicated)
# ===========================================================================


def test_gate_supabase_is_the_default_when_auth_mode_unset(monkeypatch):
    monkeypatch.delenv("AUTH_MODE", raising=False)
    at = AppTest.from_string(_GATE_SCRIPT)
    at.run()
    assert at.text_input[0].label == "Email"  # the Supabase form, not legacy's "Username"


def test_gate_explicit_legacy_mode_uses_legacy_form(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "legacy")
    at = AppTest.from_string(_GATE_SCRIPT)
    at.run()
    assert at.text_input[0].label == "Username"


def test_gate_invalid_auth_mode_fails_closed_no_form_rendered(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "bogus")
    at = AppTest.from_string(_GATE_SCRIPT)
    at.run()
    assert at.error[0].value == "Invalid AUTH_MODE configuration. Contact an administrator."
    assert len(at.text_input) == 0


def test_gate_session_mismatch_cleared_before_branching(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "legacy")
    at = AppTest.from_string(_GATE_SCRIPT)
    for key, value in make_supabase_session().items():  # stale supabase-mode session
        at.session_state[key] = value
    at.run()
    assert at.session_state["authenticated"] is False
    assert at.text_input[0].label == "Username"  # fresh legacy form, not a relabeled supabase session


def test_gate_no_automatic_fallback_from_supabase_failure_to_legacy(monkeypatch):
    monkeypatch.delenv("AUTH_MODE", raising=False)  # defaults to supabase

    def _raise(*_a, **_k):
        raise AuthenticationError("sign-in failed (auth_failed)")

    monkeypatch.setattr("src.utils.auth_session.sign_in", _raise)

    def _verify_must_not_be_called(*_a, **_k):
        raise AssertionError("legacy verify_credentials() must not be called after a Supabase failure")

    monkeypatch.setattr("src.utils.auth.verify_credentials", _verify_must_not_be_called)

    at = AppTest.from_string(_GATE_SCRIPT)
    at.run()
    at.text_input[0].input("a@example.com")
    at.text_input[1].input("wrong-password")
    at.button[0].click()
    at.run()

    assert at.error[0].value == "Sign-in failed. Check your credentials and try again."


def test_gate_legacy_auth_mode_tagging_only_on_fresh_successful_login(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "legacy")
    at = AppTest.from_string(_GATE_SCRIPT_WITH_TEST_LEGACY_USER)
    at.run()
    at.text_input[0].input("testuser")
    at.text_input[1].input("test-only-password")
    at.button[0].click()
    at.run()

    assert at.session_state["authenticated"] is True
    assert at.session_state["auth_mode"] == "legacy"


# ===========================================================================
# Phase 4B.2C-i: app.py gate -- restore_session() wiring
# ===========================================================================


def test_gate_fresh_unauthenticated_session_never_calls_restore_session(monkeypatch):
    def _must_not_be_called(*_a, **_k):
        raise AssertionError("restore_session() must not be called for a never-authenticated session")

    monkeypatch.setattr("src.utils.auth_session.restore_session", _must_not_be_called)
    monkeypatch.delenv("AUTH_MODE", raising=False)
    at = AppTest.from_string(_GATE_SCRIPT)
    at.run()
    assert at.text_input[0].label == "Email"


def test_gate_valid_unexpired_session_continues_without_refresh_call(monkeypatch):
    def _refresh_must_not_be_called(*_a, **_k):
        raise AssertionError("refresh() must not be called for a still-fresh session")

    monkeypatch.setattr("src.utils.auth_session.refresh", _refresh_must_not_be_called)
    monkeypatch.delenv("AUTH_MODE", raising=False)
    at = AppTest.from_string(_GATE_SCRIPT)
    for key, value in make_supabase_session(expires_at=9999999999).items():
        at.session_state[key] = value
    at.run()
    assert len(at.warning) == 0
    assert at.session_state["authenticated"] is True
    assert len(at.sidebar.markdown) == 1  # logout button rendered, not a fresh login form


def test_gate_expired_refreshable_session_refreshes_and_continues(monkeypatch):
    fake_identity = make_identity(role="Scientist", access_token="new-token", refresh_token="new-refresh")
    monkeypatch.setattr("src.utils.auth_session.refresh", lambda token: fake_identity)
    monkeypatch.delenv("AUTH_MODE", raising=False)
    at = AppTest.from_string(_GATE_SCRIPT)
    for key, value in make_supabase_session(role="Admin", expires_at=1).items():  # already expired
        at.session_state[key] = value
    at.run()
    assert len(at.warning) == 0
    assert at.session_state["authenticated"] is True
    assert at.session_state["role"] == "Scientist"
    assert at.session_state["access_token"] == "new-token"


def test_gate_failed_refresh_clears_state_and_shows_expected_warning(monkeypatch):
    def _raise(*_a, **_k):
        raise AuthenticationError("session refresh failed (auth_failed)")

    monkeypatch.setattr("src.utils.auth_session.refresh", _raise)
    monkeypatch.delenv("AUTH_MODE", raising=False)
    at = AppTest.from_string(_GATE_SCRIPT)
    for key, value in make_supabase_session(expires_at=1).items():  # already expired
        at.session_state[key] = value
    at.run()
    assert at.warning[0].value == "Your session expired. Please sign in again."
    _assert_fully_cleared(at.session_state)
    assert len(at.text_input) == 2  # fresh login form rendered


def test_gate_legacy_mode_never_calls_restore_session(monkeypatch):
    def _must_not_be_called(*_a, **_k):
        raise AssertionError("restore_session() must not be called in legacy mode")

    monkeypatch.setattr("src.utils.auth_session.restore_session", _must_not_be_called)
    monkeypatch.setenv("AUTH_MODE", "legacy")
    at = AppTest.from_string(_GATE_SCRIPT)
    at.run()
    assert at.text_input[0].label == "Username"


# ===========================================================================
# Phase 4B.2C-ii: remote sign_out() wiring
# ===========================================================================


def test_render_supabase_logout_button_calls_sign_out_with_correct_tokens(monkeypatch):
    captured = {}

    def _spy(access_token, refresh_token):
        captured["access_token"] = access_token
        captured["refresh_token"] = refresh_token

    monkeypatch.setattr("src.utils.auth_session.sign_out", _spy)
    at = AppTest.from_string(_LOGOUT_BUTTON_SCRIPT)
    for key, value in make_supabase_session().items():
        at.session_state[key] = value
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert captured == {"access_token": VALID_ACCESS_TOKEN, "refresh_token": VALID_REFRESH_TOKEN}


def test_render_supabase_logout_button_successful_remote_sign_out_clears_locally_no_notice(monkeypatch):
    monkeypatch.setattr("src.utils.auth_session.sign_out", lambda *_a, **_k: None)
    at = AppTest.from_string(_LOGOUT_BUTTON_SCRIPT)
    for key, value in make_supabase_session().items():
        at.session_state[key] = value
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert at.session_state["authenticated"] is False
    for key in SUPPLEMENTAL_KEYS:
        assert key not in at.session_state
    assert "logout_notice" not in at.session_state


def test_render_supabase_logout_button_authentication_error_still_clears_and_sets_notice(monkeypatch):
    failure_message = "sign-out failed (auth_failed)"

    def _raise(*_a, **_k):
        raise AuthenticationError(failure_message)

    monkeypatch.setattr("src.utils.auth_session.sign_out", _raise)
    at = AppTest.from_string(_LOGOUT_BUTTON_SCRIPT)
    for key, value in make_supabase_session().items():
        at.session_state[key] = value
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert at.session_state["authenticated"] is False
    for key in SUPPLEMENTAL_KEYS:
        assert key not in at.session_state
    assert at.session_state["logout_notice"] is True
    _assert_no_leaked_auth_details(at, VALID_ACCESS_TOKEN, VALID_REFRESH_TOKEN, failure_message)


def test_render_supabase_logout_button_bare_persistence_error_still_clears_and_sets_notice(monkeypatch):
    failure_message = "required environment variable SUPABASE_URL is not set"

    def _raise(*_a, **_k):
        raise PersistenceError(failure_message)

    monkeypatch.setattr("src.utils.auth_session.sign_out", _raise)
    at = AppTest.from_string(_LOGOUT_BUTTON_SCRIPT)
    for key, value in make_supabase_session().items():
        at.session_state[key] = value
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert at.session_state["authenticated"] is False
    for key in SUPPLEMENTAL_KEYS:
        assert key not in at.session_state
    assert at.session_state["logout_notice"] is True
    _assert_no_leaked_auth_details(at, VALID_ACCESS_TOKEN, VALID_REFRESH_TOKEN, failure_message)


def test_render_supabase_logout_button_missing_tokens_skips_remote_call_but_clears_locally(monkeypatch):
    def _must_not_be_called(*_a, **_k):
        raise AssertionError("sign_out() must not be called when tokens are missing")

    monkeypatch.setattr("src.utils.auth_session.sign_out", _must_not_be_called)
    at = AppTest.from_string(_LOGOUT_BUTTON_SCRIPT)
    seed = make_supabase_session()
    seed["access_token"] = None
    seed["refresh_token"] = None
    for key, value in seed.items():
        at.session_state[key] = value
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert at.session_state["authenticated"] is False


# ===========================================================================
# Phase 4B.2C-iii: one-shot local sign-out notice
# ===========================================================================


def test_render_supabase_login_form_shows_and_consumes_logout_notice():
    at = AppTest.from_string(_LOGIN_FORM_SCRIPT)
    at.session_state["logout_notice"] = True
    at.run()
    assert at.info[0].value == "Signed out locally; the server session may still be active."
    assert "logout_notice" not in at.session_state  # consumed -- one-shot


def test_render_supabase_login_form_no_notice_when_flag_absent():
    at = AppTest.from_string(_LOGIN_FORM_SCRIPT)
    at.run()
    assert len(at.info) == 0
