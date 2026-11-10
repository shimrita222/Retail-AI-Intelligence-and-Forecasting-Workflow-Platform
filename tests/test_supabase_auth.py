"""Phase 4B.1 tests for the reusable Supabase Auth foundation
(src/persistence/supabase_auth.py). Exercised entirely against fake in-memory
auth/postgrest clients -- no live Supabase connection, no real credentials,
no STAGING mutation.
"""

from __future__ import annotations

import pytest

from src.persistence.supabase_auth import (
    VALID_APPLICATION_ROLES,
    AuthenticatedIdentity,
    AuthenticationError,
    classify_auth_error,
    refresh,
    resolve_identity_from_tokens,
    sign_in,
    sign_out,
)

VALID_ACCESS_TOKEN = "fake-access-token"
VALID_REFRESH_TOKEN = "fake-refresh-token"
VALID_USER_ID = "11111111-1111-1111-1111-111111111111"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, access_token=VALID_ACCESS_TOKEN, refresh_token=VALID_REFRESH_TOKEN, expires_at=9999999999):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at


class _FakeUser:
    def __init__(self, id=VALID_USER_ID, email="admin@example.com"):
        self.id = id
        self.email = email


class _FakeAuthResponse:
    def __init__(self, session, user):
        self.session = session
        self.user = user


class _AuthApiError(Exception):
    """Stands in for supabase_auth's AuthApiError -- classify_auth_error
    matches on the class name, so this fake's name matters."""


class _FakeProfilesQuery:
    def __init__(self, rows):
        self._rows = rows
        self._filters: dict[str, object] = {}

    def select(self, *_args):
        return self

    def eq(self, column, value):
        self._filters[column] = value
        return self

    def execute(self):
        filtered = [r for r in self._rows if all(r.get(k) == v for k, v in self._filters.items())]
        return type("Resp", (), {"data": filtered})()


class _FakeTable:
    def __init__(self, name, profiles_rows):
        self._name = name
        self._profiles_rows = profiles_rows

    def select(self, *args):
        assert self._name == "profiles"
        return _FakeProfilesQuery(self._profiles_rows).select(*args)


class _FakeAuth:
    def __init__(
        self,
        sign_in_result=None,
        sign_in_error: Exception | None = None,
        refresh_result=None,
        refresh_error: Exception | None = None,
        set_session_result=None,
        set_session_error: Exception | None = None,
        sign_out_error: Exception | None = None,
    ):
        self._sign_in_result = sign_in_result
        self._sign_in_error = sign_in_error
        self._refresh_result = refresh_result
        self._refresh_error = refresh_error
        self._set_session_result = set_session_result
        self._set_session_error = set_session_error
        self._sign_out_error = sign_out_error
        self.sign_out_called = False

    def sign_in_with_password(self, _credentials):
        if self._sign_in_error is not None:
            raise self._sign_in_error
        return self._sign_in_result

    def refresh_session(self, _refresh_token):
        if self._refresh_error is not None:
            raise self._refresh_error
        return self._refresh_result

    def set_session(self, _access_token, _refresh_token):
        if self._set_session_error is not None:
            raise self._set_session_error
        return self._set_session_result

    def sign_out(self):
        self.sign_out_called = True
        if self._sign_out_error is not None:
            raise self._sign_out_error


class _FakeClient:
    def __init__(self, auth: _FakeAuth, profiles_rows):
        self.auth = auth
        self._profiles_rows = profiles_rows

    def table(self, name):
        return _FakeTable(name, self._profiles_rows)


def _patch_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "fake-anon-key")


def _patch_client_factory(monkeypatch, client: _FakeClient):
    monkeypatch.setattr("src.persistence.supabase_auth.create_client", lambda *_a, **_kw: client)


# ---------------------------------------------------------------------------
# sign_in
# ---------------------------------------------------------------------------


def test_sign_in_success_normalizes_identity(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(), _FakeUser())
    client = _FakeClient(_FakeAuth(sign_in_result=auth_response), profiles_rows=[{"id": VALID_USER_ID, "role": "Admin"}])
    _patch_client_factory(monkeypatch, client)

    identity = sign_in("admin@example.com", "correct-password")

    assert isinstance(identity, AuthenticatedIdentity)
    assert identity.user_id == VALID_USER_ID
    assert identity.email == "admin@example.com"
    assert identity.role == "Admin"
    assert identity.access_token == VALID_ACCESS_TOKEN
    assert identity.refresh_token == VALID_REFRESH_TOKEN


def test_sign_in_failed_login_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    client = _FakeClient(
        _FakeAuth(sign_in_error=_AuthApiError("Invalid login credentials")),
        profiles_rows=[],
    )
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError) as excinfo:
        sign_in("admin@example.com", "wrong-password")
    assert "auth_failed" in str(excinfo.value)


def test_sign_in_malformed_response_no_session_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(session=None, user=_FakeUser())
    client = _FakeClient(_FakeAuth(sign_in_result=auth_response), profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError):
        sign_in("admin@example.com", "correct-password")


def test_sign_in_missing_user_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(session=_FakeSession(), user=None)
    client = _FakeClient(_FakeAuth(sign_in_result=auth_response), profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError):
        sign_in("admin@example.com", "correct-password")


def test_sign_in_network_error_classified(monkeypatch):
    _patch_env(monkeypatch)
    client = _FakeClient(_FakeAuth(sign_in_error=Exception("Connection timed out")), profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError) as excinfo:
        sign_in("admin@example.com", "correct-password")
    assert "connectivity_error" in str(excinfo.value)


# ---------------------------------------------------------------------------
# profile / role resolution
# ---------------------------------------------------------------------------


def test_sign_in_missing_profile_row_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(), _FakeUser())
    client = _FakeClient(_FakeAuth(sign_in_result=auth_response), profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError, match="found 0"):
        sign_in("admin@example.com", "correct-password")


def test_sign_in_multiple_profile_rows_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(), _FakeUser())
    # Simulates the unfiltered-Admin-select hazard being defended against --
    # even if two rows somehow matched, this must still fail closed.
    client = _FakeClient(
        _FakeAuth(sign_in_result=auth_response),
        profiles_rows=[{"id": VALID_USER_ID, "role": "Admin"}, {"id": VALID_USER_ID, "role": "Admin"}],
    )
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError, match="found 2"):
        sign_in("admin@example.com", "correct-password")


def test_sign_in_unsupported_role_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(), _FakeUser())
    client = _FakeClient(
        _FakeAuth(sign_in_result=auth_response),
        profiles_rows=[{"id": VALID_USER_ID, "role": "SuperAdmin"}],
    )
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError, match="not a recognized application role"):
        sign_in("admin@example.com", "correct-password")


def test_sign_in_malformed_role_none_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(), _FakeUser())
    client = _FakeClient(
        _FakeAuth(sign_in_result=auth_response),
        profiles_rows=[{"id": VALID_USER_ID, "role": None}],
    )
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError):
        sign_in("admin@example.com", "correct-password")


@pytest.mark.parametrize("role", sorted(VALID_APPLICATION_ROLES))
def test_sign_in_accepts_every_valid_role(monkeypatch, role):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(), _FakeUser())
    client = _FakeClient(
        _FakeAuth(sign_in_result=auth_response),
        profiles_rows=[{"id": VALID_USER_ID, "role": role}],
    )
    _patch_client_factory(monkeypatch, client)

    identity = sign_in("user@example.com", "correct-password")
    assert identity.role == role


def test_profile_lookup_rls_denial_classified(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(), _FakeUser())

    class _RLSDenyingTable(_FakeTable):
        def select(self, *_args):
            raise Exception("new row violates row-level security policy for table \"profiles\"")

    class _RLSDenyingClient(_FakeClient):
        def table(self, name):
            return _RLSDenyingTable(name, self._profiles_rows)

    client = _RLSDenyingClient(_FakeAuth(sign_in_result=auth_response), profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError) as excinfo:
        sign_in("admin@example.com", "correct-password")
    assert "rls_denied" in str(excinfo.value)


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


def test_refresh_success(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(access_token="new-token"), _FakeUser())
    client = _FakeClient(
        _FakeAuth(refresh_result=auth_response), profiles_rows=[{"id": VALID_USER_ID, "role": "Analyst"}]
    )
    _patch_client_factory(monkeypatch, client)

    identity = refresh(VALID_REFRESH_TOKEN)
    assert identity.access_token == "new-token"
    assert identity.role == "Analyst"


def test_refresh_failure_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    client = _FakeClient(_FakeAuth(refresh_error=_AuthApiError("invalid_grant")), profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError):
        refresh("expired-or-invalid-refresh-token")


# ---------------------------------------------------------------------------
# resolve_identity_from_tokens
# ---------------------------------------------------------------------------


def test_resolve_identity_from_tokens_success(monkeypatch):
    _patch_env(monkeypatch)
    auth_response = _FakeAuthResponse(_FakeSession(), _FakeUser())
    client = _FakeClient(
        _FakeAuth(set_session_result=auth_response), profiles_rows=[{"id": VALID_USER_ID, "role": "Business"}]
    )
    _patch_client_factory(monkeypatch, client)

    identity = resolve_identity_from_tokens(VALID_ACCESS_TOKEN, VALID_REFRESH_TOKEN)
    assert identity.role == "Business"


def test_resolve_identity_from_tokens_invalid_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    client = _FakeClient(_FakeAuth(set_session_error=_AuthApiError("invalid token")), profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError):
        resolve_identity_from_tokens("bad-token", "bad-refresh")


# ---------------------------------------------------------------------------
# sign_out
# ---------------------------------------------------------------------------


def test_sign_out_success(monkeypatch):
    _patch_env(monkeypatch)
    fake_auth = _FakeAuth(set_session_result=_FakeAuthResponse(_FakeSession(), _FakeUser()))
    client = _FakeClient(fake_auth, profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    sign_out(VALID_ACCESS_TOKEN, VALID_REFRESH_TOKEN)
    assert fake_auth.sign_out_called is True


def test_sign_out_failure_fails_closed(monkeypatch):
    _patch_env(monkeypatch)
    fake_auth = _FakeAuth(
        set_session_result=_FakeAuthResponse(_FakeSession(), _FakeUser()),
        sign_out_error=Exception("network error"),
    )
    client = _FakeClient(fake_auth, profiles_rows=[])
    _patch_client_factory(monkeypatch, client)

    with pytest.raises(AuthenticationError):
        sign_out(VALID_ACCESS_TOKEN, VALID_REFRESH_TOKEN)


# ---------------------------------------------------------------------------
# classify_auth_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_AuthApiError("Invalid login credentials"), "auth_failed"),
        (Exception("invalid_grant"), "auth_failed"),
        (Exception('new row violates row-level security policy for table "profiles"'), "rls_denied"),
        (Exception("permission denied for table profiles"), "privilege_denied"),
        (Exception("Connection timed out"), "connectivity_error"),
        (Exception("getaddrinfo failed"), "connectivity_error"),
        (Exception("some completely unrelated failure"), "unknown"),
    ],
)
def test_classify_auth_error(exc, expected):
    assert classify_auth_error(exc) == expected


# ---------------------------------------------------------------------------
# no legacy fallback / no service-role / no secret leakage
# ---------------------------------------------------------------------------


def test_sign_in_failure_does_not_reference_config_yaml_or_legacy_auth(monkeypatch):
    """The module must never import or call into the legacy config.yaml
    authentication path -- confirming there is no code-level coupling that
    could become an automatic fallback."""
    import src.persistence.supabase_auth as mod

    source = mod.__file__
    with open(source, encoding="utf-8") as f:
        content = f.read()
    assert "config.yaml" not in content
    assert "src.utils.auth" not in content
    assert "verify_credentials" not in content


def test_module_never_reads_service_role_key():
    """Mirrors test_persistence_env_and_safety.py's existing check: the
    actual env-var read call (as opposed to documentation mentioning the
    name to explain it is deliberately unused) must never occur here."""
    import src.persistence.supabase_auth as mod

    with open(mod.__file__, encoding="utf-8") as f:
        content = f.read()
    assert '_require_env("SUPABASE_SERVICE_ROLE_KEY")' not in content
    assert "os.environ" not in content  # no direct env read at all -- only via _require_env(URL/ANON_KEY)
    assert "build_service_role_client()" not in content


def test_authenticated_identity_repr_never_exposes_tokens():
    identity = AuthenticatedIdentity(
        user_id=VALID_USER_ID,
        email="admin@example.com",
        role="Admin",
        access_token="super-secret-access-token",
        refresh_token="super-secret-refresh-token",
        expires_at=9999999999,
    )
    rendered = repr(identity)
    assert "super-secret-access-token" not in rendered
    assert "super-secret-refresh-token" not in rendered
    assert "<redacted>" in rendered

    rendered_str = str(identity)
    assert "super-secret-access-token" not in rendered_str
    assert "super-secret-refresh-token" not in rendered_str
