"""Phase 4B.4 tests for safe background-workflow identity propagation.

Scope: §12 invariant #3 (the background thread must never read
st.session_state for credentials) and §16 item 4 of the Phase 4B Obsidian
note. Covers capture-before-thread, fail-closed on a missing/empty token
in Supabase mode, legacy-mode non-requirement, user-scoped client
construction (success/failure), the identity_preparation outcome field,
and token-leakage prevention (direct and indirect, via the root logger).

No live Supabase/STAGING contact, no real RetailFlow execution, no real
background thread, no real Streamlit server -- all fakes, same discipline
as tests/test_run_ownership.py. Kept in a dedicated file, separate from
tests/test_run_ownership.py (run ownership) and tests/test_auth_session.py
(authentication), per the approved Phase 4B.4 scope.

Two distinct test strategies are used:

1. start_new_run()/render_admin_page() tests: extracted verbatim from the
   real app.py via AST node boundaries and run through
   streamlit.testing.v1.AppTest, with the same scoped fake-threading.Thread
   technique proven in test_run_ownership.py (patched and restored INSIDE
   the exec'd script, never from outside via pytest's monkeypatch, which
   would break AppTest's own internal thread use).

2. _execute_flow() tests: this function never touches st.session_state
   (that is the entire point of the invariant it implements), so it is
   extracted the same way but exec'd directly as plain Python and called
   synchronously -- no AppTest needed. src.flows.retail_flow is replaced
   with a fake module (via monkeypatch.setitem(sys.modules, ...)) so the
   function's own internal `from src.flows.retail_flow import RetailFlow`
   resolves to a fake, inert class; build_user_scoped_client is replaced
   via monkeypatch.setattr on the real (already-safe-to-import)
   src.persistence.supabase_client module.
"""

from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import src.persistence.supabase_client as supabase_client_module
from src.persistence.interfaces import PersistenceError

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

_DUMMY_TOKEN = "dummy-test-token"
_CANARY_TOKEN = "canary-secret-token-value"


def _extract_node_source(app_path: Path, name: str) -> str:
    """Extract one top-level function/class's exact source from the real
    app.py via its AST node boundaries -- tests the real code verbatim.
    Fails loudly if the name is renamed/removed, rather than silently
    testing stale logic."""
    lines = app_path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno]) + "\n"
    raise AssertionError(f"{name!r} not found in app.py -- extraction markers no longer match")


_LIST_LOG_HANDLER_SOURCE = _extract_node_source(APP_PATH, "_ListLogHandler")
_MISSING_IDENTITY_ERROR_SOURCE = _extract_node_source(APP_PATH, "MissingIdentityError")
_EXECUTE_FLOW_SOURCE = _extract_node_source(APP_PATH, "_execute_flow")
_START_NEW_RUN_SOURCE = _extract_node_source(APP_PATH, "start_new_run")
_RENDER_ADMIN_PAGE_SOURCE = _extract_node_source(APP_PATH, "render_admin_page")


# ---------------------------------------------------------------------------
# Strategy 1: start_new_run() via AppTest, fake-threaded, real MissingIdentityError
# ---------------------------------------------------------------------------


def _make_start_new_run_script(auth_mode: str) -> str:
    return (
        "from __future__ import annotations\n"
        "import threading\n"
        "import uuid\n"
        "from datetime import datetime, timezone\n"
        "from typing import Any\n"
        "import streamlit as st\n"
        f"AUTH_MODE = {auth_mode!r}\n"
        "\n"
        + _MISSING_IDENTITY_ERROR_SOURCE
        + "\n"
        "def _execute_flow(run_id, entry, access_token):\n"
        "    raise AssertionError('_execute_flow must never actually run in this test')\n"
        "\n"
        "_fake_thread_instances = []\n"
        "\n"
        "class _FakeThread:\n"
        "    \"\"\"Records constructor args but never starts a real thread. Scoped\n"
        "    to the single start_new_run() call below via try/finally -- the real\n"
        "    threading.Thread class is restored before this script finishes, so no\n"
        "    pollution survives into AppTest's own internals or any other test.\"\"\"\n"
        "    def __init__(self, target=None, args=(), daemon=None):\n"
        "        self.target = target\n"
        "        self.args = args\n"
        "        _fake_thread_instances.append(self)\n"
        "    def start(self):\n"
        "        pass\n"
        "\n"
        "if 'run_registry' not in st.session_state:\n"
        "    st.session_state.run_registry = {}\n"
        "if 'active_run_id' not in st.session_state:\n"
        "    st.session_state.active_run_id = None\n"
        "\n"
        + _START_NEW_RUN_SOURCE
        + "\n"
        "_captured_exception = None\n"
        "_real_thread_class = threading.Thread\n"
        "threading.Thread = _FakeThread\n"
        "try:\n"
        "    try:\n"
        "        st.session_state['_test_run_id'] = start_new_run()\n"
        "    except MissingIdentityError as exc:\n"
        "        _captured_exception = exc\n"
        "finally:\n"
        "    threading.Thread = _real_thread_class\n"
        "st.session_state['_test_exception_raised'] = _captured_exception is not None\n"
        "st.session_state['_test_thread_args'] = [t.args for t in _fake_thread_instances]\n"
    )


def test_start_new_run_supabase_mode_missing_token_raises_and_creates_nothing():
    at = AppTest.from_string(_make_start_new_run_script("supabase"))
    at.session_state["username"] = "alice"
    # no access_token key set at all
    at.run()
    assert at.session_state["_test_exception_raised"] is True
    assert at.session_state["run_registry"] == {}
    assert at.session_state["_test_thread_args"] == []


def test_start_new_run_supabase_mode_empty_string_token_also_fails_closed():
    at = AppTest.from_string(_make_start_new_run_script("supabase"))
    at.session_state["username"] = "alice"
    at.session_state["access_token"] = ""
    at.run()
    assert at.session_state["_test_exception_raised"] is True
    assert at.session_state["run_registry"] == {}
    assert at.session_state["_test_thread_args"] == []


def test_start_new_run_supabase_mode_valid_token_passed_to_thread_not_stored_in_entry():
    at = AppTest.from_string(_make_start_new_run_script("supabase"))
    at.session_state["username"] = "alice"
    at.session_state["access_token"] = _DUMMY_TOKEN
    at.run()
    assert at.session_state["_test_exception_raised"] is False
    thread_args = at.session_state["_test_thread_args"]
    assert len(thread_args) == 1
    run_id, entry, access_token = thread_args[0]
    assert access_token == _DUMMY_TOKEN
    assert "access_token" not in entry
    assert "refresh_token" not in entry
    assert entry["identity_preparation"] is None  # thread faked -- _execute_flow never actually ran
    stored_entry = at.session_state["run_registry"][run_id]
    assert "access_token" not in stored_entry
    assert "refresh_token" not in stored_entry


def test_start_new_run_legacy_mode_unaffected_by_missing_token_check():
    at = AppTest.from_string(_make_start_new_run_script("legacy"))
    at.session_state["username"] = "bob"
    # no access_token set at all -- must not matter in legacy mode
    at.run()
    assert at.session_state["_test_exception_raised"] is False
    thread_args = at.session_state["_test_thread_args"]
    assert len(thread_args) == 1
    _, _, access_token = thread_args[0]
    assert access_token is None


def test_post_launch_session_mutation_does_not_affect_already_passed_token():
    at = AppTest.from_string(_make_start_new_run_script("supabase"))
    at.session_state["username"] = "alice"
    at.session_state["access_token"] = _DUMMY_TOKEN
    at.run()
    _, _, captured_token = at.session_state["_test_thread_args"][0]
    # Simulate logout clearing the session AFTER the thread already launched.
    at.session_state["access_token"] = None
    assert captured_token == _DUMMY_TOKEN  # already-captured value, unaffected by later mutation


# ---------------------------------------------------------------------------
# Strategy 1b: render_admin_page() button-handler integration (fail-closed path)
# ---------------------------------------------------------------------------

_ADMIN_BUTTON_SCRIPT = (
    "from __future__ import annotations\n"
    "import threading\n"
    "import uuid\n"
    "from datetime import datetime, timezone\n"
    "from typing import Any\n"
    "import streamlit as st\n"
    "import pandas as pd\n"
    "AUTH_MODE = 'supabase'\n"
    "\n"
    + _MISSING_IDENTITY_ERROR_SOURCE
    + "\n"
    "def _execute_flow(run_id, entry, access_token):\n"
    "    raise AssertionError('_execute_flow must never actually run in this test')\n"
    "\n"
    "def _render_live_log_panel(run_id):\n"
    "    pass\n"
    "\n"
    "class _FakeThread:\n"
    "    def __init__(self, target=None, args=(), daemon=None):\n"
    "        pass\n"
    "    def start(self):\n"
    "        pass\n"
    "\n"
    "runs = []\n"
    "if 'run_registry' not in st.session_state:\n"
    "    st.session_state.run_registry = {}\n"
    "if 'active_run_id' not in st.session_state:\n"
    "    st.session_state.active_run_id = None\n"
    "\n"
    + _START_NEW_RUN_SOURCE
    + "\n"
    + _RENDER_ADMIN_PAGE_SOURCE
    + "\n"
    "_real_thread_class = threading.Thread\n"
    "threading.Thread = _FakeThread\n"
    "try:\n"
    "    render_admin_page()\n"
    "finally:\n"
    "    threading.Thread = _real_thread_class\n"
)


def test_button_handler_shows_error_and_creates_nothing_on_missing_identity():
    at = AppTest.from_string(_ADMIN_BUTTON_SCRIPT)
    at.session_state["username"] = "alice"
    # no access_token set
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)
    assert any("missing required identity information" in e.value for e in at.error)
    assert at.session_state["run_registry"] == {}


# ---------------------------------------------------------------------------
# Strategy 2: _execute_flow() direct, synchronous, no AppTest needed
# ---------------------------------------------------------------------------


class _FakeFlowState:
    def __init__(self, status: str = "COMPLETED") -> None:
        self.status = status

    def model_dump(self) -> dict:
        return {"status": self.status}


class _FakeRetailFlow:
    def __init__(self, raw_dir: Any, artifacts_root: Any, run_id: str) -> None:
        self.state = _FakeFlowState()

    def kickoff(self) -> None:
        pass


def _make_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "logs": [],
        "status": "RUNNING",
        "running": True,
        "error": None,
        "state": None,
        "identity_preparation": None,
        "owner_user_id": "alice",
    }
    entry.update(overrides)
    return entry


@pytest.fixture(autouse=True)
def _preserve_root_logger_state():
    """_execute_flow() mutates the real root logger's level/handlers (as it
    already did before this change) -- restore both after each test in this
    file so calling the real function here cannot pollute any other test's
    logging behavior."""
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = list(root_logger.handlers)
    yield
    root_logger.setLevel(original_level)
    root_logger.handlers = original_handlers


@pytest.fixture
def execute_flow_namespace(monkeypatch):
    fake_retail_flow_module = types.ModuleType("src.flows.retail_flow")
    fake_retail_flow_module.RetailFlow = _FakeRetailFlow
    monkeypatch.setitem(sys.modules, "src.flows.retail_flow", fake_retail_flow_module)

    namespace: dict[str, Any] = {
        "logging": logging,
        "RAW_DIR": Path("fake-raw"),
        "ARTIFACTS_DIR": Path("fake-artifacts"),
    }
    script = "from __future__ import annotations\n" + _LIST_LOG_HANDLER_SOURCE + "\n" + _EXECUTE_FLOW_SOURCE
    exec(compile(script, "<extracted _execute_flow>", "exec"), namespace)  # noqa: S102 - trusted, AST-extracted real source
    return namespace


def test_execute_flow_sets_identity_preparation_success_on_successful_construction(execute_flow_namespace, monkeypatch):
    monkeypatch.setattr(supabase_client_module, "build_user_scoped_client", lambda token: object())
    entry = _make_entry()
    execute_flow_namespace["_execute_flow"]("run-1", entry, _DUMMY_TOKEN)
    assert entry["identity_preparation"] == "success"
    assert entry["running"] is False
    assert entry["status"] == "COMPLETED"
    assert any("constructed" in line for line in entry["logs"])


def test_execute_flow_sets_identity_preparation_failed_on_persistence_error(execute_flow_namespace, monkeypatch):
    def _raise(token: str) -> None:
        raise PersistenceError("required environment variable SUPABASE_URL is not set -- see .env.example")

    monkeypatch.setattr(supabase_client_module, "build_user_scoped_client", _raise)
    entry = _make_entry()
    execute_flow_namespace["_execute_flow"]("run-1", entry, _DUMMY_TOKEN)
    assert entry["identity_preparation"] == "failed"
    assert entry["running"] is False
    assert entry["status"] == "COMPLETED"  # the (faked) workflow still completed independently
    assert any("FAILED" in line for line in entry["logs"])


def test_execute_flow_sets_identity_preparation_not_applicable_in_legacy_mode(execute_flow_namespace):
    entry = _make_entry()
    execute_flow_namespace["_execute_flow"]("run-1", entry, None)
    assert entry["identity_preparation"] == "not_applicable"
    assert entry["running"] is False
    assert entry["status"] == "COMPLETED"


def test_identity_preparation_field_is_always_a_plain_string_never_a_client_or_token(execute_flow_namespace, monkeypatch):
    monkeypatch.setattr(supabase_client_module, "build_user_scoped_client", lambda token: object())
    entry = _make_entry()
    execute_flow_namespace["_execute_flow"]("run-1", entry, _DUMMY_TOKEN)
    assert isinstance(entry["identity_preparation"], str)
    assert entry["identity_preparation"] not in (_DUMMY_TOKEN, None)


def test_execute_flow_unexpected_construction_failure_does_not_leave_running_stuck_true(execute_flow_namespace, monkeypatch):
    """A non-PersistenceError raised during construction is now caught by
    the widened identity-preparation block (B1 fix) -- RetailFlow still
    runs and completes, and `entry["running"]` still reaches False via the
    existing finally clause."""

    def _raise(token: str) -> None:
        raise RuntimeError("unexpected SDK internal error")

    monkeypatch.setattr(supabase_client_module, "build_user_scoped_client", _raise)
    entry = _make_entry()
    execute_flow_namespace["_execute_flow"]("run-1", entry, _DUMMY_TOKEN)
    assert entry["running"] is False
    assert entry["status"] == "COMPLETED"
    assert entry["identity_preparation"] == "failed"


# ---------------------------------------------------------------------------
# Token-leakage prevention: direct and indirect (via the root logger)
# ---------------------------------------------------------------------------


def test_execute_flow_client_construction_failure_message_does_not_leak_token(execute_flow_namespace, monkeypatch):
    def _raise(received_token: str) -> None:
        raise PersistenceError(f"sdk failure, token was {received_token}")

    monkeypatch.setattr(supabase_client_module, "build_user_scoped_client", _raise)
    entry = _make_entry()
    execute_flow_namespace["_execute_flow"]("run-1", entry, _CANARY_TOKEN)
    rendered = "\n".join(entry["logs"])
    assert _CANARY_TOKEN not in rendered
    assert entry.get("error") is None or _CANARY_TOKEN not in str(entry["error"])


def test_execute_flow_non_persistence_error_during_construction_does_not_leak_token(execute_flow_namespace, monkeypatch):
    """Regression for the security-review BLOCKER finding: a non-
    PersistenceError construction failure must still be caught, set
    identity_preparation to 'failed', never log the raw exception message
    or the token, and must not prevent the (faked) workflow from
    completing normally."""

    def _raise(received_token: str) -> None:
        raise ValueError(f"malformed SUPABASE_URL, token was {received_token}")

    monkeypatch.setattr(supabase_client_module, "build_user_scoped_client", _raise)
    entry = _make_entry()
    execute_flow_namespace["_execute_flow"]("run-1", entry, _CANARY_TOKEN)
    assert entry["identity_preparation"] == "failed"
    assert _CANARY_TOKEN not in "\n".join(entry["logs"])
    assert entry["error"] is None
    assert entry["status"] == "COMPLETED"  # RetailFlow still ran and completed
    assert entry["running"] is False


def test_execute_flow_redacts_token_appearing_indirectly_via_root_logger(execute_flow_namespace, monkeypatch):
    """Simulates a hypothetical verbose SDK log record (not under this
    code's control) emitted during construction. Because _execute_flow's
    handler is attached to the root logger, any propagating record is
    covered by the same redaction filter as our own log lines."""

    def _emit_verbose_log_then_succeed(received_token: str):
        logging.getLogger("fake.sdk").info("debug: using token %s", received_token)
        return object()

    monkeypatch.setattr(supabase_client_module, "build_user_scoped_client", _emit_verbose_log_then_succeed)
    entry = _make_entry()
    execute_flow_namespace["_execute_flow"]("run-1", entry, _CANARY_TOKEN)
    rendered = "\n".join(entry["logs"])
    assert _CANARY_TOKEN not in rendered
    assert "<redacted>" in rendered


def test_execute_flow_never_references_st_session_state():
    assert "st.session_state" not in _EXECUTE_FLOW_SOURCE
    assert "session_state" not in _EXECUTE_FLOW_SOURCE


def test_execute_flow_never_imports_or_calls_build_service_role_client():
    assert "build_service_role_client" not in _EXECUTE_FLOW_SOURCE
    assert "build_user_scoped_client" in _EXECUTE_FLOW_SOURCE
