"""Phase 4B.2D tests for workflow-run ownership isolation in app.py.

Fakes only: start_new_run() and render_admin_page() are extracted verbatim
from the real app.py via their AST node boundaries (never a hand-duplicated
copy) and run through streamlit.testing.v1.AppTest with stubbed
dependencies -- no real background thread, no RetailFlow execution, no
Streamlit server, no Supabase/STAGING contact. Kept in a dedicated file,
separate from tests/test_auth_session.py, per the approved Phase 4B.2D
scope (run ownership, not authentication).

threading.Thread isolation: AppTest.run() itself internally spawns a real
threading.Thread to execute the script, so patching threading.Thread from
OUTSIDE the script (e.g. via pytest's monkeypatch, before calling .run())
breaks AppTest's own machinery (verified empirically -- it raises
TypeError inside AppTest's script runner). The fake-thread substitution is
therefore performed entirely INSIDE the exec'd script, scoped with
try/finally around the single start_new_run() call, so the real
threading.Thread class is restored before the script (and .run()) even
finishes -- verified empirically to leave zero pollution for any
subsequent AppTest run or any other test in the session.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _extract_function_source(app_path: Path, function_name: str) -> str:
    """Extract one top-level function's exact source from the real app.py
    via its AST node boundaries (node.lineno/end_lineno) -- tests the real
    code verbatim. Fails loudly if the function is renamed/removed, rather
    than silently testing stale logic."""
    lines = app_path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno]) + "\n"
    raise AssertionError(f"{function_name!r} not found in app.py -- extraction markers no longer match")


_START_NEW_RUN_SOURCE = _extract_function_source(APP_PATH, "start_new_run")
_RENDER_ADMIN_PAGE_SOURCE = _extract_function_source(APP_PATH, "render_admin_page")

_START_NEW_RUN_SCRIPT = (
    "import threading\n"
    "import uuid\n"
    "from datetime import datetime, timezone\n"
    "from typing import Any\n"
    "import streamlit as st\n"
    "\n"
    "def _execute_flow(run_id, entry):\n"
    "    raise AssertionError('_execute_flow must never actually run in this test')\n"
    "\n"
    "class _FakeThread:\n"
    "    \"\"\"Records construction args but never starts a real thread.\n"
    "    Scoped to the single start_new_run() call below via try/finally --\n"
    "    threading.Thread is restored to the real class before this script\n"
    "    finishes, so no pollution survives into AppTest's own internals or\n"
    "    any other test.\"\"\"\n"
    "    def __init__(self, target=None, args=(), daemon=None):\n"
    "        self.target = target\n"
    "        self.args = args\n"
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
    "_real_thread_class = threading.Thread\n"
    "threading.Thread = _FakeThread\n"
    "try:\n"
    "    run_id = start_new_run()\n"
    "finally:\n"
    "    threading.Thread = _real_thread_class\n"
    "st.session_state['_test_run_id'] = run_id\n"
)

_ADMIN_PAGE_PREAMBLE = (
    "import streamlit as st\n"
    "import pandas as pd\n"
    "\n"
    "_log_panel_calls = []\n"
    "def _render_live_log_panel(run_id):\n"
    "    _log_panel_calls.append(run_id)\n"
    "\n"
    "def start_new_run():\n"
    "    raise AssertionError('start_new_run must not be called -- button not clicked in these tests')\n"
    "\n"
    "runs = []\n"
    "if 'run_registry' not in st.session_state:\n"
    "    st.session_state.run_registry = {}\n"
    "if 'active_run_id' not in st.session_state:\n"
    "    st.session_state.active_run_id = None\n"
)

_ADMIN_PAGE_SCRIPT = (
    _ADMIN_PAGE_PREAMBLE
    + _RENDER_ADMIN_PAGE_SOURCE
    + "\nrender_admin_page()\nst.session_state['_log_panel_calls'] = _log_panel_calls\n"
)


def _make_entry(owner_user_id="alice", running=True, **overrides):
    entry = {
        "logs": ["sensitive log line"],
        "status": "RUNNING",
        "running": running,
        "error": "sensitive error text",
        "state": None,
        "owner_user_id": owner_user_id,
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# Requirement 1: start_new_run() stores owner_user_id before starting the thread
# ---------------------------------------------------------------------------


def test_start_new_run_stores_owner_user_id_without_spawning_a_real_thread():
    at = AppTest.from_string(_START_NEW_RUN_SCRIPT)
    at.session_state["username"] = "alice"
    at.run()
    run_id = at.session_state["_test_run_id"]
    entry = at.session_state["run_registry"][run_id]
    assert entry["owner_user_id"] == "alice"
    # all original fields preserved
    assert entry["status"] == "RUNNING"
    assert entry["running"] is True
    assert entry["error"] is None
    assert entry["state"] is None
    assert entry["logs"] == []


# ---------------------------------------------------------------------------
# Requirements 2, 6, 7: owned run display + any_running guard + missing-field handling
# ---------------------------------------------------------------------------


def test_render_admin_page_owned_run_displays_and_invokes_log_panel():
    at = AppTest.from_string(_ADMIN_PAGE_SCRIPT)
    at.session_state["username"] = "alice"
    at.session_state["run_registry"] = {"run-1": _make_entry(owner_user_id="alice")}
    at.session_state["active_run_id"] = "run-1"
    at.run()
    assert at.session_state["_log_panel_calls"] == ["run-1"]
    assert any("run-1" in s.value for s in at.subheader)


@pytest.mark.parametrize("owner_user_id", ["alice", None, "someone-else"])
def test_render_admin_page_any_running_guard_effective_regardless_of_ownership(owner_user_id):
    at = AppTest.from_string(_ADMIN_PAGE_SCRIPT)
    at.session_state["username"] = "bob"
    at.session_state["run_registry"] = {"run-1": _make_entry(owner_user_id=owner_user_id, running=True)}
    at.session_state["active_run_id"] = "run-1"
    at.run()
    assert at.button[0].disabled is True  # "Start New Run" stays disabled no matter who owns the running entry


def test_render_admin_page_any_running_guard_not_disabled_when_nothing_running():
    at = AppTest.from_string(_ADMIN_PAGE_SCRIPT)
    at.session_state["username"] = "bob"
    at.session_state["run_registry"] = {}
    at.session_state["active_run_id"] = None
    at.run()
    assert at.button[0].disabled is False


def test_render_admin_page_missing_owner_user_id_treated_as_non_owned():
    at = AppTest.from_string(_ADMIN_PAGE_SCRIPT)
    at.session_state["username"] = "alice"
    entry = _make_entry()
    del entry["owner_user_id"]  # pre-4B.2D-i entry shape, or any malformed/missing case
    at.session_state["run_registry"] = {"run-1": entry}
    at.session_state["active_run_id"] = "run-1"
    at.run()
    assert at.session_state["_log_panel_calls"] == []
    assert any(i.value == "A workflow run is currently executing." for i in at.info)


# ---------------------------------------------------------------------------
# Requirements 3, 4: non-owned run shows only the neutral message, nothing else
# ---------------------------------------------------------------------------


def test_render_admin_page_non_owned_run_shows_neutral_message_only():
    at = AppTest.from_string(_ADMIN_PAGE_SCRIPT)
    at.session_state["username"] = "bob"
    at.session_state["run_registry"] = {"run-1": _make_entry(owner_user_id="alice")}
    at.session_state["active_run_id"] = "run-1"
    at.run()
    assert at.session_state["_log_panel_calls"] == []
    assert any(i.value == "A workflow run is currently executing." for i in at.info)
    # The later "Run History" subheader legitimately renders regardless of
    # ownership (it is unconditional, before the `if not runs:` check) --
    # only the live-run identifier itself must never appear in any subheader.
    assert not any("run-1" in s.value for s in at.subheader)
    assert len(at.code) == 0  # no log text rendered at all


def test_render_admin_page_non_owned_run_never_leaks_sensitive_fields_in_rendered_output():
    at = AppTest.from_string(_ADMIN_PAGE_SCRIPT)
    at.session_state["username"] = "bob"
    at.session_state["run_registry"] = {
        "run-1": _make_entry(owner_user_id="alice", logs=["TOP SECRET LOG LINE"], error="TOP SECRET ERROR")
    }
    at.session_state["active_run_id"] = "run-1"
    at.run()
    # Sweep every text-bearing element category the non-owned branch (or any
    # earlier/later code in render_admin_page) could possibly render through,
    # including st.write/st.markdown (-> at.markdown), st.caption, and
    # st.text, not just the handful the current implementation happens to
    # use -- so this stays a true negative-space check rather than only
    # verifying today's specific call sites.
    rendered_texts = (
        [s.value for s in at.subheader]
        + [c.value for c in at.code]
        + [e.value for e in at.error]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
        + [s.value for s in at.success]
        + [m.value for m in at.markdown]  # covers both st.markdown(...) and st.write(str)
        + [c.value for c in at.caption]
        + [t.value for t in at.text]
    )
    # Forbidden substrings are specific secret markers unique to this test's
    # seeded entry -- they cannot coincidentally match unrelated static page
    # content (e.g. the "Start New Run" caption, "Admin — Workflow Control"
    # title, or "No runs yet..." info text), so a match here only ever
    # indicates a genuine leak, not a false positive from boilerplate UI text.
    for forbidden in ("run-1", "TOP SECRET LOG LINE", "TOP SECRET ERROR", "RUNNING"):
        assert not any(forbidden in text for text in rendered_texts), f"{forbidden!r} leaked into rendered output"


# ---------------------------------------------------------------------------
# Requirement 5: hiding a non-owned run never mutates the entry
# ---------------------------------------------------------------------------


def test_render_admin_page_non_owned_run_does_not_mutate_entry_or_registry():
    at = AppTest.from_string(_ADMIN_PAGE_SCRIPT)
    at.session_state["username"] = "bob"
    seed_entry = _make_entry(owner_user_id="alice")
    at.session_state["run_registry"] = {"run-1": dict(seed_entry)}
    at.session_state["active_run_id"] = "run-1"
    at.run()
    assert at.session_state["run_registry"]["run-1"] == seed_entry  # byte-for-byte unchanged
    assert at.session_state["_log_panel_calls"] == []  # the background-thread-adjacent panel was never touched
