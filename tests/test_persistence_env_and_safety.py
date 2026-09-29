"""Phase 4A Step 1 tests: environment-variable contract and secret-safety
guardrails. No live Supabase connection required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.persistence.interfaces import PersistenceError
from src.persistence.supabase_client import build_service_role_client, build_user_scoped_client

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_supabase_env(monkeypatch):
    for name in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_build_user_scoped_client_requires_url():
    with pytest.raises(PersistenceError):
        build_user_scoped_client(access_token="fake-jwt")


def test_build_user_scoped_client_requires_anon_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    with pytest.raises(PersistenceError):
        build_user_scoped_client(access_token="fake-jwt")


def test_build_service_role_client_requires_service_role_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    with pytest.raises(PersistenceError):
        build_service_role_client()


def test_env_example_declares_names_only_no_values():
    content = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line, f"malformed .env.example line: {line}"
        name, _, value = line.partition("=")
        assert value == "" or name == "APP_ENV" or name == "ARTIFACT_ROOT", (
            f".env.example must declare {name} as a bare name with no value, found {value!r}"
        )


def test_env_example_covers_required_supabase_variables():
    content = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for required in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
        assert required in content


def test_env_is_gitignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore


def test_no_persistence_source_file_contains_hardcoded_supabase_key_literal():
    # Service-role/anon keys are JWTs and always start with this prefix.
    # None of our own source files may contain a literal one.
    persistence_dir = REPO_ROOT / "src" / "persistence"
    for path in persistence_dir.glob("*.py"):
        content = path.read_text(encoding="utf-8")
        assert "eyJhbGciOi" not in content, f"possible hardcoded JWT/key literal in {path}"


def test_supabase_client_module_reads_service_role_key_in_exactly_one_call_site():
    content = (REPO_ROOT / "src" / "persistence" / "supabase_client.py").read_text(encoding="utf-8")
    # The actual env-var read call (as opposed to documentation mentioning
    # the name) must occur exactly once.
    assert content.count('_require_env("SUPABASE_SERVICE_ROLE_KEY")') == 1


def test_no_other_source_file_reads_service_role_key_directly():
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if path.name == "supabase_client.py":
            continue
        content = path.read_text(encoding="utf-8")
        assert "SUPABASE_SERVICE_ROLE_KEY" not in content, (
            f"{path} must not read SUPABASE_SERVICE_ROLE_KEY directly -- "
            "go through src/persistence/supabase_client.py"
        )


def test_no_crewai_agent_module_imports_persistence_or_supabase():
    agents_dir = REPO_ROOT / "src" / "agents"
    for path in agents_dir.glob("*.py"):
        content = path.read_text(encoding="utf-8")
        assert "supabase" not in content.lower(), f"{path} must not reference Supabase directly (agent security boundary)"
        assert "src.persistence" not in content, f"{path} must not import the persistence layer directly"
