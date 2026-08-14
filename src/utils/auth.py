"""Local, config.yaml-based authentication for the Streamlit app.

No external identity provider: usernames, display names, roles, and
SHA-256 password hashes all come from config.yaml. Session state is
Streamlit's own st.session_state -- nothing is persisted beyond the
browser session.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

SESSION_KEYS = ("authenticated", "username", "name", "role", "permissions")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_credentials(username: str, password: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the user's profile dict (name, role, permissions) if the
    username/password match config.yaml, else None.
    """
    users = config.get("credentials", {}).get("usernames", {})
    user_entry = users.get(username)
    if user_entry is None:
        return None
    if hash_password(password) != user_entry.get("password_hash"):
        return None

    role = user_entry.get("role")
    permissions = config.get("roles", {}).get(role, [])
    return {
        "username": username,
        "name": user_entry.get("name", username),
        "role": role,
        "permissions": permissions,
    }


def _init_session_state() -> None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.name = None
        st.session_state.role = None
        st.session_state.permissions = []


def is_authenticated() -> bool:
    _init_session_state()
    return bool(st.session_state.authenticated)


def current_user() -> dict[str, Any] | None:
    _init_session_state()
    if not st.session_state.authenticated:
        return None
    return {
        "username": st.session_state.username,
        "name": st.session_state.name,
        "role": st.session_state.role,
        "permissions": st.session_state.permissions,
    }


def has_permission(permission: str) -> bool:
    user = current_user()
    if user is None:
        return False
    return permission in user["permissions"]


def login(username: str, password: str, config: dict[str, Any]) -> bool:
    profile = verify_credentials(username, password, config)
    if profile is None:
        return False
    st.session_state.authenticated = True
    st.session_state.username = profile["username"]
    st.session_state.name = profile["name"]
    st.session_state.role = profile["role"]
    st.session_state.permissions = profile["permissions"]
    return True


def logout() -> None:
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)
    _init_session_state()


def render_login_form(config: dict[str, Any]) -> bool:
    """Render a login form in the current Streamlit context. Returns True
    once authenticated (either just now or from a prior rerun).
    """
    _init_session_state()
    if st.session_state.authenticated:
        return True

    st.title("Retail AI Intelligence & Forecasting Workflow Platform")
    st.subheader("Sign in")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if login(username, password, config):
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return st.session_state.authenticated


def render_logout_button() -> None:
    user = current_user()
    if user is None:
        return
    with st.sidebar:
        st.markdown(f"**{user['name']}**  \nRole: `{user['role']}`")
        if st.button("Log out"):
            logout()
            st.rerun()
