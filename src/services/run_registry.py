"""Pure, Streamlit-free helpers for discovering genuine CrewAI Flow run
directories under an artifacts root.

Split out from app.py so run-discovery logic (in particular, telling a real
workflow run apart from a hand-built fixture) is unit-testable without
importing Streamlit.
"""

from __future__ import annotations

import re
from pathlib import Path

# Matches exactly what RetailFlow.initiate_flow() (src/flows/retail_flow.py)
# generates: strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]. Any directory
# under the artifacts root that doesn't match this pattern was not produced by
# a real CrewAI Flow run (e.g. a hand-built UI test fixture) and must never be
# surfaced as a genuine workflow run, regardless of what its run_metadata.json
# claims.
REAL_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")


def is_real_run_id(run_id: str) -> bool:
    return bool(REAL_RUN_ID_PATTERN.match(run_id))


def list_real_run_dirs(artifacts_dir: str | Path) -> list[Path]:
    """Return real-run subdirectories of artifacts_dir, newest first."""
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        return []
    return sorted(
        (d for d in artifacts_dir.iterdir() if d.is_dir() and is_real_run_id(d.name)),
        reverse=True,
    )
