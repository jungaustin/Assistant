"""Keep the test suite away from the real state/ databases.

Several tests construct a full Agent, and Agent.stream() checkpoints the
conversation and writes episodic memory. Without this, `just test` wrote into
the live daily thread — and on 2026-08-25 a looping test appended ~1,470
turns to the user's real conversation before it was killed. Tests must never
touch state/conversations.db, state/memory.db or state/tracker.db.

These are set at conftest import, which pytest does before importing any test
module, so robot.config reads them when it is first imported. load_dotenv()
does not override variables already present in the environment, so a
developer's own .env cannot pull the suite back onto real data.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="robot-tests-"))

for _var, _name in (
    ("STATE_DB_PATH", "conversations.db"),
    ("MEMORY_DB_PATH", "memory.db"),
    ("TRACKER_DB_PATH", "tracker.db"),
    ("DISCORD_CURSOR_PATH", "discord_cursors.json"),
    ("CAMERA_LOG_PATH", "camera_access.log"),
):
    os.environ[_var] = str(_TMP / _name)

# Belt and braces: a stray real-DB write would land here, not in state/.
os.environ.setdefault("BRAIN_PREWARM", "0")


import pytest  # noqa: E402


@pytest.fixture
def tmp_state_dir() -> Path:
    """The throwaway directory the suite's databases live in."""
    return _TMP


def pytest_report_header(config) -> str:
    return f"robot state dbs redirected to {_TMP}"
