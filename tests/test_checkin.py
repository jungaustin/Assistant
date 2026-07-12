"""Tests for the nightly check-in (core/checkin.py).

Uses a real TrackerDB on tmp_path (it's just SQLite) and an async mock
for speak — no TTS, no 22:00 wait: the loop test patches
seconds_until_hour to fire immediately.

Covers:
  - missing_daily_types against logged/unlogged days
  - prompt grammar for 1 / 2 / 3+ missing types, None when complete
  - seconds_until_hour rolls to tomorrow when the hour has passed
  - checkin_loop speaks the prompt for missing types
  - checkin_loop stays silent when everything is logged
  - disabled configs (hour=-1, empty required list) exit immediately
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime

import pytest

import robot.core.checkin as checkin
from robot.core.checkin import (
    build_checkin_prompt,
    checkin_loop,
    missing_daily_types,
    seconds_until_hour,
)
from robot.tools.inner.log import TrackerDB


@pytest.fixture
def db(tmp_path):
    d = TrackerDB(str(tmp_path / "tracker.db"))
    yield d
    d.close()


REQUIRED = ["calories", "exercise", "sleep"]


# ---------------------------------------------------------------------------
# missing_daily_types
# ---------------------------------------------------------------------------


def test_missing_all_on_empty_day(db):
    assert missing_daily_types(db, REQUIRED) == REQUIRED


def test_logged_types_drop_out_in_order(db):
    db.log_entry("calories", 600, "lunch")
    db.log_entry("mood", 8)  # not required — shouldn't matter
    assert missing_daily_types(db, REQUIRED) == ["exercise", "sleep"]


def test_yesterdays_entries_dont_count(db):
    db.log_entry("sleep", 8, entry_date="2026-06-11")
    assert "sleep" in missing_daily_types(db, REQUIRED, day=date.today().isoformat())


def test_all_logged_means_nothing_missing(db):
    for t in REQUIRED:
        db.log_entry(t, 1)
    assert missing_daily_types(db, REQUIRED) == []


# ---------------------------------------------------------------------------
# build_checkin_prompt
# ---------------------------------------------------------------------------


def test_prompt_grammar():
    assert build_checkin_prompt([]) is None
    assert "logged sleep today" in build_checkin_prompt(["sleep"])
    assert "calories or sleep" in build_checkin_prompt(["calories", "sleep"])
    assert "calories, exercise, or sleep" in build_checkin_prompt(REQUIRED)


# ---------------------------------------------------------------------------
# seconds_until_hour
# ---------------------------------------------------------------------------


def test_seconds_until_hour_future_today():
    now = datetime(2026, 6, 12, 21, 0, 0)
    assert seconds_until_hour(22, now=now) == 3600


def test_seconds_until_hour_rolls_to_tomorrow():
    now = datetime(2026, 6, 12, 22, 30, 0)
    assert seconds_until_hour(22, now=now) == 23.5 * 3600


# ---------------------------------------------------------------------------
# checkin_loop
# ---------------------------------------------------------------------------


async def _run_one_firing(db, monkeypatch, required):
    """Run the loop long enough for exactly one firing; return spoken texts."""
    monkeypatch.setattr(checkin, "seconds_until_hour", lambda hour: 0.01)
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    task = asyncio.create_task(checkin_loop(db, speak, hour=22, required=required))
    await asyncio.sleep(0.2)  # one firing, then it's in the 61s settle sleep
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return spoken


async def test_loop_speaks_when_types_missing(db, monkeypatch):
    db.log_entry("calories", 600)
    spoken = await _run_one_firing(db, monkeypatch, REQUIRED)
    assert len(spoken) == 1
    assert "exercise or sleep" in spoken[0]


async def test_loop_silent_when_all_logged(db, monkeypatch):
    for t in REQUIRED:
        db.log_entry(t, 1)
    spoken = await _run_one_firing(db, monkeypatch, REQUIRED)
    assert spoken == []


async def test_loop_disabled_by_hour(db):
    # Returns immediately instead of looping — await would hang otherwise.
    await asyncio.wait_for(
        checkin_loop(db, speak=None, hour=-1, required=REQUIRED), timeout=1
    )


async def test_loop_disabled_by_empty_required(db):
    await asyncio.wait_for(
        checkin_loop(db, speak=None, hour=22, required=[]), timeout=1
    )
