"""Tests for background timers (TimerManager + TimerTools).

Covers:
  - set_timer confirms with label and spoken duration
  - timers fire in the background and call on_done (the beep hook)
  - the conversation isn't blocked: set_timer returns immediately
  - list_timers shows remaining time, sorted soonest-first
  - list_timers with nothing running
  - cancel by label
  - cancel with no label and exactly one timer
  - cancel with no label and several timers asks which
  - cancel unknown label lists what's running
  - cancelled timers never fire
  - duration formatting (seconds / minutes / hours)

on_done is injected as a mock — no audio plays during tests. Real timers
with tiny durations (50ms) are used to test actual background firing.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from robot.tools.inner.timer_tools import TimerManager, TimerTools, _format_duration


def _mgr() -> tuple[TimerManager, MagicMock]:
    done = MagicMock()
    return TimerManager(on_done=done), done


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# set_timer / firing
# ---------------------------------------------------------------------------


def test_set_timer_confirms():
    mgr, _ = _mgr()
    result = mgr.set_timer(seconds=600, label="tea")
    assert "tea" in result
    assert "10 minutes" in result
    mgr.cancel_timer("tea")


def test_set_timer_rejects_nonpositive():
    mgr, _ = _mgr()
    assert "positive" in mgr.set_timer(seconds=0)
    assert "positive" in mgr.set_timer(seconds=-5)


def test_timer_fires_in_background(capsys):
    mgr, done = _mgr()
    mgr.set_timer(seconds=0.05, label="quick")
    assert _wait_for(lambda: done.called)
    assert "quick" in capsys.readouterr().out
    assert mgr.list_timers() == "No timers running."


def test_set_timer_returns_immediately():
    mgr, _ = _mgr()
    start = time.monotonic()
    mgr.set_timer(seconds=60, label="slow")
    assert time.monotonic() - start < 0.5
    mgr.cancel_timer("slow")


# ---------------------------------------------------------------------------
# list_timers
# ---------------------------------------------------------------------------


def test_list_timers_empty():
    mgr, _ = _mgr()
    assert mgr.list_timers() == "No timers running."


def test_list_timers_sorted_soonest_first():
    mgr, _ = _mgr()
    mgr.set_timer(seconds=600, label="laundry")
    mgr.set_timer(seconds=60, label="tea")
    lines = mgr.list_timers().splitlines()
    assert lines[0].startswith("tea")
    assert lines[1].startswith("laundry")
    assert "left" in lines[0]
    mgr.cancel_timer("tea")
    mgr.cancel_timer("laundry")


# ---------------------------------------------------------------------------
# cancel_timer
# ---------------------------------------------------------------------------


def test_cancel_by_label():
    mgr, _ = _mgr()
    mgr.set_timer(seconds=60, label="tea")
    assert "tea" in mgr.cancel_timer("tea")
    assert mgr.list_timers() == "No timers running."


def test_cancel_label_is_case_insensitive():
    mgr, _ = _mgr()
    mgr.set_timer(seconds=60, label="Tea")
    assert "Tea" in mgr.cancel_timer("tea")


def test_cancel_unlabeled_with_single_timer():
    mgr, _ = _mgr()
    mgr.set_timer(seconds=60, label="tea")
    assert "tea" in mgr.cancel_timer()
    assert mgr.list_timers() == "No timers running."


def test_cancel_unlabeled_with_multiple_timers_asks():
    mgr, _ = _mgr()
    mgr.set_timer(seconds=60, label="tea")
    mgr.set_timer(seconds=120, label="laundry")
    result = mgr.cancel_timer()
    assert "which one" in result
    assert "tea" in result and "laundry" in result
    # Nothing was cancelled.
    assert "tea" in mgr.list_timers()
    mgr.cancel_timer("tea")
    mgr.cancel_timer("laundry")


def test_cancel_unknown_label_lists_running():
    mgr, _ = _mgr()
    mgr.set_timer(seconds=60, label="tea")
    result = mgr.cancel_timer("pasta")
    assert "No timer named 'pasta'" in result
    assert "tea" in result
    mgr.cancel_timer("tea")


def test_cancel_with_no_timers():
    mgr, _ = _mgr()
    assert mgr.cancel_timer() == "No timers running."


def test_cancelled_timer_never_fires():
    mgr, done = _mgr()
    mgr.set_timer(seconds=0.1, label="doomed")
    mgr.cancel_timer("doomed")
    time.sleep(0.25)
    done.assert_not_called()


# ---------------------------------------------------------------------------
# duration formatting / tool surface
# ---------------------------------------------------------------------------


def test_format_duration():
    assert _format_duration(1) == "1 second"
    assert _format_duration(45) == "45 seconds"
    assert _format_duration(60) == "1 minute"
    assert _format_duration(90) == "1 minute 30 seconds"
    assert _format_duration(600) == "10 minutes"
    assert _format_duration(3600) == "1 hour"
    assert _format_duration(5400) == "1 hour 30 minutes"


def test_tool_names():
    tools = TimerTools(TimerManager(on_done=MagicMock()))
    names = {
        tools.create_set_timer_tool().name,
        tools.create_list_timers_tool().name,
        tools.create_cancel_timer_tool().name,
    }
    assert names == {"set_timer", "list_timers", "cancel_timer"}
