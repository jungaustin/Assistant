"""Deleting by description, so no id ever has to be guessed or looked up.

Austin was opening sqlite by hand to find ids before every deletion, because
delete_entry needs one. Asking the model to produce an id is also what caused
the 2026-09-01 data loss — it batched query_entries with delete_entry and
invented the id. Filters come straight from the user's words instead.
"""

from __future__ import annotations

import pytest

from robot.tools.inner.log import LogTools, TrackerDB


@pytest.fixture
def db(tmp_path):
    d = TrackerDB(str(tmp_path / "t.db"))
    d.log_entry(type="calories", value=650, note="steak", entry_date="2026-06-11")
    d.log_entry(type="calories", value=490, note="Panda Express Orange Chicken",
                entry_date="2026-08-31")
    d.log_entry(type="calories", value=600, note="Panda Express Chow Mein",
                entry_date="2026-08-31")
    d.log_entry(type="sleep", value=7.5, entry_date="2026-08-31")
    return d


def _ids(d) -> list[int]:
    return [r[0] for r in d._conn.execute("SELECT id FROM entries ORDER BY id")]


def test_delete_a_whole_day_leaves_other_days_alone(db):
    out = db.delete_entries(entry_date="2026-08-31")
    assert "Deleted 3 entries" in out
    assert _ids(db) == [1], "the June row must survive"


def test_delete_by_type_within_a_day(db):
    db.delete_entries(entry_date="2026-08-31", type="calories")
    remaining = db._conn.execute(
        "SELECT type FROM entries WHERE entry_date='2026-08-31'").fetchall()
    assert remaining == [("sleep",)]


def test_delete_by_note_text_uses_the_users_own_words(db):
    out = db.delete_entries(note_contains="panda")
    assert "Deleted 2 entries" in out
    assert _ids(db) == [1, 4]


def test_a_call_with_no_filters_is_refused(db):
    before = _ids(db)
    out = db.delete_entries()
    assert out.startswith("Refused")
    assert _ids(db) == before, "nothing may be deleted without a filter"


def test_no_match_reports_honestly_and_changes_nothing(db):
    before = _ids(db)
    out = db.delete_entries(entry_date="2020-01-01")
    assert "nothing was deleted" in out.lower()
    assert _ids(db) == before


def test_result_names_every_row_so_the_readback_is_checkable(db):
    out = db.delete_entries(entry_date="2026-08-31", type="calories")
    assert "#2" in out and "#3" in out
    assert "Orange Chicken" in out and "Chow Mein" in out


def test_date_range_bounds_are_inclusive(db):
    db.delete_entries(start_date="2026-06-01", end_date="2026-06-30")
    assert 1 not in _ids(db)
    assert _ids(db) == [2, 3, 4]


def test_tool_exposes_the_filter_arguments(tmp_path):
    tool = LogTools(TrackerDB(str(tmp_path / "t.db"))).create_delete_entries_tool()
    assert tool.name == "delete_entries"
    for arg in ("entry_date", "start_date", "end_date", "type", "note_contains"):
        assert arg in tool.args, arg
    assert "entry_id" not in tool.args, "must not invite an id"
