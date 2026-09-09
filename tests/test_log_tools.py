"""Tests for the personal data logger (TrackerDB + LogTools).

Covers:
  - log_entry defaults entry_date to today
  - log_entry accepts an explicit entry_date (retroactive logging)
  - query_entries returns correct rows for a date range
  - query_entries returns [] when nothing is logged
  - query_entries filters by type
  - upsert_period_note creates on first call
  - upsert_period_note appends on second call (default replace=False)
  - upsert_period_note replaces when replace=True
  - get_period_note returns the note
  - get_period_note returns a 'No note' message when nothing is saved
  - TrackerDB persists across instances
  - LogTools exposes all 4 tool names
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from robot.tools.inner.log import FoodItem, LogTools, TrackerDB


def _db(tmp_path: Path) -> TrackerDB:
    return TrackerDB(str(tmp_path / "tracker.db"))


# ---------------------------------------------------------------------------
# log_entry
# ---------------------------------------------------------------------------


def test_log_entry_defaults_to_today(tmp_path: Path):
    db = _db(tmp_path)
    result = db.log_entry(type="calories", value=500)
    today = date.today().isoformat()
    assert today in result
    assert "calories" in result
    db.close()


def test_log_entry_returns_id_update_entry_can_target(tmp_path: Path):
    """The '#id' is what lets an immediate correction move the row it just
    created instead of logging a duplicate ('sorry, make that yesterday')."""
    db = _db(tmp_path)
    result = db.log_entry(type="calories", value=160, note="Takis")
    entry_id = int(result.split("#")[1].split(":")[0])

    db.update_entry(entry_id=entry_id, entry_date="2026-08-09")

    assert db.query_entries(type="calories") == "[]"  # gone from today
    moved = db.query_entries(type="calories", start_date="2026-08-09", end_date="2026-08-09")
    assert "Takis" in moved
    assert len(moved.splitlines()) == 1  # moved, not duplicated
    db.close()


def test_log_entry_explicit_entry_date(tmp_path: Path):
    db = _db(tmp_path)
    result = db.log_entry(type="sleep", value=7.5, entry_date="2026-06-01")
    assert "2026-06-01" in result
    assert "sleep" in result
    db.close()


def test_log_entry_text_only(tmp_path: Path):
    db = _db(tmp_path)
    result = db.log_entry(type="exercise", note="5k run")
    assert "exercise" in result
    db.close()


def test_log_entry_normalizes_type_to_lowercase(tmp_path: Path):
    db = _db(tmp_path)
    result = db.log_entry(type="Calories", value=300)
    assert "calories" in result
    db.close()


# ---------------------------------------------------------------------------
# query_entries
# ---------------------------------------------------------------------------


def test_query_entries_returns_empty_when_nothing_logged(tmp_path: Path):
    db = _db(tmp_path)
    result = db.query_entries(start_date="2026-01-01", end_date="2026-01-31")
    assert result == "[]"
    db.close()


def test_query_entries_finds_logged_entry(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=600, entry_date="2026-06-05")
    result = db.query_entries(start_date="2026-06-05", end_date="2026-06-05")
    assert "calories" in result
    assert "600" in result
    assert result != "[]"
    db.close()


def test_query_entries_date_range_filters_correctly(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="sleep", value=8.0, entry_date="2026-06-01")
    db.log_entry(type="sleep", value=7.0, entry_date="2026-06-03")
    db.log_entry(type="sleep", value=6.5, entry_date="2026-06-10")

    result = db.query_entries(type="sleep", start_date="2026-06-01", end_date="2026-06-05")
    assert "2026-06-01" in result
    assert "2026-06-03" in result
    assert "2026-06-10" not in result
    db.close()


def test_query_entries_type_filter(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=500, entry_date="2026-06-05")
    db.log_entry(type="sleep", value=7.0, entry_date="2026-06-05")

    result = db.query_entries(type="calories", start_date="2026-06-05", end_date="2026-06-05")
    assert "calories" in result
    assert "sleep" not in result
    db.close()


def test_query_entries_no_type_returns_all(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=500, entry_date="2026-06-05")
    db.log_entry(type="sleep", value=7.0, entry_date="2026-06-05")

    result = db.query_entries(start_date="2026-06-05", end_date="2026-06-05")
    assert "calories" in result
    assert "sleep" in result
    db.close()


# ---------------------------------------------------------------------------
# entry_stats
# ---------------------------------------------------------------------------


def test_entry_stats_averages_per_day_not_per_entry(tmp_path: Path):
    db = _db(tmp_path)
    # 3 entries across 2 days: per-entry avg would be 400, per-day is 600.
    db.log_entry(type="calories", value=500, entry_date="2026-06-01")
    db.log_entry(type="calories", value=100, entry_date="2026-06-01")
    db.log_entry(type="calories", value=600, entry_date="2026-06-02")
    result = db.entry_stats(type="calories", start_date="2026-06-01", end_date="2026-06-02")
    assert "total: 1200" in result
    assert "entries: 3" in result
    assert "days with logs: 2" in result
    assert "average per logged day: 600" in result
    assert "400" not in result
    db.close()


def test_entry_stats_calendar_days_span_gaps_to_end_date(tmp_path: Path):
    db = _db(tmp_path)
    # Logged on 2 of the 5 days ending at end_date; skipped days must still
    # widen the calendar denominator (1200/5), not shrink it to the last log.
    db.log_entry(type="calories", value=600, entry_date="2026-06-01")
    db.log_entry(type="calories", value=600, entry_date="2026-06-03")
    result = db.entry_stats(type="calories", start_date="2026-06-01", end_date="2026-06-05")
    assert "days with logs: 2" in result
    assert "calendar days since first log: 5" in result
    assert "average per calendar day: 240" in result
    db.close()


def test_entry_stats_defaults_to_all_time(tmp_path: Path):
    db = _db(tmp_path)
    today = date.today().isoformat()
    db.log_entry(type="calories", value=300, entry_date="2026-06-01")
    db.log_entry(type="calories", value=700, entry_date=today)
    result = db.entry_stats(type="calories")
    assert "2026-06-01" in result
    assert "total: 1000" in result
    db.close()


def test_entry_stats_reports_highest_and_lowest_day(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=2000, entry_date="2026-06-01")
    db.log_entry(type="calories", value=300, entry_date="2026-06-02")
    result = db.entry_stats(type="calories", start_date="2026-06-01", end_date="2026-06-02")
    assert "highest day: 2026-06-01 (2000)" in result
    assert "lowest day: 2026-06-02 (300)" in result
    db.close()


def test_entry_stats_no_entries(tmp_path: Path):
    db = _db(tmp_path)
    result = db.entry_stats(type="calories")
    assert "No calories entries" in result
    db.close()


def test_entry_stats_ignores_other_types(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=500, entry_date="2026-06-01")
    db.log_entry(type="sleep", value=8.0, entry_date="2026-06-01")
    result = db.entry_stats(type="calories", start_date="2026-06-01", end_date="2026-06-01")
    assert "total: 500" in result
    db.close()


# ---------------------------------------------------------------------------
# upsert_period_note / get_period_note
# ---------------------------------------------------------------------------


def test_upsert_creates_on_first_call(tmp_path: Path):
    db = _db(tmp_path)
    db.upsert_period_note("week", "2026-06-02", "Good week overall")
    note = db.get_period_note("week", "2026-06-02")
    assert note == "Good week overall"
    db.close()


def test_upsert_appends_on_second_call(tmp_path: Path):
    db = _db(tmp_path)
    db.upsert_period_note("week", "2026-06-02", "First note")
    db.upsert_period_note("week", "2026-06-02", "Second note")
    note = db.get_period_note("week", "2026-06-02")
    assert "First note" in note
    assert "Second note" in note
    assert " | " in note
    db.close()


def test_upsert_replaces_when_flag_set(tmp_path: Path):
    db = _db(tmp_path)
    db.upsert_period_note("day", "2026-06-05", "Original")
    db.upsert_period_note("day", "2026-06-05", "Replacement", replace=True)
    note = db.get_period_note("day", "2026-06-05")
    assert note == "Replacement"
    assert "Original" not in note
    db.close()


def test_upsert_different_periods_are_independent(tmp_path: Path):
    db = _db(tmp_path)
    db.upsert_period_note("week", "2026-06-02", "Week note")
    db.upsert_period_note("month", "2026-06-01", "Month note")
    assert db.get_period_note("week", "2026-06-02") == "Week note"
    assert db.get_period_note("month", "2026-06-01") == "Month note"
    db.close()


def test_get_period_note_missing_returns_no_note_message(tmp_path: Path):
    db = _db(tmp_path)
    result = db.get_period_note("day", "2026-01-01")
    assert "No note" in result
    db.close()


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


def test_persists_across_db_instances(tmp_path: Path):
    path = str(tmp_path / "tracker.db")
    db1 = TrackerDB(path)
    db1.log_entry(type="weight", value=72.5, entry_date="2026-06-05")
    db1.close()

    db2 = TrackerDB(path)
    result = db2.query_entries(type="weight", start_date="2026-06-05", end_date="2026-06-05")
    assert "72.5" in result
    db2.close()


# ---------------------------------------------------------------------------
# LogTools wiring
# ---------------------------------------------------------------------------


def test_log_tools_exposes_all_tools(tmp_path: Path):
    db = _db(tmp_path)
    lt = LogTools(db)
    tool_names = {
        lt.create_log_entry_tool().name,
        lt.create_query_entries_tool().name,
        lt.create_entry_stats_tool().name,
        lt.create_update_entry_tool().name,
        lt.create_delete_entry_tool().name,
        lt.create_upsert_period_note_tool().name,
        lt.create_get_period_note_tool().name,
        lt.create_log_meal_tool().name,
        lt.create_lookup_food_tool().name,
    }
    assert tool_names == {
        "log_meal",
        "lookup_food",
        "log_entry",
        "query_entries",
        "entry_stats",
        "update_entry",
        "delete_entry",
        "upsert_period_note",
        "get_period_note",
    }
    db.close()


# ---------------------------------------------------------------------------
# log_meal
# ---------------------------------------------------------------------------


def test_log_meal_inserts_one_row_per_item(tmp_path: Path):
    db = _db(tmp_path)
    result = db.log_meal(
        items=[
            FoodItem(name="orange chicken", calories=490),
            FoodItem(name="chow mein", calories=510),
        ],
        entry_date="2026-06-05",
    )
    assert "total: 1000" in result
    rows = db.query_entries(type="calories", start_date="2026-06-05", end_date="2026-06-05")
    assert "orange chicken" in rows
    assert "chow mein" in rows
    assert len(rows.splitlines()) == 2
    db.close()


def test_log_meal_returns_ids_update_entry_can_target(tmp_path: Path):
    """The readback's #ids must let a single item be corrected in place."""
    db = _db(tmp_path)
    result = db.log_meal(
        items=[
            FoodItem(name="orange chicken", calories=490),
            FoodItem(name="chow mein", calories=510),
        ],
        entry_date="2026-06-05",
    )
    chow_line = [ln for ln in result.splitlines() if "chow mein" in ln][0]
    chow_id = int(chow_line.split("#")[1].split(" ")[0])
    db.update_entry(entry_id=chow_id, value=770)
    rows = db.query_entries(type="calories", start_date="2026-06-05", end_date="2026-06-05")
    assert "770" in rows
    assert "490" in rows  # the other item is untouched
    db.close()


def test_log_meal_defaults_to_today(tmp_path: Path):
    db = _db(tmp_path)
    result = db.log_meal(items=[FoodItem(name="rice", calories=350)])
    assert date.today().isoformat() in result
    db.close()


def test_log_meal_empty_items(tmp_path: Path):
    db = _db(tmp_path)
    assert "No items" in db.log_meal(items=[])
    db.close()


# ---------------------------------------------------------------------------
# lookup_food
# ---------------------------------------------------------------------------


def test_lookup_food_finds_exact_phrase(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=350, note="bowl of rice", entry_date="2026-06-01")
    db.log_entry(type="calories", value=370, note="bowl of rice", entry_date="2026-06-02")
    result = db.lookup_food("bowl of rice")
    assert "2 logs" in result
    assert "360" in result
    assert "2026-06-02" in result
    db.close()


def test_lookup_food_falls_back_to_content_words(tmp_path: Path):
    """'a bowl of kimchi rice' was never logged verbatim; the words still hit."""
    db = _db(tmp_path)
    db.log_entry(type="calories", value=850, note="kimchi fried rice", entry_date="2026-06-01")
    result = db.lookup_food("a bowl of kimchi rice")
    assert "kimchi fried rice" in result
    assert "850" in result
    db.close()


def test_lookup_food_phrase_match_wins_over_word_match(tmp_path: Path):
    """An exact 'rice' phrase hit must not drag in unrelated rice dishes."""
    db = _db(tmp_path)
    db.log_entry(type="calories", value=350, note="rice", entry_date="2026-06-01")
    db.log_entry(type="calories", value=900, note="chicken burrito", entry_date="2026-06-02")
    result = db.lookup_food("rice")
    assert "burrito" not in result
    db.close()


def test_lookup_food_ignores_non_calorie_entries(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="exercise", value=30, note="rice field walk", entry_date="2026-06-01")
    assert "No past calorie logs" in db.lookup_food("rice")
    db.close()


def test_lookup_food_no_match(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=350, note="rice", entry_date="2026-06-01")
    assert "No past calorie logs" in db.lookup_food("tiramisu")
    db.close()


def test_lookup_food_empty_query(tmp_path: Path):
    db = _db(tmp_path)
    assert "No food given" in db.lookup_food("   ")
    db.close()


# ---------------------------------------------------------------------------
# update_entry / delete_entry (correcting a mis-heard log)
# ---------------------------------------------------------------------------


def test_query_entries_surfaces_id(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=800, note="lunch", entry_date="2026-06-05")
    result = db.query_entries(start_date="2026-06-05", end_date="2026-06-05")
    # The agent needs the '#id' prefix to target a specific row.
    assert result.startswith("#")


def test_update_entry_defaults_to_latest(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=800, note="lunch", entry_date="2026-06-05")
    result = db.update_entry(value=700)
    assert "700" in result
    rows = db.query_entries(type="calories", start_date="2026-06-05", end_date="2026-06-05")
    assert "700" in rows
    assert "800" not in rows
    db.close()


def test_update_entry_only_changes_provided_fields(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=800, note="lunch", entry_date="2026-06-05")
    db.update_entry(value=700)  # note must survive
    rows = db.query_entries(type="calories", start_date="2026-06-05", end_date="2026-06-05")
    assert "lunch" in rows
    db.close()


def test_update_entry_by_id(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="sleep", value=8.0, entry_date="2026-06-05")
    db.log_entry(type="calories", value=500, entry_date="2026-06-05")  # latest
    # Without an id this would hit calories; entry_id=1 targets the sleep row.
    db.update_entry(entry_id=1, value=7.0)
    sleep_rows = db.query_entries(type="sleep", start_date="2026-06-05", end_date="2026-06-05")
    assert "7.0" in sleep_rows
    cal_rows = db.query_entries(type="calories", start_date="2026-06-05", end_date="2026-06-05")
    assert "500" in cal_rows  # untouched
    db.close()


def test_update_entry_match_type_targets_latest_of_type(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="sleep", value=8.0, entry_date="2026-06-05")
    db.log_entry(type="calories", value=500, entry_date="2026-06-05")  # latest overall
    db.update_entry(match_type="sleep", value=7.0)
    sleep_rows = db.query_entries(type="sleep", start_date="2026-06-05", end_date="2026-06-05")
    assert "7.0" in sleep_rows
    db.close()


def test_update_entry_requires_a_field(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=800, entry_date="2026-06-05")
    result = db.update_entry()
    assert "Nothing to change" in result
    db.close()


def test_update_entry_no_match(tmp_path: Path):
    db = _db(tmp_path)
    result = db.update_entry(value=700)
    assert "No matching entry" in result
    db.close()


def test_delete_entry_defaults_to_latest(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="calories", value=800, entry_date="2026-06-05")
    result = db.delete_entry()
    assert "800" in result
    rows = db.query_entries(start_date="2026-06-05", end_date="2026-06-05")
    assert rows == "[]"
    db.close()


def test_delete_entry_match_type(tmp_path: Path):
    db = _db(tmp_path)
    db.log_entry(type="sleep", value=8.0, entry_date="2026-06-05")
    db.log_entry(type="calories", value=500, entry_date="2026-06-05")
    db.delete_entry(match_type="sleep")
    rows = db.query_entries(start_date="2026-06-05", end_date="2026-06-05")
    assert "sleep" not in rows
    assert "calories" in rows
    db.close()


def test_delete_entry_no_match(tmp_path: Path):
    db = _db(tmp_path)
    result = db.delete_entry()
    assert "No matching entry" in result
    db.close()
