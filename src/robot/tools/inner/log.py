"""Personal data logger tools for Nemo.

Two SQLite tables:
  entries       — structured numeric/text entries (calories, exercise, sleep, mood, etc.)
  period_notes  — free-text notes anchored to a day, week, or month

entry_date vs created_at: entry_date is WHEN the thing happened (the query axis);
created_at is when Nemo logged it. This lets "I had lunch at noon" logged at 10pm
land on the right date.

DB file: state/tracker.db (alongside conversations.db and memory.db).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


class FoodItem(BaseModel):
    """One item in a meal, for log_meal's batch insert."""

    name: str = Field(description="What the food is, e.g. 'orange chicken'")
    calories: float = Field(description="Calories for the portion actually eaten")


# Words stripped before the word-level fallback in lookup_food. A query like
# "bowl of rice" should still find "rice and soup" once the phrase match
# misses, and these carry no signal about what the food is.
_FOOD_STOPWORDS = {
    "a", "an", "and", "the", "of", "with", "some", "my", "for",
    "bowl", "plate", "cup", "slice", "piece", "serving", "order",
}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    entry_date  TEXT NOT NULL,
    type        TEXT NOT NULL,
    value       REAL,
    note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_date ON entries(entry_date);
CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(type);

CREATE TABLE IF NOT EXISTS period_notes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    period_type  TEXT NOT NULL,
    period_start TEXT NOT NULL,
    content      TEXT NOT NULL DEFAULT '',
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(period_type, period_start)
);
"""


class TrackerDB:
    """SQLite-backed personal data store.

    Thread-safe: one connection shared across tool calls, guarded by a lock.
    check_same_thread=False matches the pattern in MemoryStore and agent.py.
    """

    def __init__(self, db_path: str):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def log_entry(
        self,
        type: str,
        value: Optional[float] = None,
        note: Optional[str] = None,
        entry_date: Optional[str] = None,
    ) -> str:
        """Insert one data point. Returns a confirmation string."""
        if not entry_date:
            entry_date = date.today().isoformat()
        canonical_type = type.strip().lower()
        with self._lock:
            self._conn.execute(
                "INSERT INTO entries (entry_date, type, value, note) VALUES (?, ?, ?, ?)",
                (entry_date, canonical_type, value, note or None),
            )
            self._conn.commit()
        value_str = str(value) if value is not None else repr(note)
        return f"Logged: {canonical_type} = {value_str} on {entry_date}."

    def log_meal(
        self,
        items: list[FoodItem],
        entry_date: Optional[str] = None,
    ) -> str:
        """Insert one calories row per item in a single transaction.

        Exists so "I had orange chicken, chow mein, and a spring roll" is one
        tool call instead of three sequential log_entry round-trips — the
        latency matters over voice. Logging per item rather than as one lump
        sum is also what makes lookup_food useful later: a 'Panda Express |
        1900' row teaches nothing, three named rows teach three foods.
        """
        if not items:
            return "No items to log."
        if not entry_date:
            entry_date = date.today().isoformat()

        rows = [
            (entry_date, "calories", float(item.calories), item.name.strip() or None)
            for item in items
        ]
        # One execute per row rather than executemany, purely to collect each
        # lastrowid: the '#id' in the readback is what lets update_entry fix a
        # single item ("the chow mein was a large") instead of the meal's last
        # row, which is all update_entry's default targeting could find.
        # Still one transaction — the commit happens after the loop.
        with self._lock:
            ids = []
            for row in rows:
                cur = self._conn.execute(
                    "INSERT INTO entries (entry_date, type, value, note) VALUES (?, ?, ?, ?)",
                    row,
                )
                ids.append(cur.lastrowid)
            self._conn.commit()

        total = sum(r[2] for r in rows)

        def fmt(x: float) -> str:
            return str(int(x)) if float(x) == int(x) else f"{x:.1f}"

        lines = [f"Logged {len(rows)} items on {entry_date}:"]
        lines += [f"  #{i} | {r[3]} = {fmt(r[2])}" for i, r in zip(ids, rows)]
        lines.append(f"  total: {fmt(total)}")
        return "\n".join(lines)

    def lookup_food(self, food: str, limit: int = 5) -> str:
        """Find what this food was logged as before. Returns a summary table.

        Three passes, widening only when the previous one comes up empty:
        the whole phrase, then all content words, then any content word. This
        keeps 'rice' from returning every rice dish when 'kimchi fried rice'
        is what was asked for, while still finding something for a phrasing
        that was never logged verbatim.
        """
        term = food.strip().lower()
        if not term:
            return "No food given to look up."

        def _search(where: str, params: list) -> list:
            return self._conn.execute(
                "SELECT LOWER(note) AS food, COUNT(*), AVG(value), MIN(value), "
                "MAX(value), MAX(entry_date) FROM entries "
                f"WHERE type = 'calories' AND value IS NOT NULL AND note IS NOT NULL AND {where} "
                "GROUP BY food ORDER BY MAX(entry_date) DESC, COUNT(*) DESC LIMIT ?",
                params + [limit],
            ).fetchall()

        words = [w for w in term.split() if w not in _FOOD_STOPWORDS and len(w) > 2]

        with self._lock:
            rows = _search("LOWER(note) LIKE ?", [f"%{term}%"])
            if not rows and words:
                clause = " AND ".join("LOWER(note) LIKE ?" for _ in words)
                rows = _search(clause, [f"%{w}%" for w in words])
            if not rows and len(words) > 1:
                clause = " OR ".join("LOWER(note) LIKE ?" for _ in words)
                rows = _search(clause, [f"%{w}%" for w in words])

        if not rows:
            return f"No past calorie logs matching '{food}'."

        def fmt(x: float) -> str:
            return str(int(x)) if float(x) == int(x) else f"{x:.1f}"

        lines = [f"Past logs matching '{food}':"]
        for note, n, avg, lo, hi, last in rows:
            times = "1 log" if n == 1 else f"{n} logs"
            spread = fmt(avg) if lo == hi else f"{fmt(avg)} (range {fmt(lo)}–{fmt(hi)})"
            lines.append(f"  {note} — {times}, usually {spread}, last on {last}")
        return "\n".join(lines)

    def query_entries(
        self,
        type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Return matching entries as a newline-joined table, or '[]' if empty."""
        today = date.today().isoformat()
        start = start_date or today
        end = end_date or today

        with self._lock:
            if type:
                rows = self._conn.execute(
                    "SELECT id, entry_date, type, value, note FROM entries "
                    "WHERE type = ? AND entry_date BETWEEN ? AND ? ORDER BY entry_date, id",
                    (type.strip().lower(), start, end),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, entry_date, type, value, note FROM entries "
                    "WHERE entry_date BETWEEN ? AND ? ORDER BY entry_date, type, id",
                    (start, end),
                ).fetchall()

        if not rows:
            return "[]"

        lines = []
        for eid, entry_date, etype, val, note in rows:
            # Lead with '#id' so update_entry / delete_entry can target a row.
            parts = [f"#{eid}", entry_date, etype]
            if val is not None:
                parts.append(str(val))
            if note:
                parts.append(note)
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def entry_stats(
        self,
        type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Aggregate one entry type in SQL: totals and per-day averages.

        Exists because the model must never do this arithmetic itself —
        summing 100 rows of query_entries output by hand is where the
        wrong averages were coming from. Defaults to all-time (first log
        of this type through today).
        """
        canonical_type = type.strip().lower()
        end = end_date or date.today().isoformat()

        with self._lock:
            start = start_date
            if not start:
                row = self._conn.execute(
                    "SELECT MIN(entry_date) FROM entries WHERE type = ?",
                    (canonical_type,),
                ).fetchone()
                start = row[0]
            if not start:
                return f"No {canonical_type} entries logged yet."

            total, n_entries, days_logged, first, last = self._conn.execute(
                "SELECT COALESCE(SUM(value), 0), COUNT(*), COUNT(DISTINCT entry_date), "
                "MIN(entry_date), MAX(entry_date) FROM entries "
                "WHERE type = ? AND entry_date BETWEEN ? AND ?",
                (canonical_type, start, end),
            ).fetchone()
            if n_entries == 0:
                return f"No {canonical_type} entries between {start} and {end}."

            day_totals = self._conn.execute(
                "SELECT entry_date, SUM(value) FROM entries "
                "WHERE type = ? AND entry_date BETWEEN ? AND ? AND value IS NOT NULL "
                "GROUP BY entry_date ORDER BY SUM(value) DESC",
                (canonical_type, start, end),
            ).fetchall()

        # Calendar span runs first log → today (not → last log): a streak
        # with missed recent days should drag the per-calendar-day average
        # down, not silently shrink the denominator.
        span_end = max(date.fromisoformat(end), date.fromisoformat(last))
        calendar_days = (span_end - date.fromisoformat(first)).days + 1

        def fmt(x: float) -> str:
            return str(int(x)) if float(x) == int(x) else f"{x:.1f}"

        lines = [
            f"{canonical_type} stats, {first} to {last}:",
            f"  total: {fmt(total)}",
            f"  entries: {n_entries}",
            f"  days with logs: {days_logged}",
            f"  calendar days since first log: {calendar_days}",
            f"  average per logged day: {fmt(total / days_logged)}",
            f"  average per calendar day: {fmt(total / calendar_days)}",
        ]
        if day_totals:
            hi_date, hi = day_totals[0]
            lo_date, lo = day_totals[-1]
            lines.append(f"  highest day: {hi_date} ({fmt(hi)})")
            lines.append(f"  lowest day: {lo_date} ({fmt(lo)})")
        return "\n".join(lines)

    def _select_target(
        self,
        entry_id: Optional[int],
        match_type: Optional[str],
    ):
        """Resolve which entry update/delete should act on. Caller holds the lock.

        Precedence: an explicit entry_id wins; otherwise the most recently
        logged entry (highest id), optionally narrowed to a single type so
        "fix my sleep" finds the latest sleep row even if something else was
        logged afterward. Returns the row tuple or None.
        """
        if entry_id is not None:
            return self._conn.execute(
                "SELECT id, entry_date, type, value, note FROM entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        if match_type:
            return self._conn.execute(
                "SELECT id, entry_date, type, value, note FROM entries "
                "WHERE type = ? ORDER BY id DESC LIMIT 1",
                (match_type.strip().lower(),),
            ).fetchone()
        return self._conn.execute(
            "SELECT id, entry_date, type, value, note FROM entries "
            "ORDER BY id DESC LIMIT 1",
        ).fetchone()

    def update_entry(
        self,
        entry_id: Optional[int] = None,
        value: Optional[float] = None,
        note: Optional[str] = None,
        entry_date: Optional[str] = None,
        match_type: Optional[str] = None,
    ) -> str:
        """Correct a previously logged entry. Returns a confirmation string.

        Targets the most recently logged entry by default (the common
        "Nemo misheard what I just said" case); pass entry_id to target a
        specific row, or match_type to grab the latest entry of one type.
        Only the fields you pass (value/note/entry_date) are changed.
        """
        with self._lock:
            row = self._select_target(entry_id, match_type)
            if not row:
                return "No matching entry to update."
            eid, old_date, old_type, old_value, old_note = row

            sets: list[str] = []
            params: list = []
            changes: list[str] = []
            if value is not None:
                sets.append("value = ?")
                params.append(value)
                changes.append(f"value {old_value} → {value}")
            if note is not None:
                sets.append("note = ?")
                params.append(note or None)
                changes.append(f"note {old_note!r} → {note!r}")
            if entry_date is not None:
                sets.append("entry_date = ?")
                params.append(entry_date)
                changes.append(f"date {old_date} → {entry_date}")

            if not sets:
                return "Nothing to change — provide value, note, and/or entry_date."

            params.append(eid)
            self._conn.execute(
                f"UPDATE entries SET {', '.join(sets)} WHERE id = ?", params
            )
            self._conn.commit()

        return f"Updated entry #{eid} ({old_type}): " + "; ".join(changes) + "."

    def delete_entry(
        self,
        entry_id: Optional[int] = None,
        match_type: Optional[str] = None,
    ) -> str:
        """Delete a logged entry outright. Returns a confirmation string.

        Defaults to the most recently logged entry; pass entry_id to target a
        specific row, or match_type to remove the latest entry of one type.
        """
        with self._lock:
            row = self._select_target(entry_id, match_type)
            if not row:
                return "No matching entry to delete."
            eid, edate, etype, eval_, enote = row
            self._conn.execute("DELETE FROM entries WHERE id = ?", (eid,))
            self._conn.commit()

        desc = etype
        if eval_ is not None:
            desc += f" = {eval_}"
        if enote:
            desc += f" ({enote})"
        return f"Deleted entry #{eid}: {desc} on {edate}."

    def upsert_period_note(
        self,
        period_type: str,
        period_start: str,
        content: str,
        replace: bool = False,
    ) -> str:
        """Append (or replace) a period note. Returns a confirmation string.

        Default behavior appends content to the existing note separated by ' | '.
        Pass replace=True to overwrite the existing note entirely.
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self._conn.execute(
                "SELECT content FROM period_notes "
                "WHERE period_type = ? AND period_start = ?",
                (period_type, period_start),
            ).fetchone()

            new_content = content if (not existing or replace) else existing[0] + " | " + content

            self._conn.execute(
                "INSERT INTO period_notes (period_type, period_start, content, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(period_type, period_start) "
                "DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
                (period_type, period_start, new_content, ts),
            )
            self._conn.commit()
        return f"Saved {period_type} note starting {period_start}."

    def get_period_note(
        self,
        period_type: str,
        period_start: str,
    ) -> str:
        """Retrieve the note for a period. Returns the content or a 'no note' string."""
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM period_notes "
                "WHERE period_type = ? AND period_start = ?",
                (period_type, period_start),
            ).fetchone()
        if not row:
            return f"No note for {period_type} starting {period_start}."
        return row[0]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Phase 2 — nightly check-in prompt
# ---------------------------------------------------------------------------
# At ~10pm (time TBD), if certain entry types haven't been logged today,
# Nemo should proactively ask the user to log them.
#
# Implementation sketch:
#   - A scheduled job (cron via `schedule` lib or a background asyncio task
#     in main.py) fires at ~22:00 local time.
#   - It calls TrackerDB.query_entries(start_date=today, end_date=today) and
#     checks which of a configured "daily required" set are missing.
#     Example required set: {"calories", "exercise", "sleep"}.
#   - For each missing type, it synthesizes a voice prompt and publishes it
#     to the bus as a system-initiated utterance (same path as a user turn
#     so TTS picks it up).
#   - Example prompt: "Hey, you haven't logged sleep today — want to do that now?"
#   - If all types are covered, fire a short "all logged for today" affirmation
#     or stay silent (user preference TBD).
#
# Config knobs to add when building this:
#   DAILY_LOG_PROMPT_HOUR = 22   # local hour, 0-23
#   DAILY_REQUIRED_TYPES  = ["calories", "exercise", "sleep"]  # or env var
#
# The check itself is a one-liner using existing query_entries. Each row is
# "#id | date | type | ...", so the type lives at split index [2]:
#   logged_today = {row.split(" | ")[2] for row in
#                   db.query_entries(start_date=today, end_date=today).splitlines()
#                   if row != "[]"}
#   missing = set(DAILY_REQUIRED_TYPES) - logged_today
# ---------------------------------------------------------------------------


class LogTools:
    """LangChain StructuredTool wrappers around TrackerDB."""

    def __init__(self, db: TrackerDB):
        self.db = db

    def create_log_entry_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.log_entry,
            name="log_entry",
            description=(
                "Log a personal data point — calories, exercise, sleep, weight, mood, "
                "water, or anything else the user wants to track.\n\n"
                "Arguments:\n"
                "  type (str): what is being tracked. Use short lowercase labels like "
                "'calories', 'exercise', 'sleep', 'mood', 'weight', 'water'. "
                "Be consistent: use the same label the user used before.\n"
                "  value (float, optional): numeric amount. For calories: kcal. "
                "For sleep: hours. For weight: kg or lbs (match what the user says). "
                "For mood: 1–10 scale. Omit if the entry is text-only.\n"
                "  note (str, optional): free-text context — meal name, exercise type, "
                "mood description. Include this when the user gives qualitative detail.\n"
                "  entry_date (str, optional): ISO date (YYYY-MM-DD) for the day the "
                "thing HAPPENED. Defaults to today — OMIT it for anything happening "
                "today. Only set it when the user says the event itself was on another "
                "day ('yesterday I walked 5k', 'log this for Monday'). CRITICAL: if you "
                "just queried a past day and are now logging or copying that value to "
                "today, leave entry_date empty — do NOT reuse the date you looked up, or "
                "the entry lands on the wrong day.\n\n"
                "Examples:\n"
                "  'I had 600 calories for lunch' → type='calories', value=600, note='lunch'\n"
                "  'I ran 5k this morning' → type='exercise', note='5k run'\n"
                "  'I slept 7.5 hours' → type='sleep', value=7.5\n"
                "  'mood is a 6 today' → type='mood', value=6\n\n"
                "Returns 'Logged: ...' on success. Read back what you logged."
            ),
        )

    def create_lookup_food_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.lookup_food,
            name="lookup_food",
            description=(
                "Look up what a food was logged as BEFORE. ALWAYS call this "
                "first when the user names a food without giving a calorie "
                "number — before searching the web and before estimating "
                "yourself. Most of what the user eats repeats (rice, eggs, "
                "the usual pizza), and reusing the past number keeps the log "
                "self-consistent instead of re-guessing a new value each time.\n\n"
                "Arguments:\n"
                "  food (str): the food name, as the user said it — 'rice', "
                "'orange chicken', 'Sam's Club pizza'. One food per call; call "
                "this once per item when the user lists several.\n"
                "  limit (int, optional): max distinct foods to return, default 5.\n\n"
                "Returns past matches as 'food — N logs, usually X (range), "
                "last on DATE', or a 'No past calorie logs matching' message. "
                "If there is a clear match, reuse that number. If the matches "
                "are for a different food than the user meant, ignore them and "
                "fall back to lookup_food_calories or your own estimate."
            ),
        )

    def create_log_meal_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.log_meal,
            name="log_meal",
            description=(
                "Log several foods at once, one calories entry per item. Use "
                "this whenever the user lists more than one thing they ate — "
                "'I had orange chicken, chow mein, and a spring roll'. Prefer "
                "it over repeated log_entry calls: it is one round-trip, and "
                "it keeps each food as its own named row so lookup_food can "
                "find it next time. NEVER log a multi-item meal as a single "
                "lump sum like 'Panda Express = 1900'.\n\n"
                "Resolve each item's calories BEFORE calling this: lookup_food "
                "first, then lookup_food_calories for restaurant or packaged "
                "items, then your own estimate as a last resort.\n\n"
                "Arguments:\n"
                "  items (list): each item is {name, calories}. name is the "
                "food as the user said it; calories is a number for the "
                "portion actually eaten (double it if they had two).\n"
                "  entry_date (str, optional): ISO date (YYYY-MM-DD) for the "
                "day the meal was EATEN. Omit for today — same rules as "
                "log_entry.\n\n"
                "Returns an itemized list — each row led by '#id' — plus the "
                "total. Log first, then read the items and total back to the "
                "user; never ask them to confirm calories beforehand. Keep "
                "those ids: if the user corrects one item, pass its id to "
                "update_entry rather than re-logging the meal."
            ),
        )

    def create_query_entries_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.query_entries,
            name="query_entries",
            description=(
                "Query logged personal data entries. Use when the user asks 'how many "
                "calories did I have today', 'show me my sleep this week', 'what did I "
                "log yesterday', etc.\n\n"
                "ALWAYS call this tool for such questions — never answer from earlier "
                "messages in the conversation. The conversation can span several days "
                "and entries get added, corrected, or deleted outside it, so recalling "
                "from chat history gives stale or mixed-up-day answers. The log "
                "database is the only source of truth, even if the user mentioned the "
                "same food earlier in this chat.\n\n"
                "Arguments:\n"
                "  type (str, optional): filter by entry type ('calories', 'sleep', etc.). "
                "Omit to get all types for the date range.\n"
                "  start_date (str, optional): ISO date YYYY-MM-DD. Defaults to today.\n"
                "  end_date (str, optional): ISO date YYYY-MM-DD. Defaults to today.\n\n"
                "Returns a newline-joined table of entries in the format "
                "'date | type | value | note', or '[]' if nothing was logged. "
                "Summarize the results conversationally — don't read the raw table aloud.\n\n"
                "For totals, averages, or day counts spanning MORE THAN ONE DAY, do "
                "not use this tool and add rows up yourself — call entry_stats "
                "instead, which computes the math in SQL."
            ),
        )

    def create_entry_stats_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.entry_stats,
            name="entry_stats",
            description=(
                "Compute totals and averages for one logged entry type. ALWAYS use "
                "this for questions like 'what's my average daily calories', 'total "
                "calories this month', 'how many days have I logged'. NEVER compute "
                "these yourself by adding up query_entries rows — this tool does the "
                "math in SQL and its numbers are exact. Report its numbers verbatim.\n\n"
                "Arguments:\n"
                "  type (str, required): entry type, e.g. 'calories', 'sleep'.\n"
                "  start_date (str, optional): ISO date YYYY-MM-DD. Defaults to the "
                "first log of this type — so omitting both dates gives all-time stats, "
                "which is what 'my average' usually means.\n"
                "  end_date (str, optional): ISO date YYYY-MM-DD. Defaults to today.\n\n"
                "Returns: total, entry count, days with logs, calendar days since "
                "first log, average per logged day, average per calendar day, and "
                "the highest/lowest day. When the user asks for their 'daily "
                "average', give the average per logged day (per-day, NOT per-entry)."
            ),
        )

    def create_update_entry_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.update_entry,
            name="update_entry",
            description=(
                "Correct a personal data entry that was already logged — use when "
                "you misheard the user or they restate a value: 'that was 700 not "
                "800', 'actually that was lunch', 'no, log that for yesterday'.\n\n"
                "By default this edits the MOST RECENTLY logged entry, which is "
                "usually what 'that' / 'the last one' refers to right after logging. "
                "So a simple 'that was 700 not 800' needs only value=700.\n\n"
                "Arguments (all optional):\n"
                "  value (float): new numeric value.\n"
                "  note (str): new free-text note.\n"
                "  entry_date (str): new ISO date (YYYY-MM-DD) for when it happened.\n"
                "  entry_id (int): target a SPECIFIC entry by its id. Get ids from "
                "query_entries — each row starts with '#id'. Use this when 'the last "
                "entry' isn't the right one.\n"
                "  match_type (str): edit the latest entry OF THIS TYPE instead of "
                "the latest overall — e.g. match_type='sleep' to fix sleep even if "
                "calories was logged afterward.\n\n"
                "Pass at least one of value/note/entry_date (what to change). "
                "Returns 'Updated entry #N ...' describing the change — read it back."
            ),
        )

    def create_delete_entry_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.delete_entry,
            name="delete_entry",
            description=(
                "Remove a logged personal data entry entirely — use when the user "
                "says it shouldn't have been logged at all: 'scratch that', 'delete "
                "that last one', 'I didn't actually eat that'.\n\n"
                "By default this deletes the MOST RECENTLY logged entry. To target a "
                "different row, pass entry_id (from the '#id' shown by query_entries) "
                "or match_type to delete the latest entry of one type.\n\n"
                "Arguments (all optional):\n"
                "  entry_id (int): delete this specific entry id.\n"
                "  match_type (str): delete the latest entry of this type.\n\n"
                "Returns 'Deleted entry #N ...' describing what was removed. "
                "Prefer update_entry when the user wants to FIX a value, not erase it."
            ),
        )

    def create_upsert_period_note_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.upsert_period_note,
            name="upsert_period_note",
            description=(
                "Save a free-text note for a day, week, or month. Use when the user "
                "dictates a summary, reflection, or observation they want attached to "
                "a time period — 'note for this week: ...', 'add to today's log: ...'.\n\n"
                "Arguments:\n"
                "  period_type (str): 'day', 'week', or 'month'.\n"
                "  period_start (str): ISO date (YYYY-MM-DD) anchoring the period. "
                "For 'day': the day itself. For 'week': the Monday of that week "
                "(this week's Monday is in your context). For 'month': the first of "
                "the month (e.g. '2026-06-01').\n"
                "  content (str): the note text to save.\n"
                "  replace (bool, optional): if True, overwrite the existing note. "
                "Default False appends the new content to any existing note with ' | '.\n\n"
                "Returns 'Saved ...' on success."
            ),
        )

    def create_get_period_note_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.db.get_period_note,
            name="get_period_note",
            description=(
                "Retrieve the saved note for a day, week, or month. Use when the user "
                "asks 'what did I note for this week', 'read back today's log', etc.\n\n"
                "Arguments:\n"
                "  period_type (str): 'day', 'week', or 'month'.\n"
                "  period_start (str): ISO date anchoring the period (same rules as "
                "upsert_period_note: Monday for week, first of month for month).\n\n"
                "Returns the note text, or a 'No note' message if nothing is saved."
            ),
        )
