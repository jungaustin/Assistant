"""Nightly check-in: Nemo's first proactive behavior.

At DAILY_LOG_PROMPT_HOUR local time, look at today's tracker entries and
ask about any DAILY_REQUIRED_TYPES that haven't been logged yet — e.g.
"Hey, you haven't logged calories or sleep today — want to do that now?"
If everything's logged, stay silent (nagging beats nothing, silence beats
self-congratulation).

This implements the Phase 2 sketch at the bottom of tools/inner/log.py.
It is NOT a tool — the LLM never calls it. It's a background asyncio task
that main.py starts next to the Edge loop and that speaks through a
callback the Edge provides. Base architecture note: if it fires mid-
conversation the audio could overlap a reply; acceptable for a
once-a-day prompt at a quiet hour, revisit when the Conductor exists.

Disable by setting DAILY_LOG_PROMPT_HOUR=-1 or DAILY_REQUIRED_TYPES to
empty in .env.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable, Sequence

from robot.config import DAILY_LOG_PROMPT_HOUR, DAILY_REQUIRED_TYPES
from robot.core.logging import get_logger

log = get_logger(__name__)


def missing_daily_types(db, required: Sequence[str], day: str | None = None) -> list[str]:
    """Which of `required` have no tracker entry on `day` (default today).

    Preserves the order of `required` so the spoken prompt is stable.
    """
    day = day or date.today().isoformat()
    raw = db.query_entries(start_date=day, end_date=day)
    logged: set[str] = set()
    if raw != "[]":
        for line in raw.splitlines():
            # query_entries rows are "#id | date | type | value | note";
            # the type is at index 2.
            parts = line.split(" | ")
            if len(parts) >= 3:
                logged.add(parts[2])
    return [t for t in required if t not in logged]


def build_checkin_prompt(missing: Sequence[str]) -> str | None:
    """Voice-ready prompt for the missing types, or None if nothing is."""
    if not missing:
        return None
    if len(missing) == 1:
        things = missing[0]
    elif len(missing) == 2:
        things = f"{missing[0]} or {missing[1]}"
    else:
        things = ", ".join(missing[:-1]) + f", or {missing[-1]}"
    return f"Hey, you haven't logged {things} today — want to do that now?"


def seconds_until_hour(hour: int, now: datetime | None = None) -> float:
    """Seconds until the next local occurrence of HH:00."""
    now = now or datetime.now()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def checkin_loop(
    db,
    speak: Callable[[str], Awaitable[None]],
    *,
    hour: int | None = None,
    required: Sequence[str] | None = None,
) -> None:
    """Run forever; cancellation is the normal shutdown signal.

    `db` is a TrackerDB. `speak` is an async callable that voices one
    string (the Edge wires this to its TTS).
    """
    hour = DAILY_LOG_PROMPT_HOUR if hour is None else hour
    required = DAILY_REQUIRED_TYPES if required is None else list(required)
    if hour < 0 or not required:
        log.info("checkin_disabled", hour=hour, required=required)
        return

    while True:
        await asyncio.sleep(seconds_until_hour(hour))
        try:
            missing = missing_daily_types(db, required)
            prompt = build_checkin_prompt(missing)
            if prompt:
                log.info("checkin_prompt", missing=missing)
                await speak(prompt)
            else:
                log.info("checkin_all_logged")
        except asyncio.CancelledError:
            raise
        except Exception:
            # A failed check-in shouldn't kill the loop — try again
            # tomorrow.
            log.exception("checkin_failed")
        # Step past the top of the hour so seconds_until_hour targets
        # tomorrow, not a second firing tonight.
        await asyncio.sleep(61)
