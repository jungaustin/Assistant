"""Google Calendar client. Wraps `google-api-python-client` so the
StructuredTool wrappers in `calendar_tools.py` stay free of API
boilerplate.

Auth state lives on disk in two files (see config.GOOGLE_CALENDAR_*_PATH):
- credentials.json — the OAuth client identity (downloaded once from
  the Google Cloud Console by the user).
- token.json — the user's refresh token, written by
  `setup/google_calendar_oauth.py` after first browser auth, refreshed
  automatically here.

This module never opens a browser. If token.json is missing or
unrefreshable, every call returns a friendly error string the LLM can
read back, prompting the user to run `just oauth-calendar`.

Time handling: `list_events` accepts a small set of named windows
(today / tomorrow / this_week / next_week). `add_event` requires an ISO
8601 datetime from the LLM — the model is reliably good at converting
"Wednesday at 7pm" → "2026-06-10T19:00:00", and keeping fuzzy parsing
out of the tool API keeps the contract predictable.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from robot.config import (
    GOOGLE_CALENDAR_CREDENTIALS_PATH,
    GOOGLE_CALENDAR_ID,
    GOOGLE_CALENDAR_TOKEN_PATH,
)

logger = logging.getLogger(__name__)

# Same scope as the bootstrap script — read + write events.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


class CalendarNotConfigured(Exception):
    """Raised by the runtime client when token.json is missing. The
    tool wrappers catch this and return a friendly string the LLM can
    speak ("Calendar isn't set up yet — run `just oauth-calendar`")."""


class GoogleCalendar:
    """Stateful client. Loads + refreshes credentials lazily on first use.

    Injectable for tests: pass `service=<mock>` to skip the credential
    dance entirely. Otherwise the runtime path loads token.json from
    disk, refreshes if expired, and builds the API service.
    """

    def __init__(self, service=None, calendar_id: Optional[str] = None):
        self._service = service  # populated lazily by _ensure_service
        self.calendar_id = calendar_id or GOOGLE_CALENDAR_ID

    # ---------- Auth ----------

    def _ensure_service(self):
        """Lazy-load the Google API service. Raises CalendarNotConfigured
        if the user hasn't run the OAuth bootstrap yet."""
        if self._service is not None:
            return self._service
        creds = self._load_credentials()
        # cache_discovery=False avoids a noisy ImportError warning from
        # googleapiclient when running without `oauth2client` (newer
        # versions use google-auth instead).
        self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def _load_credentials(self) -> Credentials:
        token_path = Path(GOOGLE_CALENDAR_TOKEN_PATH)
        if not token_path.exists():
            raise CalendarNotConfigured(
                f"No token at {token_path}. "
                f"Run `just oauth-calendar` to authenticate."
            )
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                raise CalendarNotConfigured(
                    f"Token at {token_path} is no longer valid ({e}). "
                    f"Re-run `just oauth-calendar`."
                )
            # Persist the refreshed token so we don't refresh every call.
            token_path.write_text(creds.to_json())
        return creds

    # ---------- Public API ----------

    def list_events(self, when: str = "today") -> str:
        """Return events in the named window as a newline-joined string
        ready for the LLM to read back, or a 'nothing' string if empty.

        Friendly errors (CalendarNotConfigured, API failures) come back
        as strings too — the LLM should be able to speak ANY failure
        rather than the tool throwing into the agent loop.
        """
        try:
            time_min, time_max = _window_to_range(when)
            service = self._ensure_service()
            result = (
                service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
        except CalendarNotConfigured as e:
            return str(e)
        except HttpError as e:
            logger.exception("calendar list_events failed")
            return f"Calendar API error: {e.reason or e}"

        events = result.get("items", [])
        if not events:
            return f"Nothing on the calendar for {when.replace('_', ' ')}."
        return "\n".join(_format_event_line(ev) for ev in events)

    def add_event(
        self,
        title: str,
        start_iso: str,
        duration_minutes: int = 30,
    ) -> str:
        """Add an event. start_iso must be ISO 8601 (the persona examples
        show the LLM how). Returns 'Added: <title> at <when> (id=<id>)'
        so the LLM can confirm and optionally remember the id for later
        deletion."""
        try:
            start_dt = _parse_iso_to_aware(start_iso)
        except ValueError as e:
            return f"Couldn't parse start time {start_iso!r}: {e}"
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        body = {
            "summary": title,
            "start": {"dateTime": start_dt.isoformat()},
            "end": {"dateTime": end_dt.isoformat()},
        }
        try:
            service = self._ensure_service()
            created = (
                service.events()
                .insert(calendarId=self.calendar_id, body=body)
                .execute()
            )
        except CalendarNotConfigured as e:
            return str(e)
        except HttpError as e:
            logger.exception("calendar add_event failed")
            return f"Calendar API error: {e.reason or e}"

        return (
            f"Added: {title} at {start_dt.strftime('%Y-%m-%d %H:%M')} "
            f"(id={created['id']})"
        )

    def delete_event(self, event_id: str) -> str:
        """Delete by event id. The LLM must have a real id from a prior
        list_events / add_event call — if it hallucinates one, the API
        returns a clear 404 we relay."""
        try:
            service = self._ensure_service()
            service.events().delete(
                calendarId=self.calendar_id, eventId=event_id
            ).execute()
        except CalendarNotConfigured as e:
            return str(e)
        except HttpError as e:
            logger.exception("calendar delete_event failed (id=%s)", event_id)
            return f"Couldn't delete: {e.reason or e}"
        return "Deleted."


# ---------- Helpers ----------


def _window_to_range(when: str) -> tuple[str, str]:
    """Map a named window to (timeMin, timeMax) RFC3339 strings.
    Unknown windows fall back to 'today'."""
    when = (when or "today").strip().lower().replace(" ", "_")
    today = date.today()

    def _bounds(start: date, end_inclusive: date) -> tuple[str, str]:
        # Local-time start of `start` to local-time end of `end_inclusive`,
        # both stamped with local tz so Google interprets them correctly.
        tz = datetime.now().astimezone().tzinfo
        t_min = datetime.combine(start, time.min, tzinfo=tz)
        t_max = datetime.combine(end_inclusive, time.max, tzinfo=tz)
        return t_min.isoformat(), t_max.isoformat()

    if when == "tomorrow":
        d = today + timedelta(days=1)
        return _bounds(d, d)
    if when in ("this_week", "week"):
        return _bounds(today, today + timedelta(days=6))
    if when == "next_week":
        start = today + timedelta(days=7)
        return _bounds(start, start + timedelta(days=6))
    # Default + explicit "today"
    return _bounds(today, today)


def _parse_iso_to_aware(start_iso: str) -> datetime:
    """Parse ISO 8601 → aware datetime. If the input is naive (no tz),
    stamp it with the local timezone (matches what 'add a 7pm reminder'
    actually means to the user)."""
    dt = datetime.fromisoformat(start_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def _format_event_line(event: dict) -> str:
    """One event → one line for the LLM. Drops cruft, keeps
    title / start / id (so a follow-up delete can name the id)."""
    title = event.get("summary", "(untitled)")
    start = event.get("start", {})
    # All-day events use 'date'; timed events use 'dateTime'.
    when = start.get("dateTime") or start.get("date") or "?"
    event_id = event.get("id", "?")
    return f"{when} — {title} (id={event_id})"
