"""LangChain StructuredTool wrappers around GoogleCalendar.

These are the surfaces the LLM sees. Descriptions matter more than code
here — they're how the model decides which tool to call. Keep them
example-heavy and unambiguous about format (especially the ISO datetime
contract for add_calendar_event).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from robot.tools.inner.google_calendar import GoogleCalendar


class CalendarTools:
    def __init__(self, calendar: GoogleCalendar):
        self.calendar = calendar

    def create_list_calendar_events_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.calendar.list_events,
            name="list_calendar_events",
            description=(
                "Look up events on the user's Google Calendar. Use when the "
                "user asks 'what's on my calendar', 'what do I have today', "
                "'do I have any meetings tomorrow', etc.\n\n"
                "The `when` argument accepts one of these named windows:\n"
                "  - 'today'      (default)\n"
                "  - 'tomorrow'\n"
                "  - 'this_week'  (today through 6 days out)\n"
                "  - 'next_week'  (the following 7 days)\n\n"
                "Returns a newline-joined list of 'TIME — TITLE (id=...)' "
                "lines, or a 'Nothing on the calendar' string if empty. "
                "Read the event titles back to the user; mention times only "
                "if they ask. Don't read the ids out loud — they're for "
                "delete_calendar_event."
            ),
        )

    def create_add_calendar_event_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.calendar.add_event,
            name="add_calendar_event",
            description=(
                "Add a reminder or event to the user's Google Calendar. Use "
                "when the user asks to be reminded ('remind me to X at Y'), "
                "or to add an event ('schedule a meeting Tuesday at 3').\n\n"
                "Arguments:\n"
                "  title (str): the event/reminder text. Keep it short.\n"
                "  start_iso (str): ISO 8601 datetime, e.g. "
                "'2026-06-10T19:00:00'. You must convert natural-language "
                "times like 'tomorrow at 7pm' or 'next Monday at 3' to ISO "
                "yourself — today's date and the user's current time are in "
                "your context. The user's local timezone is assumed if you "
                "omit a tz offset; that's fine.\n"
                "  duration_minutes (int): default 30. Use a longer duration "
                "for meetings (60+), shorter for one-off reminders (15 is "
                "fine).\n\n"
                "Returns 'Added: TITLE at WHEN (id=...)' on success. "
                "Confirm to the user with the title; you can omit the id."
            ),
        )

    def create_delete_calendar_event_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.calendar.delete_event,
            name="delete_calendar_event",
            description=(
                "Delete an event from the user's Google Calendar by id. Use "
                "when the user asks to cancel something ('cancel my 7pm "
                "reminder', 'remove the dentist appointment').\n\n"
                "You MUST pass a real event_id that came from a prior "
                "list_calendar_events or add_calendar_event call. Never "
                "invent an id. If the user references an event by name, "
                "call list_calendar_events first to find its id, then "
                "delete it.\n\n"
                "Returns 'Deleted.' on success."
            ),
        )
