"""Background timer tools.

Timers run on daemon threading.Timer threads, fully off the conversation
loop — setting a 10-minute timer doesn't block anything, and you can keep
talking to Nemo (set more timers, play music, ask questions) the whole time.

When a timer expires it plays a short double beep (voice/beep.py) and prints
a transcript line. It does NOT speak or interrupt the conversation — the beep
is the whole notification, by design.
"""

from __future__ import annotations

import itertools
import threading
import time

from langchain_core.tools import BaseTool, StructuredTool

from robot.voice.beep import timer_done_beep


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        parts = [f"{minutes} minute{'s' if minutes != 1 else ''}"]
        if secs:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
        return " ".join(parts)
    hours, minutes = divmod(minutes, 60)
    parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts)


class TimerManager:
    """Tracks active timers. Thread-safe; expiry fires on a daemon thread."""

    def __init__(self, on_done=timer_done_beep):
        self._on_done = on_done
        self._timers: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def set_timer(self, seconds: float, label: str = "") -> str:
        if seconds <= 0:
            return "Timer duration must be positive."
        label = label.strip() or "timer"
        timer_id = next(self._ids)
        t = threading.Timer(seconds, self._fire, args=(timer_id,))
        t.daemon = True
        with self._lock:
            self._timers[timer_id] = {
                "label": label,
                "ends_at": time.monotonic() + seconds,
                "thread": t,
            }
        t.start()
        return f"Timer set: {label}, {_format_duration(seconds)}."

    def _fire(self, timer_id: int) -> None:
        with self._lock:
            entry = self._timers.pop(timer_id, None)
        if entry is None:  # cancelled in the race window
            return
        print(f"⏰ timer done: {entry['label']}")
        self._on_done()

    def list_timers(self) -> str:
        now = time.monotonic()
        with self._lock:
            entries = [
                (e["label"], e["ends_at"] - now) for e in self._timers.values()
            ]
        if not entries:
            return "No timers running."
        lines = [
            f"{label}: {_format_duration(max(remaining, 0))} left"
            for label, remaining in sorted(entries, key=lambda e: e[1])
        ]
        return "\n".join(lines)

    def cancel_timer(self, label: str = "") -> str:
        label = label.strip().lower()
        with self._lock:
            if not self._timers:
                return "No timers running."
            if label:
                matches = [
                    (tid, e)
                    for tid, e in self._timers.items()
                    if e["label"].lower() == label
                ]
                if not matches:
                    return f"No timer named '{label}'. " + self._unlocked_list()
            elif len(self._timers) == 1:
                matches = list(self._timers.items())
            else:
                return (
                    "Multiple timers running — say which one to cancel.\n"
                    + self._unlocked_list()
                )
            cancelled = []
            for tid, entry in matches:
                entry["thread"].cancel()
                del self._timers[tid]
                cancelled.append(entry["label"])
        return f"Cancelled: {', '.join(cancelled)}."

    def _unlocked_list(self) -> str:
        # Caller must hold self._lock.
        now = time.monotonic()
        return "Running: " + ", ".join(
            f"{e['label']} ({_format_duration(max(e['ends_at'] - now, 0))} left)"
            for e in self._timers.values()
        )


class TimerTools:
    def __init__(self, manager: TimerManager | None = None):
        self.manager = manager or TimerManager()

    def create_set_timer_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.manager.set_timer,
            name="set_timer",
            description=(
                "Start a countdown timer. Use when the user asks for a timer "
                "('set a timer for 10 minutes', 'remind me in 45 seconds', "
                "'tea timer, 3 minutes').\n\n"
                "  seconds (number): the duration in seconds. Convert "
                "natural language yourself: '10 minutes' → 600, 'an hour "
                "and a half' → 5400.\n"
                "  label (str): short name like 'tea' or 'laundry'. Use the "
                "user's words; leave empty if they gave none.\n\n"
                "The timer runs in the background — the user can keep "
                "talking to you while it counts down. When it finishes it "
                "plays a short beep on its own; you do NOT need to watch it "
                "or follow up. Confirm with the duration, e.g. 'Timer set, "
                "10 minutes.'"
            ),
        )

    def create_list_timers_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.manager.list_timers,
            name="list_timers",
            description=(
                "List running timers and their remaining time. Use when the "
                "user asks 'how long left on the timer', 'what timers are "
                "running', etc. Returns one 'LABEL: TIME left' line per "
                "timer, or 'No timers running.'"
            ),
        )

    def create_cancel_timer_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.manager.cancel_timer,
            name="cancel_timer",
            description=(
                "Cancel a running timer. Use when the user says 'cancel the "
                "timer', 'stop the tea timer', etc.\n\n"
                "  label (str): the timer's name. Leave empty if the user "
                "didn't name one — that works when exactly one timer is "
                "running. If several are running and no label matches, the "
                "result lists them; ask the user which one they meant."
            ),
        )
