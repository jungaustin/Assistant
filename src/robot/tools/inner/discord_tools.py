"""Discord tools: send, catch up, mark read, list channels.

Catch-up model (the design the user picked): Nemo's cursor is a *floor*,
not the truth. It advances only when Nemo summarizes, so it can re-cover
ground the user already read in the app but can never skip unseen
messages. Three escape hatches keep the staleness harmless:
  - `since` overrides the cursor ("catch me up since 3pm")
  - mark_discord_read snaps the cursor to now ("I just read it myself")
  - first contact with a channel asks for a time instead of guessing

catch_up returns the transcript with the summarize instructions prepended
to it (CATCHUP_INSTRUCTIONS), not in the tool description — output-handling
guidance lands harder riding alongside the output than buried up-context at
call time. The summarizing brain is already in the loop, so the tool needs
no LLM call of its own.

Like WebTools, everything degrades gracefully: no DISCORD_BOT_TOKEN just
means the tools answer "not set up yet" instead of breaking boot.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from langchain_core.tools import BaseTool, StructuredTool

from robot.config import DISCORD_CURSOR_PATH
from robot.tools.inner.discord_client import (
    CursorStore,
    DiscordClient,
    datetime_from_snowflake,
    snowflake_from_datetime,
)

# Channel assumed when the user asks about Discord without naming one. The
# tool descriptions also tell the LLM to pass this, but the local brain
# sometimes omits the channel arg entirely — so we enforce it here too, where
# it can't be skipped. Set to "" to go back to asking when there are several
# channels. Must match the channel's real name (case-insensitive, no #).
_DEFAULT_CHANNEL = "gameing"

_MAX_MESSAGES = 300
# Keep the transcript well under the local brain's context window — the
# system prompt and 20+ tool schemas already claim a few thousand tokens.
_MAX_TRANSCRIPT_CHARS = 6000

_NOT_CONFIGURED = (
    "Discord isn't set up yet — DISCORD_BOT_TOKEN is missing from .env."
)

# Prepended to the transcript catch_up_discord hands back, so the brain reads
# these when it's actually composing the summary (not when deciding to call
# the tool). Tuned for tight, reply-first prose with concrete names + details.
CATCHUP_INSTRUCTIONS = (
    "Below is a raw Discord transcript. Catch the user up on it.\n\n"
    "Open with what needs them: @mentions, direct questions, or anything "
    "clearly waiting on a reply — name who's waiting and on what. If nothing "
    "is aimed at them, say so in one line and move on.\n\n"
    "Then walk through what was discussed as tight, plain-prose paragraphs — "
    "no bullets, no headers. Be thorough: cover each distinct thread and name "
    "who said what, with the actual specifics ('Custardbagel was venting "
    "about Spotify cancelling his plan and asking if anyone's switched to "
    "Apple Music,' not 'discussed customer support'). Keep it dense — every "
    "sentence should carry a real detail, no vibe-filler like 'an engaging "
    "exchange' and no restating that people were chatting.\n\n"
    "Don't recap message by message, and don't end with an offer to help."
)

_RELATIVE_RE = re.compile(
    r"(?:last\s+|past\s+)?(\d+(?:\.\d+)?)\s*"
    r"(minutes?|mins?|m|hours?|hrs?|h|days?|d)\b(?:\s+ago)?",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", re.IGNORECASE)


def _parse_since(text: str) -> datetime | None:
    """Turn a spoken time reference into a datetime (local).

    Accepts: "2 hours ago" / "last 30 minutes", clock times ("3pm",
    "3:30pm", "15:00" — today, or yesterday if that'd be in the future),
    "yesterday" / "today" / "this morning" (midnight), and ISO strings.
    Returns None if nothing matches — the tool asks for a clearer time.
    """
    text = text.strip().lower()
    if not text:
        return None
    now = datetime.now()

    if text == "yesterday":
        return (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if text in ("today", "this morning"):
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    m = _RELATIVE_RE.fullmatch(text)
    if m:
        amount = float(m.group(1))
        unit = m.group(2)[0]  # m / h / d
        minutes = {"m": 1, "h": 60, "d": 1440}[unit] * amount
        return now - timedelta(minutes=minutes)

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    m = _CLOCK_RE.fullmatch(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = (m.group(3) or "").lower()
        if hour > 23 or minute > 59:
            return None
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # "Since 9pm" said at 8am means yesterday evening, not tonight.
        if candidate > now:
            candidate -= timedelta(days=1)
        return candidate

    return None


def _format_message(message: dict) -> str | None:
    author = message.get("author") or {}
    name = author.get("global_name") or author.get("username") or "someone"
    text = (message.get("content") or "").strip()
    if not text:
        if message.get("attachments"):
            text = "[attachment]"
        elif message.get("embeds"):
            text = "[link/embed]"
        else:
            return None
    timestamp = datetime_from_snowflake(message["id"]).astimezone()
    return f"[{timestamp:%a %I:%M%p}] {name}: {text}"


class DiscordTools:
    def __init__(
        self,
        client: DiscordClient | None = None,
        cursors: CursorStore | None = None,
    ):
        self._client = client or DiscordClient()
        self._cursors = cursors or CursorStore(DISCORD_CURSOR_PATH)

    # -- channel resolution -------------------------------------------------

    def _resolve_channel(self, name: str) -> tuple[dict | None, str | None]:
        """Returns (channel, None) or (None, voice-ready error string)."""
        try:
            channels = self._client.list_text_channels()
        except Exception as e:
            return None, f"Couldn't reach Discord: {e}"
        if not channels:
            return None, (
                "The bot isn't in any servers yet — invite it to one first."
            )
        if not name:
            # No channel named: use the default if it's visible, then fall
            # back to the only channel if there's just one, else ask.
            if _DEFAULT_CHANNEL:
                default = _DEFAULT_CHANNEL.lstrip("#").strip().lower()
                for c in channels:
                    if c["name"].lower() == default:
                        return c, None
            if len(channels) == 1:
                return channels[0], None
            options = ", ".join(f"#{c['name']}" for c in channels)
            return None, f"Which channel? I can see: {options}."
        wanted = name.lstrip("#").strip().lower()
        matches = [c for c in channels if c["name"].lower() == wanted]
        if not matches:
            options = ", ".join(f"#{c['name']}" for c in channels)
            return None, f"No channel called '#{wanted}'. I can see: {options}."
        if len(matches) > 1:
            return None, (
                f"'#{wanted}' exists in more than one server — I can only "
                "handle unique channel names for now."
            )
        return matches[0], None

    # -- tool bodies ----------------------------------------------------------

    def send_discord_message(self, channel: str, message: str) -> str:
        if not self._client.is_configured:
            return _NOT_CONFIGURED
        target, error = self._resolve_channel(channel)
        if error:
            return error
        try:
            self._client.send_message(target["id"], message)
        except Exception as e:
            return f"Couldn't send the Discord message: {e}"
        return f"Sent to #{target['name']}."

    def catch_up_discord(self, channel: str = "", since: str = "") -> str:
        if not self._client.is_configured:
            return _NOT_CONFIGURED
        target, error = self._resolve_channel(channel)
        if error:
            return error

        if since:
            start = _parse_since(since)
            if start is None:
                return (
                    f"I couldn't understand the time '{since}' — try "
                    "something like '2 hours ago', '3pm', or 'yesterday'."
                )
            after = snowflake_from_datetime(start)
            since_phrase = since
        else:
            stored = self._cursors.get(target["id"])
            if stored is None:
                return (
                    f"I haven't read #{target['name']} before, so I don't "
                    "know where you left off — give me a starting time, "
                    "like 'since 3pm' or 'the last 2 hours'."
                )
            after = stored
            since_phrase = "your last catch-up"

        try:
            messages = self._client.fetch_messages_after(
                target["id"], after, max_messages=_MAX_MESSAGES
            )
        except Exception as e:
            return f"Couldn't read Discord: {e}"

        if not messages:
            return f"Nothing new in #{target['name']} since {since_phrase}."

        # Advance the cursor first — even if formatting drops every line
        # (all-attachment spam), these messages now count as covered.
        self._cursors.set(target["id"], messages[-1]["id"])

        lines = [l for l in (_format_message(m) for m in messages) if l]
        total = len(lines)
        truncated = 0
        while lines and sum(len(l) + 1 for l in lines) > _MAX_TRANSCRIPT_CHARS:
            lines.pop(0)  # drop oldest; the recent end matters most
            truncated += 1

        header = (
            f"{len(messages)} new messages in #{target['name']} "
            f"since {since_phrase}"
        )
        if truncated:
            header += f" (showing the most recent {total - truncated})"
        transcript = header + ":\n" + "\n".join(lines)
        return f"{CATCHUP_INSTRUCTIONS}\n\n---\n\n{transcript}"

    def mark_discord_read(self, channel: str = "") -> str:
        if not self._client.is_configured:
            return _NOT_CONFIGURED
        if channel:
            target, error = self._resolve_channel(channel)
            if error:
                return error
            targets = [target]
        else:
            try:
                targets = self._client.list_text_channels()
            except Exception as e:
                return f"Couldn't reach Discord: {e}"
            if not targets:
                return (
                    "The bot isn't in any servers yet — invite it to one "
                    "first."
                )
        now_marker = snowflake_from_datetime(datetime.now())
        for t in targets:
            self._cursors.set(t["id"], now_marker)
        if len(targets) == 1:
            return f"Marked #{targets[0]['name']} as read."
        return f"Marked all {len(targets)} channels as read."

    def list_discord_channels(self) -> str:
        if not self._client.is_configured:
            return _NOT_CONFIGURED
        try:
            channels = self._client.list_text_channels()
        except Exception as e:
            return f"Couldn't reach Discord: {e}"
        if not channels:
            return "The bot isn't in any servers yet — invite it to one first."
        by_guild: dict[str, list[str]] = {}
        for c in channels:
            by_guild.setdefault(c["guild"], []).append(f"#{c['name']}")
        return "\n".join(
            f"{guild}: {', '.join(names)}" for guild, names in by_guild.items()
        )

    # -- tool factories -------------------------------------------------------

    def create_send_discord_message_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.send_discord_message,
            name="send_discord_message",
            description=(
                "Send a message to a Discord channel.\n\n"
                "  channel (str): channel name, like 'general' (# optional). "
                "Default to 'gameing' when the user doesn't name a channel; "
                "only use a different one if they say so.\n"
                "  message (str): the text to send — use the user's wording.\n\n"
                "Returns a confirmation or a plain-English error."
            ),
        )

    def create_catch_up_discord_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.catch_up_discord,
            name="catch_up_discord",
            description=(
                "Catch the user up on Discord messages they haven't seen.\n\n"
                "Args:\n"
                "  channel (str, optional): channel name. Default to "
                "'gameing' when the user doesn't name a channel; only use a "
                "different one if they say so.\n"
                "  since (str, optional): starting time like '2 hours ago', "
                "'3pm', or 'yesterday'. ONLY pass this when the user names "
                "a time — otherwise leave it empty and the marker from the "
                "last catch-up is used."
            ),
        )

    def create_mark_discord_read_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.mark_discord_read,
            name="mark_discord_read",
            description=(
                "Move the Discord 'last read' marker to now WITHOUT "
                "summarizing anything. Use when the user says they've "
                "already read Discord themselves, e.g. 'mark Discord as "
                "read' — the next catch-up then starts from this moment.\n\n"
                "  channel (str, optional): one channel name; omit to mark "
                "every channel as read."
            ),
        )

    def create_list_discord_channels_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.list_discord_channels,
            name="list_discord_channels",
            description=(
                "List the Discord channels the bot can see, grouped by "
                "server. Use when the user asks what channels there are or "
                "when another Discord tool says it can't find a channel."
            ),
        )
