"""Tests for the Discord tools (send / catch up / mark read / list).

All Discord API access is mocked — a MagicMock stands in for
DiscordClient, so no HTTP happens. CursorStore runs against tmp_path.

Covers:
  - every tool reports "not set up" when the token is missing
  - channel resolution: by name, case/# insensitive, unknown, ambiguous,
    omitted-with-one-channel, omitted-with-many
  - catch_up: first contact asks for a time; since overrides; cursor is
    used and advanced; nothing-new leaves the cursor alone; transcript
    formatting and the size cap
  - mark_read: one channel and all channels, cursor lands at "now"
  - cursor store: roundtrip, corrupt file tolerated
  - _parse_since: relative, clock, keyword, ISO, garbage
  - snowflake <-> datetime roundtrip
  - tool names
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from robot.tools.inner.discord_client import (
    CursorStore,
    datetime_from_snowflake,
    snowflake_from_datetime,
)
from robot.tools.inner.discord_tools import DiscordTools, _parse_since

_GENERAL = {"id": "111", "name": "general", "guild": "Friends"}
_GAMING = {"id": "222", "name": "gaming", "guild": "Friends"}


def _msg(message_id: int, author: str = "alice", content: str = "hi", **extra):
    return {
        "id": str(message_id),
        "author": {"username": author},
        "content": content,
        "attachments": [],
        "embeds": [],
        **extra,
    }


def _tools(tmp_path, channels=None, configured=True):
    client = MagicMock()
    client.is_configured = configured
    client.list_text_channels.return_value = (
        channels if channels is not None else [_GENERAL]
    )
    cursors = CursorStore(tmp_path / "cursors.json")
    return DiscordTools(client=client, cursors=cursors), client, cursors


# ---------------------------------------------------------------------------
# not configured
# ---------------------------------------------------------------------------


def test_all_tools_report_missing_token(tmp_path):
    tools, client, _ = _tools(tmp_path, configured=False)
    for result in (
        tools.send_discord_message("general", "yo"),
        tools.catch_up_discord(),
        tools.mark_discord_read(),
        tools.list_discord_channels(),
    ):
        assert "DISCORD_BOT_TOKEN" in result
    client.list_text_channels.assert_not_called()


# ---------------------------------------------------------------------------
# channel resolution / send
# ---------------------------------------------------------------------------


def test_send_resolves_name_and_sends(tmp_path):
    tools, client, _ = _tools(tmp_path)
    result = tools.send_discord_message("#General", "heading out")
    client.send_message.assert_called_once_with("111", "heading out")
    assert result == "Sent to #general."


def test_send_unknown_channel_lists_options(tmp_path):
    tools, client, _ = _tools(tmp_path, channels=[_GENERAL, _GAMING])
    result = tools.send_discord_message("memes", "hello")
    assert "No channel called '#memes'" in result
    assert "#general" in result and "#gaming" in result
    client.send_message.assert_not_called()


def test_ambiguous_channel_name_errors(tmp_path):
    dupe = {"id": "333", "name": "general", "guild": "Work"}
    tools, _, _ = _tools(tmp_path, channels=[_GENERAL, dupe])
    assert "more than one server" in tools.send_discord_message("general", "x")


def test_omitted_channel_with_single_channel_auto_selects(tmp_path):
    tools, client, _ = _tools(tmp_path)
    tools.send_discord_message("", "hello")
    client.send_message.assert_called_once_with("111", "hello")


def test_omitted_channel_with_many_asks_which(tmp_path):
    tools, _, _ = _tools(tmp_path, channels=[_GENERAL, _GAMING])
    result = tools.send_discord_message("", "hello")
    assert "Which channel?" in result


def test_omitted_channel_defaults_to_configured_channel(tmp_path):
    # Regression: "summarize the last hour of the Discord server" made the LLM
    # call the tool with no channel; among many channels it must land on the
    # default instead of asking which one.
    from robot.tools.inner import discord_tools as dt

    default = {"id": "999", "name": dt._DEFAULT_CHANNEL, "guild": "Friends"}
    tools, client, _ = _tools(
        tmp_path, channels=[_GENERAL, _GAMING, default]
    )
    tools.send_discord_message("", "hello")
    client.send_message.assert_called_once_with("999", "hello")


def test_send_api_error_is_spoken_not_raised(tmp_path):
    tools, client, _ = _tools(tmp_path)
    client.send_message.side_effect = RuntimeError("Missing Access")
    result = tools.send_discord_message("general", "x")
    assert "Couldn't send" in result and "Missing Access" in result


# ---------------------------------------------------------------------------
# catch_up_discord
# ---------------------------------------------------------------------------


def test_catch_up_first_contact_asks_for_time(tmp_path):
    tools, client, _ = _tools(tmp_path)
    result = tools.catch_up_discord("general")
    assert "starting time" in result
    client.fetch_messages_after.assert_not_called()


def test_catch_up_with_since_fetches_and_advances_cursor(tmp_path):
    tools, client, cursors = _tools(tmp_path)
    client.fetch_messages_after.return_value = [
        _msg(1001, "alice", "movie tonight?"),
        _msg(1002, "bob", "im in"),
    ]
    result = tools.catch_up_discord("general", since="2 hours ago")
    after = client.fetch_messages_after.call_args.args[1]
    assert isinstance(after, int) and after > 0
    assert "2 new messages in #general" in result
    assert "movie tonight?" in result and "alice" in result
    assert cursors.get("111") == "1002"


def test_catch_up_uses_stored_cursor(tmp_path):
    tools, client, cursors = _tools(tmp_path)
    cursors.set("111", "5000")
    client.fetch_messages_after.return_value = [_msg(5001)]
    tools.catch_up_discord("general")
    assert client.fetch_messages_after.call_args.args[1] == "5000"
    assert cursors.get("111") == "5001"


def test_catch_up_nothing_new_keeps_cursor(tmp_path):
    tools, client, cursors = _tools(tmp_path)
    cursors.set("111", "5000")
    client.fetch_messages_after.return_value = []
    result = tools.catch_up_discord("general")
    assert "Nothing new in #general" in result
    assert cursors.get("111") == "5000"


def test_catch_up_unparseable_since(tmp_path):
    tools, client, _ = _tools(tmp_path)
    result = tools.catch_up_discord("general", since="whenever vibes")
    assert "couldn't understand the time" in result
    client.fetch_messages_after.assert_not_called()


def test_catch_up_attachment_only_message(tmp_path):
    tools, client, _ = _tools(tmp_path)
    client.fetch_messages_after.return_value = [
        _msg(1001, "alice", "", attachments=[{"url": "x"}]),
    ]
    result = tools.catch_up_discord("general", since="1 hour ago")
    assert "[attachment]" in result


def test_catch_up_transcript_is_capped(tmp_path):
    tools, client, _ = _tools(tmp_path)
    client.fetch_messages_after.return_value = [
        _msg(1000 + i, "alice", f"msg {i}: " + "blah " * 40) for i in range(100)
    ]
    result = tools.catch_up_discord("general", since="1 hour ago")
    assert len(result) < 7000
    assert "showing the most recent" in result
    # The newest message survives the cap; the oldest is dropped.
    assert "msg 99" in result
    assert "msg 0:" not in result


# ---------------------------------------------------------------------------
# mark_discord_read
# ---------------------------------------------------------------------------


def test_mark_read_single_channel(tmp_path):
    tools, _, cursors = _tools(tmp_path, channels=[_GENERAL, _GAMING])
    result = tools.mark_discord_read("gaming")
    assert result == "Marked #gaming as read."
    assert cursors.get("222") is not None
    assert cursors.get("111") is None


def test_mark_read_all_channels_sets_now_cursor(tmp_path):
    tools, _, cursors = _tools(tmp_path, channels=[_GENERAL, _GAMING])
    result = tools.mark_discord_read()
    assert "all 2 channels" in result
    floor = snowflake_from_datetime(datetime.now() - timedelta(minutes=1))
    for channel_id in ("111", "222"):
        assert int(cursors.get(channel_id)) > floor


# ---------------------------------------------------------------------------
# list_discord_channels
# ---------------------------------------------------------------------------


def test_list_channels_grouped_by_guild(tmp_path):
    work = {"id": "444", "name": "standup", "guild": "Work"}
    tools, _, _ = _tools(tmp_path, channels=[_GENERAL, _GAMING, work])
    result = tools.list_discord_channels()
    assert "Friends: #general, #gaming" in result
    assert "Work: #standup" in result


# ---------------------------------------------------------------------------
# cursor store / time parsing / snowflakes
# ---------------------------------------------------------------------------


def test_cursor_store_roundtrip(tmp_path):
    store = CursorStore(tmp_path / "deep" / "cursors.json")
    assert store.get("1") is None
    store.set("1", 42)
    assert store.get("1") == "42"


def test_cursor_store_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "cursors.json"
    path.write_text("{not json")
    store = CursorStore(path)
    assert store.get("1") is None
    store.set("1", "7")
    assert store.get("1") == "7"


def test_parse_since_relative():
    now = datetime.now()
    assert abs((now - _parse_since("2 hours ago")) - timedelta(hours=2)) < timedelta(seconds=5)
    assert abs((now - _parse_since("last 30 minutes")) - timedelta(minutes=30)) < timedelta(seconds=5)


def test_parse_since_clock_never_in_future():
    for text in ("3pm", "3:30pm", "15:00", "12am"):
        parsed = _parse_since(text)
        assert parsed is not None
        assert parsed <= datetime.now()


def test_parse_since_keywords_and_iso():
    yesterday = _parse_since("yesterday")
    assert yesterday.hour == 0 and yesterday.date() == (datetime.now() - timedelta(days=1)).date()
    assert _parse_since("this morning").date() == datetime.now().date()
    assert _parse_since("2026-06-12T08:00") == datetime(2026, 6, 12, 8, 0)


def test_parse_since_garbage_is_none():
    assert _parse_since("whenever vibes") is None
    assert _parse_since("") is None


def test_snowflake_roundtrip():
    moment = datetime.now().replace(microsecond=0)
    assert datetime_from_snowflake(snowflake_from_datetime(moment)).astimezone().replace(
        tzinfo=None
    ) == moment


# ---------------------------------------------------------------------------
# tool surface
# ---------------------------------------------------------------------------


def test_tool_names(tmp_path):
    tools, _, _ = _tools(tmp_path)
    names = {
        tools.create_send_discord_message_tool().name,
        tools.create_catch_up_discord_tool().name,
        tools.create_mark_discord_read_tool().name,
        tools.create_list_discord_channels_tool().name,
    }
    assert names == {
        "send_discord_message",
        "catch_up_discord",
        "mark_discord_read",
        "list_discord_channels",
    }
