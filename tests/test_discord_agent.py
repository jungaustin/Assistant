"""Tests for the Discord agent channel (robot.discord_agent).

No Discord connection: fake message/channel objects and a fake agent.

Covers:
  - should_handle: wrong channel, other user, bot, empty, attachment-only, valid
  - split_message: short untouched, newline split, hard split, size cap
  - state file: missing/corrupt -> None, round-trip, atomic write
  - Agent(thread_prefix=...) thread id and midnight rollover
  - turn handling: reply, error reply, state advance, ordering, long split
"""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from robot.brain import agent as agent_mod
from robot.brain.agent import Agent
from robot.discord_agent import (
    EMPTY_REPLY,
    ERROR_REPLY,
    IGNORE,
    NO_IMAGES,
    NO_IMAGES_REPLY,
    RUN,
    DiscordAgent,
    read_last_handled,
    should_handle,
    split_message,
    write_last_handled,
)
from robot.memory import MemoryStore

CHANNEL = 100
OWNER = 7


class FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeChannel:
    def __init__(self, channel_id=CHANNEL):
        self.id = channel_id

    def typing(self):
        return FakeTyping()


class FakeMessage:
    def __init__(
        self,
        message_id=1,
        content="hello",
        author_id=OWNER,
        bot=False,
        channel_id=CHANNEL,
        attachments=(),
    ):
        self.id = message_id
        self.content = content
        self.author = SimpleNamespace(id=author_id, bot=bot)
        self.channel = FakeChannel(channel_id)
        self.attachments = list(attachments)
        self.replies: list[str] = []

    async def reply(self, text, mention_author=True):
        assert mention_author is False
        self.replies.append(text)


class FakeAgent:
    def __init__(self, reply="done", raises=None):
        self.reply = reply
        self.raises = raises
        self.seen: list[str] = []

    def run(self, text):
        self.seen.append(text)
        if self.raises:
            raise self.raises
        return self.reply


# --- should_handle ---------------------------------------------------------


def test_valid_owner_message_runs():
    assert should_handle(FakeMessage(), CHANNEL, OWNER) == RUN


def test_wrong_channel_ignored():
    assert should_handle(FakeMessage(channel_id=5), CHANNEL, OWNER) == IGNORE


def test_other_user_ignored():
    assert should_handle(FakeMessage(author_id=8), CHANNEL, OWNER) == IGNORE


def test_bot_author_ignored():
    assert should_handle(FakeMessage(bot=True), CHANNEL, OWNER) == IGNORE


def test_empty_text_ignored():
    assert should_handle(FakeMessage(content="  "), CHANNEL, OWNER) == IGNORE


def test_attachment_only_gets_no_images_path():
    msg = FakeMessage(content="", attachments=["pic.png"])
    assert should_handle(msg, CHANNEL, OWNER) == NO_IMAGES


def test_text_with_attachment_runs_text():
    msg = FakeMessage(content="lunch", attachments=["pic.png"])
    assert should_handle(msg, CHANNEL, OWNER) == RUN


# --- split_message ---------------------------------------------------------


def test_short_message_untouched():
    assert split_message("hi") == ["hi"]


def test_long_text_splits_on_newlines():
    text = "\n".join(["x" * 900] * 5)
    chunks = split_message(text)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    assert "\n".join(chunks) == text  # split only at line boundaries


def test_single_long_line_hard_split():
    text = "y" * 4500
    chunks = split_message(text)
    assert all(len(c) <= 2000 for c in chunks)
    assert "".join(chunks) == text


def test_mixed_long_line_never_exceeds_limit():
    text = "short\n" + "z" * 2500 + "\nend"
    chunks = split_message(text)
    assert all(len(c) <= 2000 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


# --- state file ------------------------------------------------------------


def test_state_missing_is_none(tmp_path):
    assert read_last_handled(tmp_path / "nope.json") is None


def test_state_corrupt_is_none(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert read_last_handled(p) is None


def test_state_round_trip(tmp_path):
    p = tmp_path / "sub" / "s.json"
    write_last_handled(p, 123456789012345678)
    assert read_last_handled(p) == 123456789012345678
    assert not (tmp_path / "sub" / "s.json.tmp").exists()


# --- Agent thread prefix ---------------------------------------------------


class _LLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="ok")


def _make_agent(**kwargs) -> Agent:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return Agent(
        llm=_LLM(), checkpointer=saver, memory_store=MemoryStore(":memory:"), **kwargs
    )


def test_thread_prefix_applied(monkeypatch):
    monkeypatch.setattr(agent_mod, "daily_thread_id", lambda: "2026-10-18")
    assert _make_agent(thread_prefix="discord-").thread_id == "discord-2026-10-18"


def test_no_prefix_is_unchanged(monkeypatch):
    monkeypatch.setattr(agent_mod, "daily_thread_id", lambda: "2026-10-18")
    assert _make_agent().thread_id == "2026-10-18"


def test_prefix_survives_rollover(monkeypatch):
    monkeypatch.setattr(agent_mod, "daily_thread_id", lambda: "2026-10-18")
    agent = _make_agent(thread_prefix="discord-")
    monkeypatch.setattr(agent_mod, "daily_thread_id", lambda: "2026-10-19")
    agent.run("after midnight")
    assert agent.thread_id == "discord-2026-10-19"
    assert agent.config["configurable"]["thread_id"] == "discord-2026-10-19"


def test_pinned_thread_ignores_prefix():
    agent = _make_agent(thread_id="pinned", thread_prefix="discord-")
    assert agent.thread_id == "pinned"


# --- turn handling ---------------------------------------------------------


def _bot(tmp_path, agent):
    return DiscordAgent(agent, CHANNEL, OWNER, tmp_path / "state.json")


def test_turn_replies_with_agent_output(tmp_path):
    agent = FakeAgent("Logged it.")
    bot = _bot(tmp_path, agent)
    msg = FakeMessage(42, "log rice 200")
    asyncio.run(bot.handle_message(msg))
    assert agent.seen == ["log rice 200"]
    assert msg.replies == ["Logged it."]
    assert read_last_handled(bot.state_path) == 42


def test_empty_agent_reply_says_no_response(tmp_path):
    bot = _bot(tmp_path, FakeAgent("  "))
    msg = FakeMessage(1)
    asyncio.run(bot.handle_message(msg))
    assert msg.replies == [EMPTY_REPLY]


def test_long_reply_is_split(tmp_path):
    bot = _bot(tmp_path, FakeAgent("\n".join(["l" * 900] * 5)))
    msg = FakeMessage(1)
    asyncio.run(bot.handle_message(msg))
    assert len(msg.replies) > 1
    assert all(len(r) <= 2000 for r in msg.replies)


def test_agent_error_replies_continues_and_advances(tmp_path):
    agent = FakeAgent(raises=RuntimeError("boom"))
    bot = _bot(tmp_path, agent)
    bad = FakeMessage(10)
    asyncio.run(bot.handle_message(bad))
    assert bad.replies == [ERROR_REPLY]
    assert read_last_handled(bot.state_path) == 10

    # Process is still usable for the next turn.
    agent.raises = None
    good = FakeMessage(11)
    asyncio.run(bot.handle_message(good))
    assert good.replies == ["done"]
    assert read_last_handled(bot.state_path) == 11


def test_attachment_only_replies_without_running_agent(tmp_path):
    agent = FakeAgent()
    bot = _bot(tmp_path, agent)
    msg = FakeMessage(3, content="", attachments=["a.png"])
    asyncio.run(bot.handle_message(msg))
    assert msg.replies == [NO_IMAGES_REPLY]
    assert agent.seen == []
    assert read_last_handled(bot.state_path) == 3


def test_ignored_message_does_not_advance_state(tmp_path):
    bot = _bot(tmp_path, FakeAgent())
    asyncio.run(bot.handle_message(FakeMessage(9, author_id=99)))
    assert read_last_handled(bot.state_path) is None


def test_state_never_moves_backwards(tmp_path):
    bot = _bot(tmp_path, FakeAgent())
    asyncio.run(bot.handle_message(FakeMessage(20)))
    asyncio.run(bot.handle_message(FakeMessage(15)))
    assert read_last_handled(bot.state_path) == 20


def test_concurrent_messages_run_in_arrival_order(tmp_path):
    agent = FakeAgent()
    bot = _bot(tmp_path, agent)
    msgs = [FakeMessage(i, content=f"m{i}") for i in (1, 2, 3)]

    async def go():
        await asyncio.gather(*(bot.handle_message(m) for m in msgs))

    asyncio.run(go())
    assert agent.seen == ["m1", "m2", "m3"]


def test_message_already_handled_is_not_run_again(tmp_path):
    # Reconnect: on_ready's catch_up re-fetches a message on_message already
    # handled. Running it twice would double-log a meal.
    agent = FakeAgent()
    bot = _bot(tmp_path, agent)
    first = FakeMessage(7, content="log rice 200")
    again = FakeMessage(7, content="log rice 200")

    async def go():
        await asyncio.gather(bot.handle_message(first), bot.handle_message(again))

    asyncio.run(go())
    assert agent.seen == ["log rice 200"]
    assert again.replies == []
