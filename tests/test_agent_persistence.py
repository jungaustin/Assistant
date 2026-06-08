"""Tests for SqliteSaver-backed Agent.

Persistence behavior is tested against a real sqlite DB in a tmp file
(faster than spinning up an actual LLM — we stub the brain). The end-to-
end "Agent persists across instances" claim is verified by the smoke
script in the commit body, not here, because that needs a real OpenAI
turn and money/network.

What these tests cover:
- daily_thread_id() returns today's local date
- delete_thread wipes checkpoints (the forget_session contract)
- Agent.forget_session calls delete_thread on the configured checkpointer
- Agent's tools list includes the forget_session tool
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from robot.brain.agent import Agent, _open_checkpoint_db
from robot.config import daily_thread_id


def test_daily_thread_id_is_local_iso_date():
    assert daily_thread_id() == date.today().isoformat()


def test_open_checkpoint_db_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "nested" / "deeper" / "conv.db"
    conn = _open_checkpoint_db(str(nested))
    try:
        assert nested.exists()
        assert nested.parent.is_dir()
        # Connection works
        conn.execute("SELECT 1").fetchone()
    finally:
        conn.close()


def test_sqlite_checkpoint_persists_across_savers(tmp_path: Path):
    """End-to-end persistence at the SqliteSaver layer (no LLM): writes
    from one SqliteSaver instance are visible from a fresh instance on
    the same DB. This is the guarantee Agent inherits."""
    db = tmp_path / "checkpoints.db"
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}

    # Saver 1: write a checkpoint.
    conn1 = sqlite3.connect(str(db), check_same_thread=False)
    saver1 = SqliteSaver(conn1)
    saver1.setup()
    saver1.put(
        config,
        {"v": 1, "ts": "2026-06-01T00:00:00", "id": "ckpt-1", "channel_values": {"x": 42}, "channel_versions": {}, "versions_seen": {}, "pending_sends": []},
        {"source": "input", "step": 0, "writes": {}, "parents": {}},
        {},
    )
    conn1.close()

    # Saver 2: open the same file, read it back.
    conn2 = sqlite3.connect(str(db), check_same_thread=False)
    saver2 = SqliteSaver(conn2)
    snapshot = saver2.get(config)
    conn2.close()

    assert snapshot is not None, "checkpoint did not survive saver close+reopen"
    assert snapshot["channel_values"]["x"] == 42


def test_delete_thread_wipes_checkpoints(tmp_path: Path):
    """forget_session relies on this primitive: delete_thread removes
    all checkpoints for a thread_id."""
    db = tmp_path / "checkpoints.db"
    config = {"configurable": {"thread_id": "victim", "checkpoint_ns": ""}}

    conn = sqlite3.connect(str(db), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    saver.put(
        config,
        {"v": 1, "ts": "2026-06-01T00:00:00", "id": "ckpt-x", "channel_values": {"x": 1}, "channel_versions": {}, "versions_seen": {}, "pending_sends": []},
        {"source": "input", "step": 0, "writes": {}, "parents": {}},
        {},
    )
    assert saver.get(config) is not None, "precondition failed: nothing to delete"

    saver.delete_thread("victim")
    assert saver.get(config) is None, "delete_thread didn't wipe the thread"
    conn.close()


def test_agent_includes_forget_session_tool():
    """The agent must expose forget_session to the LLM so the user can ask
    it to forget. Test without instantiating a real LLM by injecting a
    minimal fake that records bind_tools."""

    class FakeLLM:
        def __init__(self):
            self.bound_tools = None

        def bind_tools(self, tools):
            self.bound_tools = tools
            return self

        def invoke(self, *_args, **_kwargs):
            raise NotImplementedError

    # Use SqliteSaver with an in-memory DB (`:memory:`) so we don't touch disk.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    agent = Agent(llm=FakeLLM(), thread_id="t-tool-test", checkpointer=saver)
    tool_names = [t.name for t in agent.llm.bound_tools]
    assert "forget_session" in tool_names, (
        f"forget_session tool missing from bound tools: {tool_names}"
    )
    conn.close()


def test_agent_forget_session_calls_delete_thread():
    """End-to-end: agent.forget_session() actually deletes the configured
    thread's checkpoints via the configured checkpointer."""

    class FakeLLM:
        def bind_tools(self, _):
            return self

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    agent = Agent(
        llm=FakeLLM(), thread_id="t-forget", checkpointer=saver
    )

    # Plant a checkpoint, then forget.
    saver.put(
        {"configurable": {"thread_id": "t-forget", "checkpoint_ns": ""}},
        {"v": 1, "ts": "2026-06-01T00:00:00", "id": "ckpt-f", "channel_values": {"x": 1}, "channel_versions": {}, "versions_seen": {}, "pending_sends": []},
        {"source": "input", "step": 0, "writes": {}, "parents": {}},
        {},
    )
    assert saver.get({"configurable": {"thread_id": "t-forget", "checkpoint_ns": ""}}) is not None

    result = agent.forget_session()
    assert "forg" in result.lower() or "done" in result.lower(), (
        f"unexpected forget_session response: {result!r}"
    )
    assert saver.get({"configurable": {"thread_id": "t-forget", "checkpoint_ns": ""}}) is None
    conn.close()


def test_agent_thread_id_defaults_to_today():
    """If thread_id is not passed, agent uses daily_thread_id()."""

    class FakeLLM:
        def bind_tools(self, _):
            return self

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    try:
        agent = Agent(llm=FakeLLM(), checkpointer=saver)
        assert agent.thread_id == date.today().isoformat()
    finally:
        conn.close()
