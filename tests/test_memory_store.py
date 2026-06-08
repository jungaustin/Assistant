"""Tests for the MVM episodic memory store and the recall() tool.

Covers the durable episodic log (append/search/recent), the empty-turn skip,
keyword OR-matching, recency ordering, and the agent's recall tool wiring +
its never-raise contract.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from robot.brain.agent import Agent
from robot.memory import Episode, MemoryStore


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memory.db"))


def test_append_returns_id_and_search_finds_it(tmp_path: Path):
    store = _store(tmp_path)
    eid = store.append("play vito by newjeans", "Playing Vito.", "2026-06-07")
    assert eid
    hits = store.search("vito")
    assert len(hits) == 1
    assert hits[0].id == eid
    assert hits[0].user_text == "play vito by newjeans"
    assert hits[0].assistant_text == "Playing Vito."
    store.close()


def test_append_skips_empty_user_text(tmp_path: Path):
    store = _store(tmp_path)
    assert store.append("", "orphan reply", "t") == ""
    assert store.append("   ", "whitespace only", "t") == ""
    assert store.recent() == []
    store.close()


def test_search_matches_assistant_text_too(tmp_path: Path):
    store = _store(tmp_path)
    store.append("what's the weather", "It's sunny in Seattle today.", "t")
    hits = store.search("Seattle")
    assert len(hits) == 1
    store.close()


def test_search_is_recency_ordered(tmp_path: Path):
    store = _store(tmp_path)
    store.append("first thing about whales", "ok", "t")
    store.append("second thing about whales", "ok", "t")
    hits = store.search("whales")
    assert len(hits) == 2
    # Most recent first.
    assert hits[0].user_text == "second thing about whales"
    store.close()


def test_search_multiterm_is_or_matched(tmp_path: Path):
    store = _store(tmp_path)
    store.append("about my boss sarah", "noted", "t")
    store.append("about lunch plans", "noted", "t")
    # Either term matches → both rows.
    hits = store.search("sarah lunch")
    assert len(hits) == 2
    store.close()


def test_search_respects_limit(tmp_path: Path):
    store = _store(tmp_path)
    for i in range(10):
        store.append(f"note {i} topic", "ok", "t")
    assert len(store.search("topic", limit=3)) == 3
    store.close()


def test_empty_query_returns_recent(tmp_path: Path):
    store = _store(tmp_path)
    store.append("alpha", "ok", "t")
    store.append("beta", "ok", "t")
    recent = store.search("", limit=5)
    assert len(recent) == 2
    assert recent[0].user_text == "beta"
    store.close()


def test_episode_recall_line_uses_date_only():
    ep = Episode(
        id="x",
        ts="2026-06-07T12:34:56.789+00:00",
        thread_id="2026-06-07",
        user_text="hi",
        assistant_text="hello",
    )
    line = ep.as_recall_line()
    assert "2026-06-07" in line
    assert "12:34" not in line
    assert "hi" in line and "hello" in line


def test_persists_across_store_instances(tmp_path: Path):
    db = str(tmp_path / "memory.db")
    s1 = MemoryStore(db)
    s1.append("remember the rocket project", "got it", "t")
    s1.close()

    s2 = MemoryStore(db)
    hits = s2.search("rocket")
    assert len(hits) == 1
    s2.close()


# --- Agent recall tool wiring ---


class _FakeLLM:
    def bind_tools(self, tools):
        self.bound_tools = tools
        return self


def _agent(tmp_path: Path) -> Agent:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    store = MemoryStore(str(tmp_path / "memory.db"))
    return Agent(
        llm=_FakeLLM(),
        thread_id="t-recall",
        checkpointer=saver,
        memory_store=store,
    )


def test_agent_exposes_recall_tool(tmp_path: Path):
    agent = _agent(tmp_path)
    names = [t.name for t in agent.llm.bound_tools]
    assert "recall" in names


def test_agent_recall_returns_digest(tmp_path: Path):
    agent = _agent(tmp_path)
    agent.memory_store.append("I'm building a desk robot", "cool", "old-thread")
    out = agent.recall("desk robot")
    assert "desk robot" in out


def test_agent_recall_miss_is_graceful(tmp_path: Path):
    agent = _agent(tmp_path)
    out = agent.recall("nonexistent topic")
    assert "don't have anything" in out.lower()


def test_agent_recall_never_raises(tmp_path: Path):
    agent = _agent(tmp_path)

    class _Boom:
        def search(self, *_a, **_k):
            raise RuntimeError("db on fire")

    agent.memory_store = _Boom()
    out = agent.recall("anything")  # must not raise
    assert "couldn't search" in out.lower()


def test_append_episode_records_turn(tmp_path: Path):
    agent = _agent(tmp_path)
    agent.append_episode("hello there", "hi back")
    hits = agent.memory_store.search("hello")
    assert len(hits) == 1
    assert hits[0].thread_id == "t-recall"
