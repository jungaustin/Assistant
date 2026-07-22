"""Tests for _dedupe_tool_calls — the guard against the model emitting the
same tool call twice in one turn (which double-writes data for log_entry).

Two layers: a unit test on the helper's dedup logic, and an end-to-end test
that drives the real LangGraph graph with a scripted LLM that duplicates a
tool call, proving only one execution reaches the ToolNode.
"""

from __future__ import annotations

import logging
import sqlite3

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from robot.brain.agent import Agent, _dedupe_tool_calls
from robot.memory import MemoryStore


def _tc(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def test_identical_tool_calls_collapse_to_one():
    msg = AIMessage(
        content="",
        tool_calls=[
            _tc("log_entry", {"type": "calories", "value": 720, "note": "rice"}, "a"),
            _tc("log_entry", {"type": "calories", "value": 720, "note": "rice"}, "b"),
        ],
    )
    out = _dedupe_tool_calls(msg)
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["id"] == "a"  # first occurrence is kept


def test_arg_order_does_not_defeat_dedup():
    # Same args, different key order in the dict — still a duplicate.
    msg = AIMessage(
        content="",
        tool_calls=[
            _tc("log_entry", {"type": "calories", "value": 720}, "a"),
            _tc("log_entry", {"value": 720, "type": "calories"}, "b"),
        ],
    )
    assert len(_dedupe_tool_calls(msg).tool_calls) == 1


def test_distinct_tool_calls_are_preserved():
    # Logging lunch and dinner together is legitimate — different args.
    msg = AIMessage(
        content="",
        tool_calls=[
            _tc("log_entry", {"type": "calories", "value": 600, "note": "lunch"}, "a"),
            _tc("log_entry", {"type": "calories", "value": 800, "note": "dinner"}, "b"),
        ],
    )
    assert len(_dedupe_tool_calls(msg).tool_calls) == 2


def test_same_args_different_tool_preserved():
    msg = AIMessage(
        content="",
        tool_calls=[
            _tc("query_entries", {"start_date": "2026-07-14"}, "a"),
            _tc("entry_stats", {"start_date": "2026-07-14"}, "b"),
        ],
    )
    assert len(_dedupe_tool_calls(msg).tool_calls) == 2


def test_single_or_no_tool_calls_pass_through():
    assert _dedupe_tool_calls(AIMessage(content="hi")).tool_calls == []
    one = AIMessage(content="", tool_calls=[_tc("recall", {"query": "x"}, "a")])
    assert len(_dedupe_tool_calls(one).tool_calls) == 1


def test_dropping_a_duplicate_logs_a_warning(caplog):
    msg = AIMessage(
        content="",
        tool_calls=[
            _tc("log_entry", {"type": "calories", "value": 720}, "a"),
            _tc("log_entry", {"type": "calories", "value": 720}, "b"),
        ],
    )
    with caplog.at_level(logging.WARNING, logger="robot.brain.agent"):
        _dedupe_tool_calls(msg)
    assert any("dropped duplicate tool call name=log_entry" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


class DuplicatingLLM:
    """Emits the SAME recall tool call twice on turn 1, answers on turn 2.

    recall is agent-owned and backed by the injected in-memory MemoryStore,
    so the ToolNode runs without touching any external service. If dedup
    fails, the tool executes twice and we'd see two 'tool done' lines.
    """

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    _tc("recall", {"query": "pizza"}, "c1"),
                    _tc("recall", {"query": "pizza"}, "c2"),
                ],
            )
        return AIMessage(content="all done")


def test_graph_executes_duplicated_tool_call_once(caplog):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    agent = Agent(
        llm=DuplicatingLLM(),
        thread_id="t-dedup",
        checkpointer=saver,
        memory_store=MemoryStore(":memory:"),
    )
    try:
        with caplog.at_level(logging.INFO, logger="robot.brain.agent"):
            agent.run("remember pizza?")
        messages = [r.getMessage() for r in caplog.records]
    finally:
        conn.close()

    done = [m for m in messages if m.startswith("tool done  name=recall")]
    assert len(done) == 1, f"recall ran {len(done)} times, expected 1: {messages}"
