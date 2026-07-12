"""Tests for tool-call / LLM observability logging in the agent.

The agent wires a ToolCallLogger callback into its run config and times the
LLM round-trip in the assistant node, so a hang is visible in the logs (a
'start' line with no matching 'done' line is the stuck step). These tests
drive the real LangGraph graph with a scripted fake LLM — no network — and
assert the expected log lines fire, including through the ToolNode (which
proves the callback actually propagates via config).
"""

from __future__ import annotations

import logging
import sqlite3

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from robot.brain.agent import Agent, ToolCallLogger, _truncate
from robot.memory import MemoryStore


class ScriptedLLM:
    """Requests a `recall` tool call on the first turn, answers on the second.

    `recall` is an agent-owned tool backed by the injected in-memory
    MemoryStore, so exercising it touches no external service.
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
                    {"name": "recall", "args": {"query": "pizza"}, "id": "c1", "type": "tool_call"}
                ],
            )
        return AIMessage(content="all done")


def _agent_with_scripted_llm() -> tuple[Agent, sqlite3.Connection]:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    agent = Agent(
        llm=ScriptedLLM(),
        thread_id="t-tool-log",
        checkpointer=saver,
        memory_store=MemoryStore(":memory:"),
    )
    return agent, conn


def test_tool_call_emits_start_and_done(caplog):
    agent, conn = _agent_with_scripted_llm()
    try:
        with caplog.at_level(logging.INFO, logger="robot.brain.agent"):
            agent.run("did we ever talk about pizza")
        messages = [r.getMessage() for r in caplog.records]
    finally:
        conn.close()

    # The callback fired through ToolNode (proves config propagation).
    assert any(m.startswith("tool start name=recall") for m in messages), messages
    assert any(m.startswith("tool done  name=recall") for m in messages), messages
    # And the assistant node timed the model round-trip.
    assert any(m.startswith("llm start") for m in messages), messages
    assert any(m.startswith("llm done") for m in messages), messages


def test_tool_done_reports_elapsed(caplog):
    agent, conn = _agent_with_scripted_llm()
    try:
        with caplog.at_level(logging.INFO, logger="robot.brain.agent"):
            agent.run("remember pizza?")
        done = [r.getMessage() for r in caplog.records if r.getMessage().startswith("tool done")]
    finally:
        conn.close()

    assert done, "no 'tool done' line was logged"
    assert "elapsed=" in done[0]


def test_on_tool_error_logs_warning(caplog):
    from uuid import uuid4

    handler = ToolCallLogger()
    rid = uuid4()
    with caplog.at_level(logging.WARNING, logger="robot.brain.agent"):
        handler.on_tool_start({"name": "web_search"}, "weather", run_id=rid)
        handler.on_tool_error(TimeoutError("upstream timed out"), run_id=rid)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("tool error name=web_search" in m and "TimeoutError" in m for m in warnings), warnings


def test_truncate_caps_long_values():
    assert _truncate("x" * 50, limit=200) == "x" * 50  # short values pass through
    out = _truncate("y" * 500, limit=200)
    assert out.startswith("y" * 200)
    assert "+300 chars" in out


def test_truncate_flattens_newlines():
    assert "\n" not in _truncate("line1\nline2")
