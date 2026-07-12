"""Live datetime + midnight thread rollover.

The system prompt's datetime block and the day-scoped thread_id used to be
captured once at Agent construction, so a robot running past local midnight
kept yesterday's date (and yesterday's conversation thread) until restart.
These tests pin the fixed behavior:

- the datetime block is rebuilt on every LLM call, not at construction
- the default (unpinned) thread rolls to today's id when the date changes
- an explicitly passed thread_id never rolls (tests / off-policy sessions)
"""

from __future__ import annotations

import sqlite3

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from robot.brain import agent as agent_mod
from robot.brain.agent import Agent, build_datetime_context
from robot.memory import MemoryStore


class FakeLLM:
    """Records every message list it's invoked with; never calls tools."""

    def __init__(self):
        self.calls: list[list] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content="ok")


def _make_agent(**kwargs) -> tuple[Agent, FakeLLM]:
    llm = FakeLLM()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    agent = Agent(
        llm=llm,
        checkpointer=saver,
        memory_store=MemoryStore(":memory:"),
        **kwargs,
    )
    return agent, llm


def _system_content(llm: FakeLLM, call_index: int) -> str:
    msgs = llm.calls[call_index]
    assert isinstance(msgs[0], SystemMessage)
    return msgs[0].content


def test_datetime_block_is_rebuilt_per_call(monkeypatch):
    agent, llm = _make_agent(thread_id="t-datetime")
    stamps = iter(["DATECTX-TURN-1", "DATECTX-TURN-2"])
    monkeypatch.setattr(agent_mod, "build_datetime_context", lambda: next(stamps))

    agent.run("hello")
    agent.run("hello again")

    assert "DATECTX-TURN-1" in _system_content(llm, 0)
    assert "DATECTX-TURN-2" in _system_content(llm, 1)


def test_system_prompt_contains_persona_and_live_date():
    agent, llm = _make_agent(thread_id="t-persona")
    agent.run("hi")
    content = _system_content(llm, 0)
    # Persona text precedes the datetime block, same as the old baked prompt.
    assert content.startswith(agent._persona)
    assert build_datetime_context().split(".")[0] in content  # "Current date: X"


def test_default_thread_rolls_when_date_changes(monkeypatch):
    monkeypatch.setattr(agent_mod, "daily_thread_id", lambda: "2026-07-11")
    agent, _ = _make_agent()  # no thread_id: day-scoped policy
    assert agent.thread_id == "2026-07-11"

    # Midnight passes under a running process.
    monkeypatch.setattr(agent_mod, "daily_thread_id", lambda: "2026-07-12")
    agent.run("first turn after midnight")

    assert agent.thread_id == "2026-07-12"
    assert agent.config["configurable"]["thread_id"] == "2026-07-12"
    # Callbacks survive the config rebuild.
    assert agent.config.get("callbacks"), "callbacks lost in thread rollover"


def test_pinned_thread_never_rolls(monkeypatch):
    agent, _ = _make_agent(thread_id="t-pinned")
    monkeypatch.setattr(agent_mod, "daily_thread_id", lambda: "2026-07-12")

    agent.run("turn")

    assert agent.thread_id == "t-pinned"
    assert agent.config["configurable"]["thread_id"] == "t-pinned"


def test_no_rollover_when_date_unchanged(monkeypatch):
    monkeypatch.setattr(agent_mod, "daily_thread_id", lambda: "2026-07-11")
    agent, _ = _make_agent()
    config_before = agent.config

    agent.run("same-day turn")

    assert agent.thread_id == "2026-07-11"
    assert agent.config is config_before  # untouched, not rebuilt
