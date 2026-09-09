"""Tests for the brain's prompt-cache prewarm.

The per-turn prefix (persona + tool schemas) is ~8.6k tokens. A local server
runs all of it through prompt processing before the first token — ~40s cold,
~0.3s once the server has it cached. prewarm() pays that cost at boot, on a
daemon thread, so the first spoken question doesn't.

What matters for the cache to actually hit is that prewarm sends the SAME
prefix a real turn sends: the tool-bound model and the same system message.
A near-miss prefix re-processes from scratch and buys nothing.
"""

from __future__ import annotations

import robot.brain.agent as agent_mod
from robot.brain.agent import Agent


class FakeLLM:
    """Stands in for the chat model; records what invoke() was handed."""

    def __init__(self):
        self.bound_tools = None
        self.calls: list[tuple[list, dict]] = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        return "ok"


def _agent(tmp_path, monkeypatch, llm):
    monkeypatch.setattr(agent_mod, "STATE_DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.setattr(agent_mod, "MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    return Agent(llm=llm)


def test_prewarm_sends_the_same_prefix_a_real_turn_sends(tmp_path, monkeypatch):
    llm = FakeLLM()
    a = _agent(tmp_path, monkeypatch, llm)
    a.prewarm()

    assert len(llm.calls) == 1
    messages, kwargs = llm.calls[0]
    # Same system message object a turn builds — this is the cached prefix.
    assert messages[0].content == a._system_message().content
    # Bound to the real tools: the schemas are part of the prefix too.
    assert llm.bound_tools is a.tools or llm.bound_tools == a.tools
    # Only prompt processing matters; the generated token is thrown away.
    assert kwargs.get("max_tokens") == 1


def test_prewarm_swallows_errors(tmp_path, monkeypatch):
    """A cold cache is a slow first question, not a broken robot."""

    class Boom(FakeLLM):
        def invoke(self, messages, **kwargs):
            raise RuntimeError("ollama not up yet")

    a = _agent(tmp_path, monkeypatch, Boom())
    a.prewarm()  # must not raise


def test_start_prewarm_runs_on_a_daemon_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BRAIN_PREWARM", True)
    llm = FakeLLM()
    a = _agent(tmp_path, monkeypatch, llm)

    thread = a.start_prewarm()
    assert thread is not None
    assert thread.daemon, "a running prewarm must never hold up shutdown"
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(llm.calls) == 1


def test_start_prewarm_is_a_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BRAIN_PREWARM", False)
    llm = FakeLLM()
    a = _agent(tmp_path, monkeypatch, llm)

    assert a.start_prewarm() is None
    assert llm.calls == []
