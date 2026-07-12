"""Tests for per-call history trimming (_trim_history).

The agent sends only the last MAX_HISTORY_MESSAGES to the LLM so latency
doesn't grow as the day's thread accumulates. The trim must never start
mid tool-exchange — OpenAI rejects a tool result whose tool_call was
trimmed away — and must never drop the current user turn entirely.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from robot.brain.agent import _trim_history


def _ai_tool(tid: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "query_entries", "args": {}, "id": tid, "type": "tool_call"}],
    )


def _long_thread() -> list:
    """Six turns, some with tool calls, ending on a tool result."""
    msgs: list = []
    for i in range(5):
        msgs.append(HumanMessage(f"turn {i}", id=f"h{i}"))
        msgs.append(AIMessage(f"reply {i}", id=f"a{i}"))
    # Final turn uses a tool, so the thread ends on a tool result.
    msgs.append(HumanMessage("what did I log", id="h5"))
    msgs.append(_ai_tool("t5"))
    msgs.append(ToolMessage("rows", tool_call_id="t5", id="tm5"))
    return msgs


def test_short_history_unchanged():
    msgs = [HumanMessage("hi", id="h0"), AIMessage("hello", id="a0")]
    assert _trim_history(msgs, 12) == msgs


def test_long_history_is_bounded():
    msgs = _long_thread()  # 13 messages
    out = _trim_history(msgs, 6)
    assert len(out) <= 6
    # Keeps the most recent content, including the final tool result.
    assert out[-1].id == "tm5"


def test_trim_never_starts_on_tool_message():
    # Window boundary lands right after a tool_call; start_on='human' must
    # drop the orphan so OpenAI doesn't 400.
    msgs = _long_thread()
    out = _trim_history(msgs, 4)
    assert not isinstance(out[0], ToolMessage), [type(m).__name__ for m in out]
    assert isinstance(out[0], HumanMessage)


def test_keeps_tool_call_and_result_together():
    msgs = _long_thread()
    out = _trim_history(msgs, 4)
    # Every tool result in the window has its tool_call present in the window.
    call_ids = {
        tc["id"]
        for m in out
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    }
    for m in out:
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in call_ids


def test_fallback_when_window_smaller_than_turn():
    # A single turn with more tool rounds than the window: a naive trim with
    # start_on='human' returns []. The fallback keeps the current user turn.
    msgs = [
        HumanMessage("old", id="h0"),
        AIMessage("old reply", id="a0"),
        HumanMessage("do the thing", id="h1"),
        _ai_tool("t1"),
        ToolMessage("r1", tool_call_id="t1", id="tm1"),
        _ai_tool("t2"),
        ToolMessage("r2", tool_call_id="t2", id="tm2"),
    ]
    out = _trim_history(msgs, 2)  # last 2 = [_ai_tool, Tool] → no human → []
    assert out, "fallback must not return an empty history"
    assert isinstance(out[0], HumanMessage)
    assert out[0].id == "h1"  # everything since the most recent user message


def test_empty_history_returns_empty():
    assert _trim_history([], 12) == []
