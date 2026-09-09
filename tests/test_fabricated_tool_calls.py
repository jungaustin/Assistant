"""The robot must never claim work that no tool actually did.

Observed on qwen2.5:14b (2026-08-25): four meals "logged" over a voice
session, a day's totals read back as fact, and sqlite untouched — the model
wrote "[calling log_entry]" as prose and invented the rest. Nothing in the
transcript distinguished it from a real turn. These cover the guard that
turns that silent data loss into a visible failure.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from robot.brain.agent import (
    _CLAIMS_WRITE_RE,
    _content_to_text,
    _FABRICATION_FALLBACK,
    _NARRATED_CALL_RE,
    _fabrication_reason,
    _tools_run_this_turn,
)


def _ai(text: str, tool_calls=None) -> AIMessage:
    return AIMessage(content=text, tool_calls=tool_calls or [])


def _tool(name: str) -> ToolMessage:
    return ToolMessage(content="ok", name=name, tool_call_id=f"{name}-1")


def test_real_tool_call_is_never_flagged():
    resp = _ai("", [{"name": "log_entry", "args": {}, "id": "1"}])
    assert _fabrication_reason(resp, [HumanMessage(content="log 600 rice")]) is None


def test_narrated_bracket_call_is_caught():
    # The exact shape seen in production.
    resp = _ai('[calling query_entries]\nTotaling 1200 calories for today.')
    assert _fabrication_reason(resp, [HumanMessage(content="total?")]) is not None


def test_narrated_call_variants_are_caught():
    for text in (
        "[calls log_entry(type='calories', value=160)] Logged 160 for Takis.",
        "[call query_entries(type=\"calories\")] 1,200.",
        "[invoking log_meal] Done.",
        "[using entry_stats] About 1,950 a day.",
    ):
        assert _NARRATED_CALL_RE.search(text), text


def test_claimed_write_without_a_write_tool_is_caught():
    resp = _ai("Logged 600 for rice.")
    reason = _fabrication_reason(resp, [HumanMessage(content="log 600 for rice")])
    assert reason == "claimed a write that no tool performed"


def test_claimed_write_is_fine_when_the_tool_really_ran():
    history = [
        HumanMessage(content="log 600 for rice"),
        _ai("", [{"name": "log_entry", "args": {}, "id": "1"}]),
        _tool("log_entry"),
    ]
    assert _fabrication_reason(_ai("Logged 600 for rice."), history) is None


def test_prose_without_a_number_is_not_flagged():
    # "saved you a step" must not trip the write claim.
    assert _fabrication_reason(_ai("Saved you a step."), [HumanMessage(content="hi")]) is None
    assert not _CLAIMS_WRITE_RE.search("Moved it to yesterday.")


def test_empty_reply_is_not_flagged():
    assert _fabrication_reason(_ai(""), [HumanMessage(content="hi")]) is None


def test_tools_run_this_turn_stops_at_the_previous_user_message():
    history = [
        HumanMessage(content="log 600 rice"),
        _ai("", [{"name": "log_entry", "args": {}, "id": "1"}]),
        _tool("log_entry"),
        _ai("Logged 600 for rice."),
        HumanMessage(content="what's my total?"),  # new turn starts here
    ]
    # log_entry belongs to the PREVIOUS turn — it must not excuse a fresh claim.
    assert _tools_run_this_turn(history) == set()


class _StubLLM:
    """Stands in for the tool-bound model; returns a scripted reply per call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def _run_assistant(agent, user_text: str):
    state = {"messages": [HumanMessage(content=user_text)]}
    return agent.build_graph().nodes["assistant"].invoke(state)


def test_second_fabrication_is_answered_honestly_not_spoken_as_success():
    from robot.brain.agent import Agent

    agent = Agent()
    agent.llm = _StubLLM(_ai("Logged 600 for rice."), _ai("Logged 600 for rice."))
    result = _run_assistant(agent, "log 600 calories for rice")

    spoken = result["messages"][-1]
    assert spoken.content == _FABRICATION_FALLBACK
    assert not getattr(spoken, "tool_calls", None)
    assert agent.llm.calls == 2, "one corrective retry before giving up"


def test_corrective_retry_recovers_a_real_tool_call():
    from robot.brain.agent import Agent

    agent = Agent()
    real = _ai("", [{"name": "log_entry", "args": {"value": 600}, "id": "1"}])
    agent.llm = _StubLLM(_ai("Logged 600 for rice."), real)
    result = _run_assistant(agent, "log 600 calories for rice")

    spoken = result["messages"][-1]
    assert [c["name"] for c in spoken.tool_calls] == ["log_entry"]
    assert spoken.content != _FABRICATION_FALLBACK


def test_a_clean_reply_costs_no_extra_llm_call():
    from robot.brain.agent import Agent

    agent = Agent()
    agent.llm = _StubLLM(_ai("1.5 mm on average."))
    _run_assistant(agent, "how big is an ant?")
    assert agent.llm.calls == 1


# ---------------------------------------------------------------------------
# The streaming path — what the Edge actually consumes.
#
# The first version of this guard validated the assistant node's return value
# but the Edge streamed raw model tokens straight to TTS, so a fabricated
# answer was fully spoken while the log read "refusing to speak it" — and the
# corrective retry was spoken after it. Anything asserting the guard works
# MUST go through Agent.stream().
# ---------------------------------------------------------------------------


class _FakeChat(BaseChatModel):
    """A real BaseChatModel, so LangChain callbacks fire exactly as they do in
    production. A plain stub object cannot reproduce the bug: stream_mode
    "messages" emits the model's message through the callback system the
    moment invoke() returns — before the assistant node can judge it — and a
    non-model stub never triggers that path, so the test passes against broken
    code. It mirrors OpenAICompatChat: non-streaming _generate, no _stream.
    """

    responses: list = []

    def __init__(self, *responses, **kw):
        super().__init__(responses=list(responses), **kw)
        self._calls = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self._calls < len(self.responses):
            msg = self.responses[self._calls]
        else:
            # Terminate the graph once the script runs out. Repeating a
            # tool-calling response forever loops assistant->tools->assistant.
            msg = AIMessage(content="Done.")
        self._calls += 1
        # A fresh message per call, like a real model. Returning the SAME
        # object twice lets LangGraph dedupe it by id, which silently hid the
        # streaming bug from three of these tests.
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=msg.content, tool_calls=list(msg.tool_calls or [])
                    )
                )
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def calls(self) -> int:
        return self._calls


def _agent_with(*responses):
    from robot.brain.agent import Agent

    agent = Agent()
    agent.llm = _FakeChat(*responses)
    return agent


def _spoken(agent, text="can you tell me what I ate yesterday?") -> str:
    with patch.object(agent, "append_episode"):
        return "".join(agent.stream(text))


def test_fabricated_answer_is_never_spoken():
    fabricated = _ai("[calling query_entries]\nYou ate 480 calories yesterday.")
    agent = _agent_with(fabricated, fabricated)
    spoken = _spoken(agent)

    assert spoken == _FABRICATION_FALLBACK
    assert "480" not in spoken
    assert "calling" not in spoken


def test_retry_replaces_the_first_answer_instead_of_appending():
    # The observed bug spoke BOTH attempts back to back.
    first = _ai("[calling query_entries]\nYou ate 480 calories yesterday.")
    second = _ai("[calling query_entries]\nYou ate 480 calories. That totals 480.")
    agent = _agent_with(first, second)
    spoken = _spoken(agent)

    assert spoken.count("480") == 0
    assert spoken == _FABRICATION_FALLBACK


def test_a_recovered_retry_speaks_only_the_good_answer():
    bad = _ai("Logged 600 for rice.")
    good = _ai("I don't have that in the log yet.")
    agent = _agent_with(bad, good)
    with patch.object(agent, "append_episode"):
        spoken = "".join(agent.stream("log 600 calories for rice"))

    assert spoken == "I don't have that in the log yet."
    assert "Logged 600" not in spoken, "the fabricated first answer leaked"
    assert _FABRICATION_FALLBACK not in spoken


def test_a_clean_answer_streams_through_untouched():
    agent = _agent_with(_ai("1.5 mm on average."))
    assert _spoken(agent, "how big is an ant?") == "1.5 mm on average."


def test_nothing_is_emitted_before_the_verdict():
    """The generator must not yield the fabricated text even partially."""
    fabricated = _ai("[calling query_entries]\nYou ate 480 calories yesterday.")
    agent = _agent_with(fabricated, fabricated)
    with patch.object(agent, "append_episode"):
        chunks = list(agent.stream("what did I eat yesterday?"))
    assert all("480" not in c for c in chunks), chunks


def test_a_caught_fabrication_does_not_poison_the_thread():
    """The lie must not survive into history.

    A fabrication left in the thread conditions every later turn to fabricate
    too — measured 0/6 real tool calls with one fabricated exchange in
    history, versus 3/6 from a clean thread. Since daily_thread_id keeps one
    thread per day, a single unguarded miss used to wreck the rest of the day.
    """
    fabricated = _ai("[calling query_entries]\nYou ate 480 calories yesterday.")
    agent = _agent_with(fabricated, fabricated)
    with patch.object(agent, "append_episode"):
        list(agent.stream("what did I eat yesterday?"))

    history = agent.graph.get_state(agent.config).values["messages"]
    stored = [_content_to_text(m.content) for m in history]
    assert not any("480" in s for s in stored), stored
    assert not any("calling" in s for s in stored), stored
    assert _FABRICATION_FALLBACK in stored


def test_query_readback_saying_logged_is_not_blocked():
    """A real query result read back aloud is not a fabrication.

    query_entries returns rows; the natural way to say them is "you logged
    1050 for rice...". Keying on the word "logged" and demanding a *write*
    tool blocked that legitimate answer end-to-end on 2026-08-27.
    """
    history = [
        HumanMessage(content="What's my total for today?"),
        _ai("", [{"name": "query_entries", "args": {}, "id": "q1"}]),
        ToolMessage(content="#1 | calories | 1050 | bacon fried rice",
                    name="query_entries", tool_call_id="q1"),
    ]
    for text in ("You logged 1050 for bacon fried rice and 240 for tofu. 1290 total.",
                 "You've logged 1290 calories today."):
        assert _fabrication_reason(_ai(text), history) is None, text


def test_write_claim_with_no_tool_at_all_is_still_caught():
    assert _fabrication_reason(
        _ai("Logged 600 for rice."), [HumanMessage(content="log 600 rice")]
    ) == "claimed a write that no tool performed"
