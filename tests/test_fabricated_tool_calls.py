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
    _scrub_history,
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


# ---------------------------------------------------------------------------
# The 2026-09-15 partial-write session.
#
# "log 600 pork belly, 100 kimchi, 250 rice, and five spicy nuggets from last
# time." One log_meal wrote the first three; lookup_food answered the nuggets
# with a 15-piece basis. The model did the 5-piece arithmetic, read back all
# four items and a total, and never wrote the nuggets. Because log_meal HAD
# run, every check here passed it. Two follow-up turns then re-read the same
# invented list with no tool calls at all, and the barbecue sauce was lost the
# same way. sqlite ended the session holding rows #393-#395 and nothing else.
# ---------------------------------------------------------------------------

_LOG_MEAL_RESULT = ToolMessage(
    content=(
        "Logged 3 items on 2026-09-15:\n"
        "  #393 | pork belly = 600\n"
        "  #394 | kimchi = 100\n"
        "  #395 | rice = 250\n"
        "  total: 950"
    ),
    name="log_meal",
    tool_call_id="m1",
)

_LOOKUP_RESULT = ToolMessage(
    content=(
        "Past logs matching 'spicy nuggets':\n"
        "  15 piece spicy mcnuggets from mcdonald's — 1 log, usually 735, "
        "last on 2026-09-14"
    ),
    name="lookup_food",
    tool_call_id="f1",
)

_PARTIAL_TURN_HISTORY = [
    HumanMessage(content="log 600 pork belly, 100 kimchi, 250 rice, and 5 spicy nuggets"),
    _ai("", [
        {"name": "log_meal", "args": {}, "id": "m1"},
        {"name": "lookup_food", "args": {}, "id": "f1"},
    ]),
    _LOG_MEAL_RESULT,
    _LOOKUP_RESULT,
]


def test_readback_of_an_item_no_write_tool_wrote_is_caught():
    """The nuggets line. log_meal ran, so only a value check can see this."""
    reply = _ai(
        "Each piece is approximately 49 calories. Since you had five pieces, "
        "that's 245 calories. Here's the log for today:\n"
        "- pork belly: 600 calories\n"
        "- kimchi: 100 calories\n"
        "- rice: 250 calories\n"
        "- spicy nuggets: 245 calories\n"
        "Total: 1200 calories\n"
        "Is this correct?"
    )
    assert _fabrication_reason(reply, _PARTIAL_TURN_HISTORY) == (
        "read back items no tool wrote"
    )


def test_honest_readback_of_exactly_what_was_written_passes():
    """The same shape, with nothing invented, must still be speakable."""
    reply = _ai(
        "Logged the items for you:\n"
        "- pork belly: 600 calories\n"
        "- kimchi: 100 calories\n"
        "- rice: 250 calories\n"
        "Total: 950 calories"
    )
    assert _fabrication_reason(reply, _PARTIAL_TURN_HISTORY[:1] + [_LOG_MEAL_RESULT]) is None


def test_fabricated_total_is_caught_even_when_every_item_was_written():
    """1200 was never written; log_meal returned 950."""
    reply = _ai(
        "Logged:\n"
        "- pork belly: 600 calories\n"
        "- kimchi: 100 calories\n"
        "- rice: 250 calories\n"
        "Total: 1200 calories"
    )
    assert _fabrication_reason(reply, _PARTIAL_TURN_HISTORY[:1] + [_LOG_MEAL_RESULT]) is not None


def test_multiline_logged_claim_with_no_tool_is_caught():
    """Turn C. 'Logged the items for you:' then a bulleted list.

    The digit lives on the NEXT line, which the old `[^.\\n]` window could not
    reach, so this was spoken as a success with sqlite untouched.
    """
    reply = _ai(
        "Logged the items for you:\n"
        "- pork belly: 600 calories\n"
        "- barbecue sauce: 90 calories\n"
        "Total: 1385 calories\n"
        "All logged for today."
    )
    assert _fabrication_reason(reply, [HumanMessage(content="log the bbq sauce too")]) == (
        "claimed a write that no tool performed"
    )


def test_future_tense_promise_with_no_tool_is_caught():
    """Turn B. "I'll log ..." is a promise, not a write."""
    reply = _ai(
        "Sure, I'll log the barbecue sauce and the spicy nuggets as well.\n"
        "Here's the log for today:\n"
        "- pork belly: 600 calories\n"
        "- barbecue sauce: 90 calories\n"
        "Total: 1385 calories"
    )
    assert _fabrication_reason(reply, [HumanMessage(content="log the bbq sauce too")]) is not None


def test_partial_write_fallback_does_not_claim_nothing_was_saved():
    """Rows #393-#395 were real. Telling the user nothing saved would be a lie."""
    from robot.brain.agent import _fallback_for

    msg = _fallback_for("read back items no tool wrote")
    assert msg != _FABRICATION_FALLBACK
    assert "nothing was saved" not in msg.lower()


def test_query_readback_in_list_form_is_not_blocked():
    """No write tool ran, so a flat list is a query readback, not a claim.

    Guards the 2026-08-27 fix: demanding database backing for every listed
    number would break reading query_entries rows aloud.
    """
    history = [
        HumanMessage(content="what did I eat today?"),
        _ai("", [{"name": "query_entries", "args": {}, "id": "q1"}]),
        ToolMessage(
            content="#1 | calories | 1050 | rice\n#2 | calories | 240 | tofu",
            name="query_entries",
            tool_call_id="q1",
        ),
    ]
    reply = _ai("Here's today:\n- rice: 1050\n- tofu: 240\nTotal: 1290")
    assert _fabrication_reason(reply, history) is None


def test_partial_write_is_not_spoken_through_the_streaming_path():
    """End-to-end on Agent.stream(), which is what the Edge consumes.

    The node-level checks above passed against the original broken build
    because the Edge streamed raw model tokens straight to TTS. A partial
    write must be caught on the path the user actually hears.
    """
    real_call = _ai("", [{
        "name": "log_meal",
        "args": {"items": [
            {"name": "pork belly", "calories": 600},
            {"name": "kimchi", "calories": 100},
            {"name": "rice", "calories": 250},
        ]},
        "id": "m1",
    }])
    # log_meal wrote three items; the reply reads back a fourth.
    invented = _ai(
        "Here's the log for today:\n"
        "- pork belly: 600 calories\n"
        "- kimchi: 100 calories\n"
        "- rice: 250 calories\n"
        "- spicy nuggets: 245 calories\n"
        "Total: 1195 calories"
    )
    agent = _agent_with(real_call, invented, invented)
    with patch.object(agent, "append_episode"):
        spoken = "".join(agent.stream("log pork belly, kimchi, rice and 5 spicy nuggets"))

    assert "245" not in spoken, "the item that was never written got spoken"
    assert "1195" not in spoken
    assert "nothing was saved" not in spoken.lower(), "three rows really were written"
    assert "didn't save" in spoken


def test_corrective_retry_logs_the_missing_item_instead_of_giving_up():
    """The outcome that actually saves the user's data.

    Catching the partial write is only half the job — the nudge should get
    the model to make the call it skipped, so the item lands in sqlite and
    the user never has to re-dictate it.
    """
    real_call = _ai("", [{
        "name": "log_meal",
        "args": {"items": [
            {"name": "pork belly", "calories": 600},
            {"name": "rice", "calories": 250},
        ]},
        "id": "m1",
    }])
    invented = _ai(
        "Here's the log:\n"
        "- pork belly: 600 calories\n"
        "- rice: 250 calories\n"
        "- spicy nuggets: 245 calories"
    )
    # The retry makes the call it should have made the first time.
    recovered = _ai("", [{
        "name": "log_entry",
        "args": {"type": "calories", "value": 245, "note": "5 piece spicy nuggets"},
        "id": "e1",
    }])
    agent = _agent_with(real_call, invented, recovered)
    with patch.object(agent, "append_episode"):
        spoken = "".join(agent.stream("log pork belly, rice and 5 spicy nuggets"))

    assert "didn't save" not in spoken, "recovered — no failure message needed"

    from robot.brain.agent import _content_to_text
    history = agent.graph.get_state(agent.config).values["messages"]
    written = [
        _content_to_text(m.content)
        for m in history
        if getattr(m, "type", None) == "tool" and getattr(m, "name", None) == "log_entry"
    ]
    assert written and "245" in written[0], f"nuggets never reached sqlite: {written}"


def test_web_lookup_does_not_excuse_a_logged_claim():
    """2026-09-21: two lookup_food_calories calls, then "Logged 290 ... Logged
    1790 ..." with no write. A web lookup cannot back a logged claim."""
    history = [
        HumanMessage(content="look up a Raising Cane's combo and log it"),
        _ai("", [{"name": "lookup_food_calories", "args": {}, "id": "w1"}]),
        ToolMessage(content="Web result: 1790 calories per serving.",
                    name="lookup_food_calories", tool_call_id="w1"),
    ]
    reply = _ai("Logged 1790 calories for Raising Cane's Kenya combo.")
    assert _fabrication_reason(reply, history) == "claimed a write that no tool performed"


def test_claimed_update_with_no_tool_is_caught():
    reply = _ai("Updated entry for Arizona's Kenya combo to 280 calories.")
    history = [HumanMessage(content="the first one was 280")]
    assert _fabrication_reason(reply, history) == "claimed a write that no tool performed"


def test_invented_row_ids_are_caught():
    """2026-09-22: query returned Sept 20 only; the model invented #419-#426."""
    history = [
        HumanMessage(content="calories for each of the past three days?"),
        _ai("", [{"name": "query_entries", "args": {}, "id": "q1"}]),
        ToolMessage(content="#414 | 2026-09-20 | calories | 140.0 | Sprite",
                    name="query_entries", tool_call_id="q1"),
    ]
    fake = _ai("2690 on September 20th.\n#419 | 2026-09-21 | calories | 160.0 | Takis")
    assert _fabrication_reason(fake, history) == "cited log rows no tool returned"
    real = _ai("#414 was the Sprite, 140.")
    assert _fabrication_reason(real, history) is None


# ---------------------------------------------------------------------------
# Keeping bad turns out of the prompt.
#
# 2026-09-22: the daily thread held fabricated "Logged 290 ..." and "Updated
# entry ... to 280" replies from before the guard knew those shapes. Replaying
# that thread against qwen2.5:14b, "Log 1050 calories for a frozen pizza" got
# a real log_entry 0/4 times; with those turns removed, 4/4. Each blocked turn
# then added a tool-less fallback reply to a log request — more of the same.
# ---------------------------------------------------------------------------


class _RecordingChat(_FakeChat):
    """_FakeChat that keeps the messages each call was sent."""

    def __init__(self, *responses, **kw):
        super().__init__(*responses, **kw)
        self._sent = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._sent.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


def _texts(messages) -> list[str]:
    return [_content_to_text(m.content) for m in messages]


def test_retry_nudge_does_not_rewrite_the_system_prompt():
    """Ollama's qwen2.5 template folds any system message into the top system
    block, so a trailing SystemMessage nudge invalidated the cached prefix and
    the retry always timed out at 10s. The retry must be the first call's
    messages plus a user-role nudge on the end."""
    from langchain_core.messages import SystemMessage

    fabricated = _ai("Logged 1050 calories for a frozen pizza.")
    chat = _RecordingChat(fabricated, fabricated)
    from robot.brain.agent import Agent

    agent = Agent()
    agent.llm = chat
    with patch.object(agent, "append_episode"):
        list(agent.stream("Log 1050 calories for a frozen pizza."))

    first, retry = chat._sent[0], chat._sent[1]
    assert retry[: len(first)] == first, "the retry changed the cached prefix"
    nudge = retry[len(first):]
    assert len(nudge) == 1
    assert isinstance(nudge[0], HumanMessage)
    assert not any(isinstance(m, SystemMessage) for m in retry[1:])


def test_a_blocked_turn_is_left_out_of_the_next_prompt():
    fabricated = _ai("Logged 1050 calories for a frozen pizza.")
    real = _ai("", [{"name": "calculate", "args": {"expression": "1+1"}, "id": "c1"}])
    chat = _RecordingChat(fabricated, fabricated, real, _ai("2."))
    from robot.brain.agent import Agent

    agent = Agent()
    agent.llm = chat
    with patch.object(agent, "append_episode"):
        first = "".join(agent.stream("Log 1050 calories for a frozen pizza."))
        list(agent.stream("what's 1 plus 1?"))
    assert first == _FABRICATION_FALLBACK

    next_prompt = _texts(chat._sent[2])
    assert "Log 1050 calories for a frozen pizza." not in next_prompt
    assert _FABRICATION_FALLBACK not in next_prompt
    assert "what's 1 plus 1?" in next_prompt

    # The checkpoint still has the whole exchange; only the prompt is scrubbed.
    stored = _texts(agent.graph.get_state(agent.config).values["messages"])
    assert "Log 1050 calories for a frozen pizza." in stored
    assert _FABRICATION_FALLBACK in stored


def _lookup_call(tid: str):
    return _ai("", [{"name": "lookup_food_calories", "args": {"item": "combo"}, "id": tid}])


def test_legacy_fabrications_already_in_the_thread_are_scrubbed():
    """The exact 2026-09-22 thread shape: saved before the guard caught it, so
    no marker — it has to be recognised by what it says."""
    from robot.brain.agent import _scrub_history

    thread = [
        HumanMessage(content="log a Raising Cane's combo"),
        _lookup_call("l1"),
        ToolMessage(content="about 1790 calories", name="lookup_food_calories",
                    tool_call_id="l1"),
        _ai("Logged 1790 calories for Raising Cane's combo."),
        HumanMessage(content="make that 1700"),
        _ai("Updated entry for Raising Cane's combo to 1700 calories."),
        HumanMessage(content="Log 1050 calories for a frozen pizza."),
    ]
    out = _scrub_history(thread)
    assert _texts(out) == ["Log 1050 calories for a frozen pizza."]


def test_old_unmarked_fallback_replies_are_scrubbed():
    from robot.brain.agent import _scrub_history

    thread = [
        HumanMessage(content="log 600 for rice"),
        _ai(_FABRICATION_FALLBACK),
        HumanMessage(content="log 600 for rice"),
    ]
    assert _texts(_scrub_history(thread)) == ["log 600 for rice"]


def test_honest_turns_and_the_current_turn_are_kept():
    from robot.brain.agent import _scrub_history

    thread = [
        HumanMessage(content="how big is an ant?"),
        _ai("1.5 mm on average."),
        HumanMessage(content="what's my total?"),
        _ai("", [{"name": "query_entries", "args": {}, "id": "q1"}]),
        ToolMessage(content="#1 | calories | 1050 | rice", name="query_entries",
                    tool_call_id="q1"),
        _ai("You logged 1050 for rice."),
        HumanMessage(content="log 600 for rice"),
        _ai("", [{"name": "log_entry", "args": {"value": 600}, "id": "e1"}]),
        ToolMessage(content="#2 logged 600", name="log_entry", tool_call_id="e1"),
    ]
    assert _scrub_history(thread) == thread


# The claim checks used to fire on anything with "added/updated/recorded/#" and
# a digit, so ordinary replies about calendars, timers and trivia were blocked
# as fake log writes, answered with "Nothing was saved", retried with a nudge
# to make a tool call, and scrubbed from the next turn's context.


def _turn(user: str, tools: list[tuple[str, str]]) -> list:
    history = [HumanMessage(content=user)]
    if tools:
        history.append(_ai("", [
            {"name": name, "args": {}, "id": f"{name}-{i}"}
            for i, (name, _) in enumerate(tools)
        ]))
        history += [
            ToolMessage(content=out, name=name, tool_call_id=f"{name}-{i}")
            for i, (name, out) in enumerate(tools)
        ]
    return history


def test_replies_outside_the_log_are_not_flagged():
    cases = [
        (_turn("add dentist tomorrow 3pm", [("add_calendar_event", "Created.")]),
         "Added the dentist to your calendar for 3 PM tomorrow."),
        (_turn("move it to 4", [("add_calendar_event", "Created.")]),
         "Updated it to 4 PM."),
        (_turn("hottest temperature ever?", []),
         "The record high is 134 degrees Fahrenheit, in Death Valley in 1913."),
        (_turn("apple earnings?", [("web_search", "Apple revenue 94.9B")]),
         "Apple recorded revenue of 94.9 billion dollars."),
        (_turn("top song?", [("web_search", "Golden tops the chart")]),
         "Golden is the #1 song right now."),
    ]
    for history, text in cases:
        assert _fabrication_reason(_ai(text), history) is None, text


def test_a_real_calendar_delete_may_say_it_deleted():
    history = _turn("cancel my dentist appointment", [
        ("list_calendar_events", "abc Dentist"),
        ("delete_calendar_event", "Deleted abc."),
    ])
    assert _fabrication_reason(_ai("Deleted it."), history) is None


def test_deleted_with_no_tool_at_all_is_still_caught():
    history = _turn("cancel my dentist appointment", [])
    assert _fabrication_reason(_ai("Deleted it."), history) is not None


def test_meal_readback_after_a_correction_is_not_flagged():
    history = _turn("rice and kimchi for lunch", [
        ("log_meal", "Logged 2 items on 2026-09-25:\n  #12 | rice = 250\n"
                     "  #13 | kimchi = 30\n  total: 280"),
    ]) + [_ai("rice: 250\nkimchi: 30\ntotal: 280")]
    history += _turn("actually the kimchi was 50", [
        ("update_entry", "Updated entry #13 (calories): value 30.0 → 50."),
    ])
    assert _fabrication_reason(_ai("Fixed.\nrice: 250\nkimchi: 50"), history) is None


def test_a_legit_calendar_turn_stays_in_context():
    history = _turn("add dentist tomorrow 3pm", [("add_calendar_event", "Created.")])
    history += [_ai("Added the dentist for 3 PM tomorrow."),
                HumanMessage(content="actually make it 4")]
    assert _scrub_history(history) == history
