"""A destructive call must never target an id the model only guessed.

2026-09-01: "remove the entries for today and relog 1,620 for Panda Express"
produced query_entries, delete_entry(entry_id=1) and log_entry in ONE batch.
ToolNode runs a batch concurrently, so the delete fired 1ms after the query and
long before it could answer — entry #1, a steak logged back in June, was
destroyed while today's rows survived. The robot then told the user it had
removed today's entries.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from robot.brain.agent import (
    _drop_unverified_destructive_calls,
    _ids_the_model_has_seen,
)


def _ai(tool_calls):
    return AIMessage(content="", tool_calls=tool_calls)


def _call(name, **args):
    return {"name": name, "args": args, "id": f"c-{name}-{len(args)}"}


def test_delete_in_same_batch_as_the_query_is_dropped():
    # The exact production shape.
    resp = _ai([
        _call("query_entries", type="calories", start_date="2026-08-31"),
        _call("delete_entry", entry_id=1),
        _call("log_entry", type="calories", value=1620, note="Panda Express"),
    ])
    out = _drop_unverified_destructive_calls(
        resp, [HumanMessage(content="remove the entries for today and relog 1,620")]
    )
    names = [c["name"] for c in out.tool_calls]
    assert "delete_entry" not in names, "guessed delete must not run"
    assert names == ["query_entries", "log_entry"], names


def test_delete_is_allowed_once_the_id_came_back_from_a_tool():
    history = [
        HumanMessage(content="remove today's entries"),
        _ai([_call("query_entries", type="calories")]),
        ToolMessage(content="#328 | 2026-08-31 | calories | 490.0 | Orange Chicken",
                    name="query_entries", tool_call_id="q"),
    ]
    resp = _ai([_call("delete_entry", entry_id=328)])
    out = _drop_unverified_destructive_calls(resp, history)
    assert [c["name"] for c in out.tool_calls] == ["delete_entry"]


def test_user_may_name_an_id_out_loud():
    resp = _ai([_call("delete_entry", entry_id=328)])
    out = _drop_unverified_destructive_calls(
        resp, [HumanMessage(content="delete entry 328 please")]
    )
    assert [c["name"] for c in out.tool_calls] == ["delete_entry"]


def test_update_entry_is_guarded_too():
    resp = _ai([_call("update_entry", entry_id=999, value=200)])
    out = _drop_unverified_destructive_calls(
        resp, [HumanMessage(content="actually make that 200")]
    )
    assert out.tool_calls == []


def test_a_different_id_than_the_one_seen_is_still_dropped():
    history = [
        HumanMessage(content="remove today's entries"),
        ToolMessage(content="#328 | calories | 490.0", name="query_entries",
                    tool_call_id="q"),
    ]
    resp = _ai([_call("delete_entry", entry_id=1)])
    assert _drop_unverified_destructive_calls(resp, history).tool_calls == []


def test_non_destructive_calls_are_never_touched():
    resp = _ai([_call("query_entries", type="calories"),
                _call("log_entry", type="calories", value=1620)])
    out = _drop_unverified_destructive_calls(resp, [HumanMessage(content="log it")])
    assert len(out.tool_calls) == 2


def test_ids_from_any_tool_result_in_the_window_are_trusted():
    """Deliberately NOT scoped to the current turn.

    Scoping it broke the natural follow-up: "delete that one too" refers to an
    id returned a turn earlier, and rejecting it made the robot refuse a
    perfectly clear request. This guard exists to stop INVENTED ids — an id
    can only appear in a tool result by being real, and a stale one deletes
    nothing at worst.
    """
    history = [
        HumanMessage(content="remove today's entries"),
        ToolMessage(content="#328 | calories", name="query_entries", tool_call_id="a"),
        AIMessage(content="Removed it."),
        HumanMessage(content="now delete the other one"),
    ]
    assert "328" in _ids_the_model_has_seen(history)
    # An id that never came back from anything is still refused.
    assert "999" not in _ids_the_model_has_seen(history)


# --- claiming a deletion that never happened -------------------------------

from robot.brain.agent import _fabrication_reason  # noqa: E402


def test_claiming_a_deletion_that_was_blocked_is_caught():
    # log_entry really ran, so the turn is not tool-less — but nothing was
    # deleted, and the reply says otherwise.
    history = [
        HumanMessage(content="remove today's entries and relog 1,620"),
        ToolMessage(content="Logged #5: calories = 1620.0", name="log_entry",
                    tool_call_id="l"),
    ]
    reason = _fabrication_reason(
        AIMessage(content="Deleted today's entries and logged 1,620 calories."),
        history,
    )
    assert reason == "claimed a deletion that no tool performed"


def test_deletion_claim_is_fine_when_delete_entry_really_ran():
    history = [
        HumanMessage(content="scratch that"),
        ToolMessage(content="Deleted entry #5: calories = 1620.0", name="delete_entry",
                    tool_call_id="d"),
    ]
    assert _fabrication_reason(AIMessage(content="Removed it."), history) is None


def test_ordinary_replies_are_not_mistaken_for_deletion_claims():
    h = [HumanMessage(content="hi")]
    for text in ("Playing.", "1,290 total for today.", "Logged 600 for rice.",
                 "I can't check the weather."):
        r = _fabrication_reason(AIMessage(content=text), h)
        assert r != "claimed a deletion that no tool performed", (text, r)


def test_a_refused_deletion_does_not_claim_nothing_was_saved():
    """The generic fallback is wrong here: the log usually DID succeed."""
    from robot.brain.agent import _fallback_for, _FABRICATION_FALLBACK

    msg = _fallback_for("claimed a deletion that no tool performed")
    assert msg != _FABRICATION_FALLBACK
    assert "deleted" in msg.lower()
    assert "saved" not in msg.lower(), msg
    # unknown reasons still get the general message
    assert _fallback_for("narrated a tool call in prose") == _FABRICATION_FALLBACK


def test_an_id_from_an_earlier_turn_is_still_usable():
    """"Delete that one too" must work — #5 was returned last turn."""
    history = [
        HumanMessage(content="remove today's and relog 1,620"),
        ToolMessage(content="Logged #5: calories = 1620.0", name="log_entry",
                    tool_call_id="l"),
        AIMessage(content="Logged 1,620 for Panda Express."),
        HumanMessage(content="Actually delete the Panda Express one too."),
    ]
    resp = _ai([_call("delete_entry", entry_id=5)])
    out = _drop_unverified_destructive_calls(resp, history)
    assert [c["name"] for c in out.tool_calls] == ["delete_entry"]


def test_dropping_the_only_call_still_says_something():
    resp = _ai([_call("delete_entry", entry_id=1)])
    out = _drop_unverified_destructive_calls(
        resp, [HumanMessage(content="delete that")]
    )
    assert out.tool_calls == []
    assert "didn't delete anything" in out.content
