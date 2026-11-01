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
    _ids_the_model_has_seen,
    _split_unverified_destructive_calls,
)


def _drop_unverified_destructive_calls(resp, history):
    """The calls that survive the guard, on a copy of the response."""
    kept, _ = _split_unverified_destructive_calls(resp.tool_calls, history)
    return resp.model_copy(update={"tool_calls": kept})


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


def test_update_with_no_id_is_refused():
    """2026-10-03: update_entry(entry_id=) for yesterday's burgers.

    Without an id the tool edits the NEWEST row, which was tonight's pork
    rice bowl, not the entry the user meant.
    """
    history = [
        HumanMessage(content="fix yesterday's burpers entry"),
        ToolMessage(content="Logged #478: calories = 750.0 (pork rice bowl)",
                    name="log_meal", tool_call_id="l"),
    ]
    for args in ({}, {"entry_id": ""}, {"entry_id": None}):
        resp = _ai([_call("update_entry", value=540, **args)])
        assert _drop_unverified_destructive_calls(resp, history).tool_calls == []


# --- the refused call is answered, not silently dropped ---------------------

from robot.brain.agent import (  # noqa: E402
    _refusal_message,
    _tools_run_this_turn,
)


def test_refusal_tells_the_model_how_to_get_the_id():
    msg = _refusal_message(_call("update_entry", value=540), prior_refusals=0)
    assert msg.status == "error"
    assert msg.name == "update_entry"
    assert "query_entries" in msg.content
    assert "nothing was changed" in msg.content


def test_refusal_does_not_count_as_the_tool_running():
    history = [
        HumanMessage(content="fix it"),
        _refusal_message(_call("update_entry", value=540), prior_refusals=0),
    ]
    assert "update_entry" not in _tools_run_this_turn(history)


def test_refusal_never_vouches_for_the_id_it_refused():
    msg = _refusal_message(_call("delete_entry", entry_id=1), prior_refusals=0)
    history = [HumanMessage(content="delete that"), msg]
    assert "1" not in _ids_the_model_has_seen(history)


def test_repeated_refusals_tell_the_model_to_stop_and_ask():
    msg = _refusal_message(_call("update_entry", value=540), prior_refusals=2)
    assert "Stop retrying" in msg.content


def test_promising_an_update_that_never_ran_is_caught():
    """The exact closing line spoken on 2026-10-03, with the entry unchanged."""
    from robot.brain.agent import _fabrication_reason

    history = [
        HumanMessage(content="can you double check that and fix it for me?"),
        ToolMessage(content="Past logs matching 'In-N-Out burger': ...",
                    name="lookup_food", tool_call_id="f"),
    ]
    for text in (
        "So two burgers would be 540 calories. I'll update the entry for you.",
        "I will correct the entry for yesterday.",
        "Let me fix that.",
    ):
        reply = AIMessage(content=text)
        assert _fabrication_reason(reply, history) == (
            "claimed an update that no tool performed"
        ), text


def test_asking_whether_to_update_is_not_a_claim():
    from robot.brain.agent import _fabrication_reason

    h = [HumanMessage(content="what did I eat yesterday?")]
    for text in ("Want me to fix that?", "Should I change it to 540?"):
        r = _fabrication_reason(AIMessage(content=text), h)
        assert r != "claimed an update that no tool performed", (text, r)


def test_refused_update_lets_the_model_look_up_the_id_and_retry():
    """End to end through the graph: the burger replay from 2026-10-03.

    The model first tries an id-less update, gets the refusal back as a tool
    result, queries yesterday, and updates the right row. Tonight's rows are
    untouched.
    """
    from datetime import date, timedelta
    from unittest.mock import patch

    from tests.test_fabricated_tool_calls import _agent_with

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    probe = _agent_with()
    db = probe._tm.log_tools.db
    burger = db.log_entry("calories", 270, "two burpers from in-n-out", yesterday)
    burger_id = int(burger.split("#")[1].split(":")[0])
    tonight = db.log_entry("calories", 750, "pork rice bowl")
    tonight_id = int(tonight.split("#")[1].split(":")[0])

    agent = _agent_with(
        AIMessage(content="", tool_calls=[
            {"name": "update_entry", "args": {"value": 540}, "id": "u1"},
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "query_entries",
             "args": {"type": "calories", "start_date": yesterday,
                      "end_date": yesterday}, "id": "q1"},
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "update_entry",
             "args": {"entry_id": burger_id, "value": 540,
                      "note": "two burgers from in-n-out"}, "id": "u2"},
        ]),
        AIMessage(content="Fixed: two burgers from In-N-Out, 540."),
    )
    with patch.object(agent, "append_episode"):
        spoken = "".join(agent.stream("fix yesterday's burpers, 270 each"))

    assert "540" in spoken
    rows = {r[0]: r for r in db._conn.execute(
        "SELECT id, value, note FROM entries WHERE id IN (?, ?)",
        (burger_id, tonight_id),
    )}
    assert rows[burger_id][1:] == (540.0, "two burgers from in-n-out")
    assert rows[tonight_id][1:] == (750.0, "pork rice bowl")
    db._conn.execute("DELETE FROM entries WHERE id IN (?, ?)", (burger_id, tonight_id))
    db._conn.commit()
