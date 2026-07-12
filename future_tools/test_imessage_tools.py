"""[PARKED] Tests for the iMessage tool — see future_tools/imessage_tools.py.

Not collected by pytest here (testpaths = ["tests"]). To re-activate, move
back to tests/ together with the tool module.

osascript never actually runs — _run_applescript is replaced on the
instance, so these verify which AppleScript would be executed, not that
Messages.app works.
"""

from __future__ import annotations

from robot.tools.inner.imessage_tools import IMessageTools, _escape


def _tools(lookup_result: str = "", fail: bool = False):
    """IMessageTools with a recording fake for _run_applescript."""
    tools = IMessageTools()
    calls: list[str] = []

    def fake(script: str) -> str:
        if fail:
            raise RuntimeError("not authorized")
        calls.append(script)
        if "Contacts" in script:
            return lookup_result
        return ""

    tools._run_applescript = fake
    return tools, calls


def test_phone_number_skips_contacts_lookup():
    tools, calls = _tools()
    result = tools.send_imessage("+1 (555) 123-4567", "on my way")
    assert result == "Sent to +1 (555) 123-4567."
    assert len(calls) == 1
    assert "Messages" in calls[0]
    assert "+1 (555) 123-4567" in calls[0]
    assert "on my way" in calls[0]


def test_email_skips_contacts_lookup():
    tools, calls = _tools()
    tools.send_imessage("alex@example.com", "hi")
    assert len(calls) == 1 and "Messages" in calls[0]


def test_name_resolves_through_contacts():
    tools, calls = _tools(lookup_result="+15551234567")
    result = tools.send_imessage("Mom", "be there soon")
    assert result == "Sent to Mom."
    assert len(calls) == 2
    assert "Contacts" in calls[0] and "Mom" in calls[0]
    assert "Messages" in calls[1] and "+15551234567" in calls[1]


def test_unknown_name_returns_spoken_error():
    tools, calls = _tools(lookup_result="")
    result = tools.send_imessage("Zorbo", "hello")
    assert "couldn't find 'Zorbo'" in result
    # Only the lookup ran — nothing was sent.
    assert len(calls) == 1 and "Contacts" in calls[0]


def test_osascript_failure_is_spoken_not_raised():
    tools, _ = _tools(fail=True)
    result = tools.send_imessage("Mom", "hello")
    assert "Couldn't search Contacts" in result and "not authorized" in result


def test_quotes_are_escaped():
    tools, calls = _tools()
    tools.send_imessage("555-123-4567", 'say "hi" for me')
    assert '\\"hi\\"' in calls[0]


def test_escape_backslash_before_quote():
    assert _escape('a\\b"c') == 'a\\\\b\\"c'


def test_tool_name():
    assert IMessageTools().create_send_imessage_tool().name == "send_imessage"
