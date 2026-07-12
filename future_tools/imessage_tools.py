"""[PARKED — not wired into the agent] iMessage tool.

Parked 2026-06-12 at the user's request: code kept for later, but the tool
is NOT registered and NOT tested right now. It lives outside src/ so it
isn't imported by the package or compiled by the `just lint` recipe.

To re-activate later:
  1. Move this file back to src/robot/tools/inner/imessage_tools.py
  2. In src/robot/tools/manager.py, re-add:
       from robot.tools.inner.imessage_tools import IMessageTools
       self.imessage_tools = IMessageTools()          # in __init__
       all_tools.append(self.imessage_tools.create_send_imessage_tool())
  3. Restore tests from future_tools/test_imessage_tools.py into tests/

------------------------------------------------------------------------

iMessage tool — sends texts through the macOS Messages app via osascript.

No API keys: AppleScript drives the locally signed-in Messages account.
First use triggers macOS Automation permission prompts (Messages, plus
Contacts when a name needs lookup) — approve once and they stick.

Contact resolution: if the recipient already looks like a phone number or
email it's used as-is; otherwise the Contacts app is searched by name and
the first match's first phone (then email) wins. Single-user simplicity —
same as addressing someone out loud, no disambiguation UI.

macOS-only by design: after the Phase 8 split this runs Brain-side (Mac),
so it never needs to work on the Pi.
"""

from __future__ import annotations

import re
import subprocess

from langchain_core.tools import BaseTool, StructuredTool

_OSASCRIPT_TIMEOUT_SECONDS = 15

_EMAIL_RE = re.compile(r"^\S+@\S+\.\S+$")
_PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")

_SEND_SCRIPT = """
tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{handle}" of targetService
    send "{message}" to targetBuddy
end tell
"""

_LOOKUP_SCRIPT = """
tell application "Contacts"
    set matches to (every person whose name contains "{name}")
    if (count of matches) = 0 then return ""
    set p to item 1 of matches
    if (count of phones of p) > 0 then return value of phone 1 of p
    if (count of emails of p) > 0 then return value of email 1 of p
    return ""
end tell
"""


def _escape(text: str) -> str:
    """Escape a Python string for embedding in an AppleScript literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


class IMessageTools:
    def _run_applescript(self, script: str) -> str:
        """Run osascript and return stdout. Raises RuntimeError on failure."""
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=_OSASCRIPT_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "osascript failed")
        return result.stdout.strip()

    def _resolve_handle(self, contact: str) -> tuple[str | None, str | None]:
        """Returns (handle, None) or (None, voice-ready error string)."""
        contact = contact.strip()
        if _EMAIL_RE.match(contact) or _PHONE_RE.match(contact):
            return contact, None
        try:
            handle = self._run_applescript(
                _LOOKUP_SCRIPT.format(name=_escape(contact))
            )
        except Exception as e:
            return None, f"Couldn't search Contacts: {e}"
        if not handle:
            return None, (
                f"I couldn't find '{contact}' in Contacts — try a phone "
                "number instead."
            )
        return handle, None

    def send_imessage(self, contact: str, message: str) -> str:
        handle, error = self._resolve_handle(contact)
        if error:
            return error
        try:
            self._run_applescript(
                _SEND_SCRIPT.format(
                    handle=_escape(handle), message=_escape(message)
                )
            )
        except Exception as e:
            return f"Couldn't send the message: {e}"
        return f"Sent to {contact.strip()}."

    def create_send_imessage_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.send_imessage,
            name="send_imessage",
            description=(
                "Send a text (iMessage) through the Mac's Messages app.\n\n"
                "  contact (str): a name from the user's contacts ('Mom', "
                "'Alex Kim') or a phone number / email.\n"
                "  message (str): the text to send — use the user's "
                "wording.\n\n"
                "Returns a confirmation or a plain-English error. If the "
                "name isn't found, ask the user for a phone number."
            ),
        )
