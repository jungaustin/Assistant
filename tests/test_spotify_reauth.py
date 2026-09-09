"""Tests for Spotify refresh-token handling under the six-month expiry policy.

Spotify expires refresh tokens after six months (effective 2026-07-20). When a
token expires or is revoked, the token endpoint returns invalid_grant. The
runtime must: detect it, discard the dead token, NOT retry, and surface a
speakable re-auth message instead of crashing the conversation turn. It must
also persist a rotated refresh token when Spotify hands out a new one on an
otherwise-normal refresh.

All HTTP is mocked — no network happens.
"""

from __future__ import annotations

import socket
import urllib.parse
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from robot.tools.inner.spotify_auth import SpotifyAuthUnavailable
from robot.tools.inner.spotify_client import (
    SpotifyClient,
    SpotifyReauthRequired,
    _BROWSER_OPEN_MESSAGE,
    _REAUTH_MESSAGE,
)
from robot.tools.inner.spotify_tools import SpotifyTools


def _client() -> SpotifyClient:
    c = SpotifyClient()
    c.CLIENT_ID = "id"
    c.CLIENT_SECRET = "secret"
    c.REFRESH_TOKEN = "stored-refresh"
    c.access_token = None
    c.expires_at = None
    return c


def _resp(status: int, body: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    return r


def test_invalid_grant_raises_discards_and_does_not_retry():
    c = _client()
    resp = _resp(400, {"error": "invalid_grant", "error_description": "revoked"})
    with patch(
        "robot.tools.inner.spotify_client.requests.post", return_value=resp
    ) as post:
        with pytest.raises(SpotifyReauthRequired):
            c.refresh_token()
    assert post.call_count == 1, "must not retry a failed refresh"
    assert c.REFRESH_TOKEN is None, "dead token must be discarded"
    assert c.access_token is None


def test_missing_token_signals_reauth_without_network():
    c = _client()
    c.REFRESH_TOKEN = None
    with patch("robot.tools.inner.spotify_client.requests.post") as post:
        with pytest.raises(SpotifyReauthRequired):
            c.refresh_token()
    post.assert_not_called()


def test_tool_method_returns_speakable_message_when_user_never_clicks():
    # Browser opens, user doesn't approve inside the window: the turn still
    # ends with something speakable rather than a stack trace.
    c = _client()
    resp = _resp(400, {"error": "invalid_grant"})
    with patch("robot.tools.inner.spotify_client.requests.post", return_value=resp), \
         patch.object(SpotifyClient, "await_reauth",
                      return_value=(False, _BROWSER_OPEN_MESSAGE)):
        msg = c.play_song("track:Ditto artist:NewJeans")
    assert msg == _BROWSER_OPEN_MESSAGE


def test_other_refresh_failure_raises_runtime_error_not_keyerror():
    c = _client()
    resp = _resp(503, {"error": "server_error"})
    with patch("robot.tools.inner.spotify_client.requests.post", return_value=resp):
        with pytest.raises(RuntimeError):
            c.refresh_token()


def test_rotated_refresh_token_is_persisted():
    c = _client()
    resp = _resp(
        200,
        {"access_token": "AT", "expires_in": 3600, "refresh_token": "rotated"},
    )
    with patch(
        "robot.tools.inner.spotify_client.requests.post", return_value=resp
    ), patch(
        "robot.tools.inner.spotify_client.find_dotenv", return_value="/tmp/x.env"
    ), patch(
        "robot.tools.inner.spotify_client.set_key"
    ) as set_key:
        c.refresh_token()
    assert c.REFRESH_TOKEN == "rotated"
    set_key.assert_called_once_with("/tmp/x.env", "REFRESH_TOKEN", "rotated")


def test_normal_refresh_without_rotation_does_not_touch_env():
    c = _client()
    resp = _resp(200, {"access_token": "AT", "expires_in": 3600})
    with patch(
        "robot.tools.inner.spotify_client.requests.post", return_value=resp
    ), patch("robot.tools.inner.spotify_client.set_key") as set_key:
        c.refresh_token()
    assert c.access_token == "AT"
    assert c.REFRESH_TOKEN == "stored-refresh"
    set_key.assert_not_called()


def test_reauth_guard_preserves_tool_arg_schema():
    # The guard decorator must not hide the typed signature LangChain needs.
    tool = SpotifyTools(_client()).create_play_song_tool()
    assert "query" in tool.args


# ---------------------------------------------------------------------------
# Browser re-authorization
#
# The six-month expiry used to dead-end the conversation: the robot could only
# tell the user to go run a script in a terminal. These cover the recovery
# path — pop the Spotify login, take the one click only a human can make, and
# resume the request that triggered it.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _local_flow_client() -> SpotifyClient:
    c = _client()
    c.REFRESH_TOKEN = None
    c.REDIRECT_URI = f"http://127.0.0.1:{_free_port()}/callback"
    return c


def test_expired_login_opens_browser_and_retries_the_original_call():
    # The whole point: the user asked for something, so once they approve we
    # do the thing they asked for instead of making them repeat it.
    c = _client()
    devices = _resp(200, {"devices": [{"name": "MacBook Pro", "id": "dev1"}]})
    paused = _resp(204, {})
    expired = _resp(400, {"error": "invalid_grant"})

    def approve():
        c.access_token = "fresh-AT"
        c.expires_at = datetime.now().timestamp() + 3600
        return True, ""

    with patch(
        "robot.tools.inner.spotify_client.requests.post", return_value=expired
    ), patch(
        "robot.tools.inner.spotify_client.requests.get", return_value=devices
    ), patch(
        "robot.tools.inner.spotify_client.requests.put", return_value=paused
    ), patch.object(
        SpotifyClient, "await_reauth", side_effect=approve
    ) as await_reauth:
        result = c.pause_playback()

    assert result == "Paused"
    assert await_reauth.call_count == 1


def test_auto_reauth_can_be_disabled():
    c = _client()
    with patch("robot.tools.inner.spotify_client._auto_reauth_enabled",
               return_value=False), \
         patch.object(SpotifyClient, "begin_reauth") as begin:
        reconnected, message = c.await_reauth()
    assert reconnected is False
    assert message == _REAUTH_MESSAGE
    begin.assert_not_called()


def test_unstartable_flow_falls_back_to_the_manual_instructions():
    # Port squatted on (macOS AirPlay likes 5000) or credentials missing:
    # there is nothing a browser can do, so send the user to the script.
    c = _client()
    with patch.object(
        SpotifyClient, "begin_reauth", side_effect=SpotifyAuthUnavailable("port busy")
    ):
        reconnected, message = c.await_reauth()
    assert reconnected is False
    assert message == _REAUTH_MESSAGE


def test_missing_credentials_cannot_start_a_flow():
    c = _client()
    c.CLIENT_ID = None
    with pytest.raises(SpotifyAuthUnavailable):
        c.begin_reauth()


def test_second_tool_call_in_a_turn_reuses_the_pending_flow():
    # "play my smoothie playlist shuffled" is two tool calls; two browser tabs
    # fighting over one callback port would be a mess.
    c = _local_flow_client()
    with patch("robot.tools.inner.spotify_auth.webbrowser.open", return_value=True) as op:
        first = c.begin_reauth()
        try:
            second = c.begin_reauth()
            assert first is second
            assert op.call_count == 1
        finally:
            first.fail("test teardown")


def test_browser_flow_updates_the_live_client_end_to_end():
    c = _local_flow_client()
    opened: dict[str, str] = {}
    exchange = _resp(
        200,
        {"access_token": "new-AT", "expires_in": 3600, "refresh_token": "new-RT"},
    )

    def fake_open(url):
        opened["url"] = url
        return True

    with patch(
        "robot.tools.inner.spotify_auth.webbrowser.open", side_effect=fake_open
    ), patch(
        "robot.tools.inner.spotify_auth.requests.post", return_value=exchange
    ), patch.object(
        SpotifyClient, "_persist_refresh_token"
    ) as persist:
        flow = c.begin_reauth()
        query = urllib.parse.parse_qs(urllib.parse.urlparse(opened["url"]).query)
        assert query["redirect_uri"] == [c.REDIRECT_URI]

        # Stand in for the click the user makes in the browser.
        page = requests.get(
            c.REDIRECT_URI,
            params={"code": "auth-code", "state": query["state"][0]},
            timeout=5,
        )
        assert flow.wait(5)

    assert page.status_code == 200
    assert flow.succeeded
    # Applied in-process, not merely written to .env: the turn that triggered
    # this has to be able to continue without a restart.
    assert c.access_token == "new-AT"
    assert c.REFRESH_TOKEN == "new-RT"
    assert c.is_token_valid()
    persist.assert_called_once_with("new-RT")


def test_callback_rejects_a_mismatched_state():
    # The callback port is open to anything on loopback; state is what stops
    # another local process from feeding us its own code.
    c = _local_flow_client()
    with patch("robot.tools.inner.spotify_auth.webbrowser.open", return_value=True), \
         patch("robot.tools.inner.spotify_auth.requests.post") as post:
        flow = c.begin_reauth()
        page = requests.get(
            c.REDIRECT_URI, params={"code": "x", "state": "wrong"}, timeout=5
        )
        assert flow.wait(5)

    assert page.status_code == 400
    post.assert_not_called()
    assert flow.error
    assert c.REFRESH_TOKEN is None


def test_stray_browser_request_does_not_end_the_flow():
    # Browsers fetch /favicon.ico unprompted; that must not count as the
    # user's answer.
    c = _local_flow_client()
    with patch("robot.tools.inner.spotify_auth.webbrowser.open", return_value=True):
        flow = c.begin_reauth()
        try:
            page = requests.get(
                f"http://127.0.0.1:{flow.port}/favicon.ico", timeout=5
            )
            assert page.status_code == 404
            assert not flow.done.is_set()
        finally:
            flow.fail("test teardown")


def test_reauthorize_tool_is_registered_and_takes_no_arguments():
    tool = SpotifyTools(_client()).create_reauthorize_spotify_tool()
    assert tool.name == "reauthorize_spotify"
    assert tool.args == {}
