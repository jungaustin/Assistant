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

from unittest.mock import MagicMock, patch

import pytest

from robot.tools.inner.spotify_client import (
    SpotifyClient,
    SpotifyReauthRequired,
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


def test_tool_method_returns_speakable_message_on_expiry():
    c = _client()
    resp = _resp(400, {"error": "invalid_grant"})
    with patch("robot.tools.inner.spotify_client.requests.post", return_value=resp):
        msg = c.play_song("track:Ditto artist:NewJeans")
    assert msg == _REAUTH_MESSAGE


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
