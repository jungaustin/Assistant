"""Discord REST client + per-channel read cursors.

Bot-token REST only — no gateway/websocket. The robot doesn't need live
events: it fetches history on demand when the user asks to catch up, and
posts through the same token. One credential covers reading and sending
(supersedes the old write-only webhook plan).

Read cursors: Discord never exposes *your* read state to a bot — it's
private per-user data — so Nemo keeps its own: the last message ID it
summarized per channel, in a JSON file under state/. The cursor can only
lag your real position (you read in the app, Nemo can't see that), never
overshoot it, so the worst case is a summary that re-covers ground.
mark_read snaps it forward after you've caught up by hand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from robot.config import DISCORD_BOT_TOKEN, DISCORD_CURSOR_PATH

_API_BASE = "https://discord.com/api/v10"
_REQUEST_TIMEOUT_SECONDS = 10

# First second of 2015 in Unix ms. Snowflake IDs carry ms-since-this in
# their top 42 bits, so any timestamp converts to a valid `after=` value —
# that's what makes "catch me up since 3pm" a pure ID comparison.
_DISCORD_EPOCH_MS = 1_420_070_400_000

_GUILD_TEXT_CHANNEL = 0


def snowflake_from_datetime(dt: datetime) -> int:
    """Synthetic snowflake for a moment in time (usable as `after=`)."""
    ms = int(dt.timestamp() * 1000) - _DISCORD_EPOCH_MS
    return max(ms, 0) << 22


def datetime_from_snowflake(snowflake: int | str) -> datetime:
    """When a real message ID was created (UTC)."""
    ms = (int(snowflake) >> 22) + _DISCORD_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class CursorStore:
    """Per-channel "last message Nemo summarized" IDs, persisted as JSON."""

    def __init__(self, path: str | Path):
        self._path = Path(path)

    def _load(self) -> dict:
        try:
            return json.loads(self._path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            # A lost cursor file costs one redundant summary — never let it
            # crash a tool call.
            return {}

    def get(self, channel_id: str) -> str | None:
        return self._load().get(str(channel_id))

    def set(self, channel_id: str, message_id: str | int) -> None:
        data = self._load()
        data[str(channel_id)] = str(message_id)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))


class DiscordClient:
    def __init__(self, token: str | None = None):
        self.token = token if token is not None else DISCORD_BOT_TOKEN
        self._channels: list[dict] | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        response = requests.request(
            method,
            f"{_API_BASE}{path}",
            headers={"Authorization": f"Bot {self.token}"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
        if response.status_code >= 400:
            # Surface Discord's own message — "Missing Access" reads better
            # over TTS than a bare 403.
            try:
                detail = response.json().get("message", "")
            except Exception:
                detail = ""
            raise RuntimeError(
                f"Discord API error {response.status_code}"
                + (f": {detail}" if detail else "")
            )
        return response

    def list_text_channels(self) -> list[dict]:
        """Every text channel the bot can see: {id, name, guild} dicts.

        Cached for the process lifetime — server membership doesn't change
        mid-session, and the guild walk is several round-trips.
        """
        if self._channels is None:
            channels: list[dict] = []
            guilds = self._request("GET", "/users/@me/guilds").json()
            for guild in guilds:
                listing = self._request(
                    "GET", f"/guilds/{guild['id']}/channels"
                ).json()
                for ch in listing:
                    if ch.get("type") == _GUILD_TEXT_CHANNEL:
                        channels.append(
                            {
                                "id": ch["id"],
                                "name": ch["name"],
                                "guild": guild["name"],
                            }
                        )
            self._channels = channels
        return self._channels

    def fetch_messages_after(
        self, channel_id: str, after: int | str, max_messages: int = 300
    ) -> list[dict]:
        """Messages newer than `after` (snowflake), oldest first."""
        collected: list[dict] = []
        cursor = str(after)
        while len(collected) < max_messages:
            limit = min(100, max_messages - len(collected))
            batch = self._request(
                "GET",
                f"/channels/{channel_id}/messages",
                params={"after": cursor, "limit": limit},
            ).json()
            if not batch:
                break
            # Discord returns newest-first within a page; we want oldest-first.
            batch.sort(key=lambda m: int(m["id"]))
            collected.extend(batch)
            cursor = batch[-1]["id"]
            if len(batch) < limit:
                break
        return collected

    def send_message(self, channel_id: str, content: str) -> None:
        self._request(
            "POST", f"/channels/{channel_id}/messages", json={"content": content}
        )
