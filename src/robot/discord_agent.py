"""Discord front-end for the Agent: the third runtime next to voice and brain.

Post in the private agent channel and Nemo acts on it and replies there. Run
with `just discord` (starts Ollama too) or `python -m robot.discord_agent`.

A separate process, not bolted into robot.main or brain_server: it builds its
own Agent against the same Ollama and the same tracker.db / memory.db, so a
meal logged from Discord shows up in voice queries and vice versa. Its
conversation thread is namespaced ("discord-YYYY-MM-DD") so the two processes
never interleave turns in one LangGraph checkpoint.

Gateway (websocket push), not polling. The gateway does not replay messages
sent while the process was down, so on_ready catches up from the last handled
message id kept in a small state file.

Owner-only on purpose: the agent has DoorDash and send-message tools, so
anything not from the owner, in the agent channel, from a human, is ignored
silently.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import discord

from robot.brain import Agent
from robot.config import (
    DISCORD_AGENT_CHANNEL_ID,
    DISCORD_AGENT_STATE_PATH,
    DISCORD_BOT_TOKEN,
    DISCORD_OWNER_ID,
)
from robot.core.logging import configure_logging, get_logger

log = get_logger(__name__)

# Discord rejects messages over 2000 characters.
MAX_MESSAGE_CHARS = 2000
# Backlog replayed after downtime. Bounded so a long outage can't queue an
# hour of slow local-LLM turns.
CATCH_UP_LIMIT = 20

NO_IMAGES_REPLY = "I can't see images yet — describe it in text."
EMPTY_REPLY = "(no response)"
ERROR_REPLY = "Something broke on my end — check the logs."

# should_handle verdicts.
IGNORE = "ignore"
RUN = "run"
NO_IMAGES = "no_images"


def should_handle(message, channel_id: int, owner_id: int) -> str:
    """Decide what to do with a message: IGNORE, RUN, or NO_IMAGES.

    Silent ignore for everything that isn't the owner, in the agent channel,
    from a non-bot author. Text + attachment runs the text; attachment-only
    gets the "can't see images" reply. Empty and attachment-free is ignored.
    """
    if message.channel.id != channel_id:
        return IGNORE
    if message.author.bot or message.author.id != owner_id:
        return IGNORE
    if (message.content or "").strip():
        return RUN
    if message.attachments:
        return NO_IMAGES
    return IGNORE


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split text into chunks of at most `limit`, preferring line boundaries.

    A single line longer than the limit is hard-split, since there is no
    better boundary to find.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        # Hard-split an over-long line into limit-sized pieces first.
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if not current:
            current = line
        elif len(current) + 1 + len(line) <= limit:
            current += "\n" + line
        else:
            chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return chunks


def read_last_handled(path: str | Path) -> int | None:
    """Last handled message id, or None if missing/corrupt.

    None just means "start fresh" — a lost state file must never crash boot.
    """
    try:
        return int(json.loads(Path(path).read_text())["last_message_id"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_last_handled(path: str | Path, message_id: int) -> None:
    """Atomic write (temp file + rename) so a crash mid-write can't leave a
    truncated file that reads back as corrupt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"last_message_id": str(message_id)}))
    os.replace(tmp, path)


class DiscordAgent:
    """Turn handling, separated from the discord.Client so it tests with fakes."""

    def __init__(self, agent, channel_id: int, owner_id: int | None, state_path):
        self.agent = agent
        self.channel_id = channel_id
        self.owner_id = owner_id
        self.state_path = state_path
        # Turns run strictly one at a time, in arrival order. Agent.run is
        # not reentrant and the local LLM is serial anyway.
        self._lock = asyncio.Lock()

    def _mark_handled(self, message_id: int) -> None:
        last = read_last_handled(self.state_path)
        if last is None or message_id > last:
            write_last_handled(self.state_path, message_id)

    async def handle_message(self, message) -> None:
        verdict = should_handle(message, self.channel_id, self.owner_id)
        if verdict == IGNORE:
            return
        async with self._lock:
            # on_ready fires again on every gateway reconnect, and its
            # catch_up can fetch a message that on_message is already
            # handling. Checked under the lock so the second copy sees the
            # first one's mark — a duplicate turn would double-log a meal.
            last = read_last_handled(self.state_path)
            if last is not None and message.id <= last:
                return
            try:
                if verdict == NO_IMAGES:
                    await message.reply(NO_IMAGES_REPLY, mention_author=False)
                else:
                    await self._run_turn(message)
            except Exception:
                # Even the error reply can fail (deleted message, network).
                # One bad turn must not kill the process.
                log.exception("discord_turn_failed", message_id=message.id)
            finally:
                # Advance on error too, or a poison message would be replayed
                # (and fail again) on every restart.
                self._mark_handled(message.id)

    async def _run_turn(self, message) -> None:
        log.info("discord_turn", message_id=message.id)
        try:
            async with message.channel.typing():
                reply = await asyncio.to_thread(self.agent.run, message.content)
        except Exception:
            log.exception("discord_agent_error", message_id=message.id)
            await message.reply(ERROR_REPLY, mention_author=False)
            return
        for chunk in split_message((reply or "").strip() or EMPTY_REPLY):
            await message.reply(chunk, mention_author=False)

    async def catch_up(self, channel) -> None:
        """Process owner messages missed while down; fresh start on first run."""
        last = read_last_handled(self.state_path)
        if last is None:
            # Don't replay history on first ever run: just remember "now".
            async for latest in channel.history(limit=1):
                self._mark_handled(latest.id)
            log.info("discord_first_run_baseline")
            return
        missed = [
            m
            async for m in channel.history(
                limit=CATCH_UP_LIMIT, after=discord.Object(id=last), oldest_first=True
            )
        ]
        log.info("discord_catch_up", count=len(missed))
        for m in missed:
            await self.handle_message(m)


async def _resolve_owner_id(client: discord.Client) -> int:
    if DISCORD_OWNER_ID:
        return int(DISCORD_OWNER_ID)
    info = await client.application_info()
    # Team-owned apps expose the team, whose owner_id is the human.
    if info.team is not None:
        return info.team.owner_id
    return info.owner.id


def main() -> None:
    configure_logging()
    if not DISCORD_BOT_TOKEN:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Create a bot at "
            "https://discord.com/developers/applications and put the token "
            "in .env (see .env.example)."
        )
    if not DISCORD_AGENT_CHANNEL_ID.strip().isdigit():
        raise SystemExit(
            "DISCORD_AGENT_CHANNEL_ID is not set (or not numeric). Enable "
            "Developer Mode in Discord, right-click the agent channel, Copy "
            "Channel ID, and put it in .env."
        )
    channel_id = int(DISCORD_AGENT_CHANNEL_ID)

    log.info("discord_agent_starting")
    # "discord-" keeps this process's day thread separate from voice's.
    agent = Agent(thread_prefix="discord-")
    # Same prompt-cache prewarm as brain_server.
    agent.start_prewarm()
    bot = DiscordAgent(agent, channel_id, None, DISCORD_AGENT_STATE_PATH)

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        bot.owner_id = await _resolve_owner_id(client)
        log.info("discord_agent_ready", owner_id=bot.owner_id, channel_id=channel_id)
        channel = client.get_channel(channel_id) or await client.fetch_channel(channel_id)
        await bot.catch_up(channel)

    @client.event
    async def on_message(message):
        # Messages before on_ready finishes resolving the owner are dropped
        # here; catch_up picks them up from history.
        if bot.owner_id is None:
            return
        await bot.handle_message(message)

    client.run(DISCORD_BOT_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
