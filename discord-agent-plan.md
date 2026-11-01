# Discord agent channel — build plan (2026-10-18)

Goal: post a message in my private agent channel on Discord and Nemo acts on
it (log food, query totals, anything the voice agent can do) and replies in
the channel. Text only for v1; images later via a vision-describer step.

## Decisions (already made — don't re-litigate)

- **Gateway, not polling.** Use `discord.py`; Discord pushes MESSAGE_CREATE
  over one websocket. No timer loop.
- **Separate process**, `just discord`, not bolted into `robot.main` or
  `brain_server`. It builds its own `Agent` and talks to the same Ollama.
  Voice and Discord can run at the same time.
- **Own conversation thread.** Discord turns use thread id
  `discord-YYYY-MM-DD` (still rolls over at local midnight), so a Discord
  chat and a voice chat never interleave in one LangGraph checkpoint across
  two processes. Shared data (tracker.db, memory.db) is shared on purpose —
  logging from Discord must show up in voice queries and vice versa.
- **Owner-only.** Only messages from the owner, in the configured channel,
  from a non-bot author, reach the agent. Everything else is ignored silently.
  This matters: the agent has DoorDash/send-message tools.
- **Text only.** A message with attachments but no text gets a short reply
  ("I can't see images yet — describe it in text"). Text + attachment: run
  the text, ignore the attachment.

## Config (robot/src/robot/config.py, next to DISCORD_BOT_TOKEN)

| Var | Required | Meaning |
|---|---|---|
| `DISCORD_BOT_TOKEN` | yes (exists) | same bot as the catch-up tools |
| `DISCORD_AGENT_CHANNEL_ID` | yes | numeric channel id of the agent channel |
| `DISCORD_OWNER_ID` | no | numeric user id allowed to command the agent. Default: the bot application's owner, from `await client.application_info()` (`.owner.id`, or the team owner if it's a team app) |
| `DISCORD_AGENT_STATE_PATH` | no | default `state/discord_agent.json` (last handled message id) |

Add all four to `.env.example` with a comment block like the existing Discord
one. Missing token or channel id → exit with a clear message (same style as
`brain_server.py`'s TRANSPORT_TOKEN check).

## Code

### 1. `Agent` thread prefix — `src/robot/brain/agent.py`, `src/robot/config.py`

Add an optional `thread_prefix: str = ""` to `Agent.__init__`. Default thread
becomes `thread_prefix + daily_thread_id()`; `_roll_thread_if_new_day`
compares/adopts the prefixed id. Explicit `thread_id` behaviour unchanged
(still pinned). Voice path passes nothing → byte-for-byte same as today.

### 2. `src/robot/discord_agent.py` (new, run as `python -m robot.discord_agent`)

- `discord.Client` with `Intents.default()` + `message_content = True`.
- `on_ready`: resolve owner id (env override else application owner), log
  it, then **catch up**: fetch channel history after the stored last-handled
  id (cap ~20 messages, oldest first) and process owner messages in order.
  First ever run (no stored id): don't replay history — store the channel's
  latest message id and start fresh. Gateway doesn't replay messages sent
  while the process was down; this covers laptop sleep / restarts.
- `on_message`: filter (channel, author == owner, not bot, non-empty
  content), then hand to the turn worker.
- **Turn worker:** an `asyncio.Lock` (or a queue) so turns run strictly one
  at a time in arrival order. `Agent.run` is blocking → call it via
  `asyncio.to_thread`. Show `async with channel.typing():` while it runs.
- Reply with `message.reply(...)` (mention off: `mention_author=False`).
  Split anything over 2000 chars on line boundaries (pure helper, tested).
  Empty agent reply → reply "(no response)" rather than nothing.
- Agent exception → log with traceback, reply "Something broke on my end —
  check the logs." and keep running. Never let one bad turn kill the process.
- After a turn finishes (success or error), persist that message id as
  last-handled (atomic write: temp file + rename).
- Build the Agent with `thread_prefix="discord-"` and call
  `agent.start_prewarm()` at startup like brain_server does.
- Use `robot.core.logging` (`configure_logging`, `get_logger`) like
  brain_server.

Keep the pure logic (should-handle filter, message splitting, state file
read/write) as plain functions so they test without a Discord connection.

### 3. `justfile` — new `discord` recipe

Same Ollama start/stop block as `brain` (copy it, including the
`OLLAMA_CONTEXT_LENGTH`/`KEEP_ALIVE` env and the "no curl pre-warm" note
reference), then `cd robot && {{uv}} run python -m robot.discord_agent`.

### 4. Dependency

`cd robot && /opt/homebrew/bin/uv add discord.py` (arm64 uv — see the
justfile note), then `just sync` so the RealtimeSTT patch is re-applied and
dev extras stay installed.

## Tests (`tests/test_discord_agent.py`)

- should-handle filter: wrong channel, other user, bot author, empty text,
  attachment-only (→ the "can't see images" path) , valid owner message.
- message splitter: <2000 untouched; long text split on newlines; a single
  line >2000 hard-split; no chunk exceeds 2000.
- state file: missing/corrupt file → None; write then read round-trips.
- `Agent(thread_prefix="discord-")` thread id is `discord-<today>`; rollover
  keeps the prefix (follow the pattern in `test_datetime_rollover.py`).
- Turn handling with a fake agent + fake channel/message objects: replies
  with agent output; agent raising → error reply, process keeps going,
  last-handled id still advances.

Run with `just test` (uses the project venv via uv — never system Python).
All existing tests must still pass.

## Out of scope for v1

Images (planned: Ollama vision model describes the image → text into the
same agent), slash commands, multiple channels, running as a LaunchAgent.

## Manual check (me, after build)

1. Discord developer portal → bot → **Message Content Intent** on.
2. Put `DISCORD_AGENT_CHANNEL_ID` in `robot/.env`.
3. `just discord`, send "how many calories did I eat yesterday" in the channel.
