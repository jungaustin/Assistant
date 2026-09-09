"""LangGraph agent. Streams tokens, dispatches tool calls, persists
conversation memory to sqlite so the process survives restart.
"""

from __future__ import annotations

import atexit
import json
import logging
import re
import sqlite3
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from robot.config import (
    BRAIN_PREWARM,
    FABRICATION_RETRY_TIMEOUT,
    MAX_HISTORY_MESSAGES,
    MEMORY_DB_PATH,
    STATE_DB_PATH,
    daily_thread_id,
    load_persona,
    make_llm,
)
from robot.memory import MemoryStore
from robot.tools.manager import ToolManager

logger = logging.getLogger(__name__)


def _truncate(value, limit: int = 200) -> str:
    """Render a value to a single short string for a log line.

    Tool args/results can be large (web search hits, playlist dumps); cap them
    so the log stays scannable while still showing what was passed.
    """
    s = str(value).replace("\n", " ")
    if len(s) <= limit:
        return s
    return s[:limit] + f"… (+{len(s) - limit} chars)"


def _dedupe_tool_calls(response):
    """Collapse byte-identical tool calls the model emitted in one turn.

    A single assistant turn occasionally carries the *same* tool call twice —
    identical name and identical args — which is almost never intended and,
    for a data-logging tool like log_entry, silently double-writes the user's
    data (e.g. logging one meal as two 720-cal entries). This is a known model
    failure mode, not something the user asked for.

    We key on (name, canonical-JSON of args) and keep the first occurrence.
    Because our OpenAI payload is rebuilt from `AIMessage.tool_calls`
    (_lc_messages_to_openai), dropping a call here also drops it from the
    serialized assistant message, so no orphaned tool_call id survives to
    trip OpenAI's "tool_call without a response" 400 on the next round-trip.

    Distinct args are left untouched — two different log_entry calls in one
    turn (logging lunch and dinner together) are legitimate and pass through.
    Returns `response` unchanged when there's nothing to drop.
    """
    tool_calls = getattr(response, "tool_calls", None) or []
    if len(tool_calls) < 2:
        return response

    seen: set[tuple[str, str]] = set()
    deduped: list = []
    for tc in tool_calls:
        try:
            args_key = json.dumps(tc.get("args") or {}, sort_keys=True)
        except (TypeError, ValueError):
            args_key = repr(tc.get("args"))
        key = (tc.get("name", ""), args_key)
        if key in seen:
            logger.warning(
                "dropped duplicate tool call name=%s args=%s",
                tc.get("name"),
                _truncate(tc.get("args")),
            )
            continue
        seen.add(key)
        deduped.append(tc)

    if len(deduped) != len(tool_calls):
        response.tool_calls = deduped
    return response


# Tools that actually change the log database. A reply claiming one of these
# happened is only true if the matching tool ran in the same turn.
_WRITE_TOOLS = {"log_entry", "log_meal", "update_entry"}

# The model sometimes WRITES a tool call instead of making one — it emits
# "[calling query_entries]" as plain text and then invents the result. Observed
# on qwen2.5:14b: four meals "logged" that never reached sqlite, followed by a
# fabricated day's totals read back as fact. Silent data loss is the worst
# failure this robot has, so both halves get caught here rather than spoken.
_NARRATED_CALL_RE = re.compile(
    r"\[\s*(?:call|calls|calling|invoke|invoking|use|using)\s+[a-z_][a-z0-9_]*",
    re.IGNORECASE,
)

# "Logged 600 for rice." — a write claim carrying a number. Requiring the digit
# keeps this off ordinary prose ("I moved it", "saved you a step").
_CLAIMS_WRITE_RE = re.compile(
    r"\b(logged|recorded|saved)\b[^.\n]{0,40}?\d", re.IGNORECASE
)

# "Deleted today's entries." — said on 2026-09-01 about a delete the guessed-id
# guard had just blocked. The write check above could not see it: log_entry had
# run that turn, so the turn was not tool-less. A removal claim needs its own
# check against a removal actually happening.
_CLAIMS_DELETE_RE = re.compile(
    r"\b(deleted|removed|cleared|erased|got rid of)\b[^.\n]{0,30}?"
    r"\b(entry|entries|row|rows|log|logs|one|ones|it|them)\b",
    re.IGNORECASE,
)

_FABRICATION_NUDGE = (
    "Your last reply described a tool call in words instead of making one. "
    "Writing about a tool does nothing — only a real tool call touches the "
    "database. Answer again and make the actual call. Never emit bracketed "
    "text like '[calling log_entry]', and never state a number you did not "
    "get back from a tool."
)

# Spoken when the model fabricates twice. Deliberately vague about what failed
# and explicit that nothing happened — the user has to know not to trust it.
_FABRICATION_FALLBACK = "Sorry — that didn't go through. Nothing was saved. Ask me again?"

# A deletion that was refused often sits alongside a log that DID succeed, so
# the generic "nothing was saved" line above would be untrue in the one
# direction that matters. Keep each message to what actually failed.
_FABRICATION_FALLBACKS = {
    "claimed a deletion that no tool performed":
        "Sorry — I couldn't remove that, so nothing was deleted. "
        "Ask me to check what's logged and I'll remove it by number.",
}


def _fallback_for(reason: str) -> str:
    return _FABRICATION_FALLBACKS.get(reason, _FABRICATION_FALLBACK)


def _tools_run_this_turn(history: list) -> set[str]:
    """Names of tools that actually executed since the last user message."""
    names = set()
    for msg in reversed(history):
        if getattr(msg, "type", None) == "human":
            break
        if getattr(msg, "type", None) == "tool":
            name = getattr(msg, "name", None)
            if name:
                names.add(name)
    return names


def _fabrication_reason(response, history: list) -> str | None:
    """Is this reply claiming work that no tool actually did?

    Only ever consulted when the model returned zero tool calls — a reply with
    real tool calls is about to do the work for real.
    """
    if getattr(response, "tool_calls", None):
        return None
    text = _content_to_text(getattr(response, "content", None))
    if not text.strip():
        return None
    if _NARRATED_CALL_RE.search(text):
        return "narrated a tool call in prose"
    # Any tool actually ran this turn => the reply is grounded in a real result,
    # so leave it alone. Requiring a *write* tool specifically was too strict: a
    # legitimate query readback ("You logged 1050 for rice and 240 for tofu,
    # 1290 total") says "logged" about rows query_entries really returned, and
    # got blocked. The fabrication this guards against emits a confirmation with
    # no tool call at all.
    if _CLAIMS_WRITE_RE.search(text) and not _tools_run_this_turn(history):
        return "claimed a write that no tool performed"
    # A removal is checked against removal tools specifically: other tools
    # running that turn (a log_entry alongside it) say nothing about whether
    # anything was actually deleted.
    if _CLAIMS_DELETE_RE.search(text) and not (
        _tools_run_this_turn(history) & _DESTRUCTIVE_TOOLS
    ):
        return "claimed a deletion that no tool performed"
    return None


# Tools that take an entry_id. That id must be one the model actually SAW,
# never one it inferred.
_ID_TARGETED_TOOLS = {"delete_entry", "update_entry"}

# Everything that can remove rows. delete_entries takes filters rather than an
# id, so it needs no id check — but a "I removed it" claim must still be backed
# by one of these having run.
_DESTRUCTIVE_TOOLS = {"delete_entry", "update_entry", "delete_entries"}

# Row ids are surfaced as "#331" by log_entry/log_meal/query_entries.
_ROW_ID_RE = re.compile(r"#(\d+)")

# An id the USER named, which needs an explicit marker to count.
_SPOKEN_ID_RE = re.compile(r"(?:#|\bentry\s+(?:id\s+)?|\brow\s+)(\d+)", re.I)


def _ids_the_model_has_seen(history: list) -> set[str]:
    """Row ids that appeared in this turn's tool results, or that the user said.

    Anything else in a delete/update call is a guess. On 2026-09-01 the model
    answered "remove today's entries and relog" by firing query_entries,
    delete_entry(entry_id=1) and log_entry in ONE batch — ToolNode runs a batch
    concurrently, so the delete went out before the query could answer, and
    entry #1 (a steak logged in June) was destroyed instead of today's rows.
    """
    seen: set[str] = set()
    for msg in history:
        kind = getattr(msg, "type", None)
        text = _content_to_text(getattr(msg, "content", None))
        if kind == "tool":
            # Every tool result still in the window counts, not just this
            # turn's. "Logged #5" one turn ago is a real row the model saw, and
            # rejecting it broke the obvious follow-up ("delete that one too").
            # A stale id at worst deletes nothing; the point of this guard is to
            # stop INVENTED ids, and an id can only get here by being returned.
            seen.update(_ROW_ID_RE.findall(text))
    for msg in reversed(history):
        kind = getattr(msg, "type", None)
        text = _content_to_text(getattr(msg, "content", None))
        if kind == "human":
            # The user may name an id out loud ("delete entry 328", "#328").
            # Deliberately NOT any bare number: almost every number the user
            # says is a calorie value, and "relog 1,620" contains a "1" that
            # would authorise deleting entry #1 — which is very likely how the
            # model landed on that id in the first place.
            seen.update(_SPOKEN_ID_RE.findall(text))
            break
    return seen


def _drop_unverified_destructive_calls(response, history: list):
    """Remove delete/update calls targeting an id the model never saw.

    Reads issued in the SAME batch cannot have answered yet, so an id that only
    a same-batch query could supply is by definition guessed. Dropping the call
    (rather than running it) lets the tool results come back and the model
    reissue it against real ids on the next pass.
    """
    calls = list(getattr(response, "tool_calls", None) or [])
    if not any(c.get("name") in _ID_TARGETED_TOOLS for c in calls):
        return response

    seen = _ids_the_model_has_seen(history)
    kept = []
    for call in calls:
        if call.get("name") in _ID_TARGETED_TOOLS:
            target = str((call.get("args") or {}).get("entry_id", ""))
            if target not in seen:
                logger.error(
                    "dropped %s(entry_id=%s): id never appeared in a tool result "
                    "this turn (seen=%s) — refusing to delete a guessed row",
                    call.get("name"), target, sorted(seen) or "none",
                )
                continue
        kept.append(call)

    if len(kept) != len(calls):
        response.tool_calls = kept
        # Dropping the only call leaves an empty turn — the robot just goes
        # silent, which reads as a crash. Say what happened instead.
        if not kept and not _content_to_text(getattr(response, "content", None)).strip():
            response.content = (
                "I couldn't tell which entry you meant, so I didn't delete "
                "anything. Tell me the day or the food and I'll remove it."
            )
    return response


def _trim_history(messages: list, max_messages: int) -> list:
    """Bound what the LLM sees to roughly the last `max_messages`.

    A tool turn costs two LLM round-trips over the *entire* thread, and the
    daily thread only grows, so latency climbs through the day. Trimming the
    per-call prompt fixes that. The full history is still checkpointed and
    older context stays reachable via recall() — this only shrinks the window
    sent to the model.

    `start_on="human"` keeps the window from beginning mid tool-exchange
    (OpenAI 400s on a tool result whose tool_call was trimmed away). If the
    last `max_messages` don't reach back to a user turn — a single turn with
    more tool rounds than the window — that guard would return nothing, so we
    fall back to everything since the most recent user message.
    """
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=len,  # count messages, not tokens
        max_tokens=max_messages,
        start_on="human",
        include_system=False,
    )
    if trimmed or not messages:
        return trimmed
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


class ToolCallLogger(BaseCallbackHandler):
    """Logs every tool call's start, finish (with duration), and errors.

    Wired into the agent's run config so each ToolNode invocation emits a
    line. The point is "where did it get stuck": a ``tool start`` with no
    matching ``tool done`` is the call that hung. Pair this with the
    ``llm start``/``llm done`` lines from the assistant node to tell a stuck
    model round-trip apart from a stuck tool.

    Keyed by run_id so concurrent tool calls (ToolNode can fan out) don't
    clobber each other's timers.
    """

    def __init__(self) -> None:
        self._starts: dict[UUID, tuple[str, float]] = {}

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or "tool"
        self._starts[run_id] = (name, time.monotonic())
        args = inputs if inputs is not None else input_str
        logger.info("tool start name=%s args=%s", name, _truncate(args))

    def on_tool_end(self, output, *, run_id, **kwargs):
        name, t0 = self._starts.pop(run_id, ("tool", time.monotonic()))
        logger.info(
            "tool done  name=%s elapsed=%.2fs result=%s",
            name,
            time.monotonic() - t0,
            _truncate(output),
        )

    def on_tool_error(self, error, *, run_id, **kwargs):
        name, t0 = self._starts.pop(run_id, ("tool", time.monotonic()))
        logger.warning(
            "tool error name=%s elapsed=%.2fs error=%s: %s",
            name,
            time.monotonic() - t0,
            error.__class__.__name__,
            error,
        )


def build_datetime_context() -> str:
    """Return a ~25-token datetime block for injection into the system prompt.

    Gives Nemo: current date, this week's Monday (for period_start on week notes),
    and this month's first day (for period_start on month notes). Local time, not
    UTC — matches how the user thinks about "today".
    """
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    return (
        f"Current date: {today.isoformat()}. "
        f"This week's Monday: {monday.isoformat()}. "
        f"This month's start: {month_start.isoformat()}."
    )


def _open_checkpoint_db(db_path: str) -> sqlite3.Connection:
    """Open the checkpoint DB, creating parent dirs as needed.

    `check_same_thread=False`: LangGraph dispatches some work onto a thread
    pool (the streaming machinery uses futures), and sqlite3 by default
    refuses cross-thread use of a connection. SqliteSaver serializes
    writes internally, so this is safe.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), check_same_thread=False)


class Agent:
    """Conversational agent with persistent memory.

    thread_id defaults to today's local date (see config.daily_thread_id),
    so turns within a day chain together and conversations reset overnight.
    Pass an explicit `thread_id` for tests or off-policy sessions.
    """

    def __init__(
        self,
        llm=None,
        thread_id: Optional[str] = None,
        checkpointer: Optional[SqliteSaver] = None,
        memory_store: Optional[MemoryStore] = None,
        clip_service=None,
    ):
        self.thread_id = thread_id or daily_thread_id()
        # An explicitly passed thread_id is pinned (tests, off-policy
        # sessions); only the default day-scoped policy rolls at midnight.
        self._thread_pinned = thread_id is not None
        # Persona is read once (file IO); the datetime block is deliberately
        # NOT baked in here — it's rebuilt per turn in _system_message() so a
        # process running past local midnight doesn't keep yesterday's date.
        self._persona = load_persona()

        # Working memory: SqliteSaver by default; tests can inject MemorySaver.
        # Own the sqlite connection so we can close it cleanly at exit.
        self._owns_connection = checkpointer is None
        if checkpointer is None:
            self._conn = _open_checkpoint_db(STATE_DB_PATH)
            self.memory = SqliteSaver(self._conn)
            atexit.register(self._close_connection)
        else:
            self._conn = None
            self.memory = checkpointer

        # Episodic memory: the durable cross-day log that recall() searches.
        # Injectable for tests (point at a tmp DB or :memory:). Only close a
        # store we created — an injected one is the caller's to manage.
        self._owns_memory_store = memory_store is None
        self.memory_store = (
            memory_store if memory_store is not None else MemoryStore(MEMORY_DB_PATH)
        )
        if self._owns_memory_store:
            atexit.register(self.memory_store.close)

        # Tools: built AFTER memory so forget_session and recall can close
        # over the agent's stores. Closure capture is the simplest binding;
        # ToolManager doesn't need to know about agent-specific tools.
        # clip_service (clip plan 4A) rides through to the save_clip tool;
        # None (the default) means no clip tool is registered.
        self._tm = ToolManager(clip_service=clip_service)
        tm = self._tm
        self.tools = tm.get_tools() + [
            self._make_forget_session_tool(),
            self._make_recall_tool(),
        ]

        llm = llm if llm is not None else make_llm()
        self.llm = llm.bind_tools(self.tools)
        # callbacks: ToolCallLogger emits a start/done/error line per tool call
        # so a hang is visible (a 'start' with no 'done' is the stuck call).
        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "callbacks": [ToolCallLogger()],
        }
        self.graph = self.build_graph()

    def _system_message(self) -> SystemMessage:
        """Fresh system prompt for this turn: static persona + live datetime.

        Rebuilt on every LLM call (string concat, no file IO) so the date the
        model sees is always wall-clock. Baking it in at construction meant a
        robot started before midnight kept yesterday's date until restart.
        """
        return SystemMessage(content=self._persona + "\n\n" + build_datetime_context())

    def prewarm(self) -> None:
        """Run the per-turn prefix through the server once so it's cached.

        Sends exactly what a real turn sends — `self.llm` is the tool-bound
        model, and `_system_message()` is the same persona + datetime block —
        so the server's cached prefix is a true prefix of every later request
        rather than a near-miss that re-processes from scratch. The datetime
        block is date-only, so the prefix stays byte-identical all day (and
        the midnight change rolls the thread anyway).

        `max_tokens=1` because only prompt processing matters here; the
        generated token is thrown away. Blocking — call via
        `start_prewarm()` to overlap it with Whisper/TTS model loading.
        """
        t0 = time.monotonic()
        logger.info("brain prewarm start")
        try:
            self.llm.invoke(
                [self._system_message(), HumanMessage(content="hi")],
                max_tokens=1,
            )
        except Exception:
            # A cold cache is a slow first question, not a broken robot —
            # never let this take the process down.
            logger.exception("brain prewarm failed")
            return
        logger.info("brain prewarm done elapsed=%.1fs", time.monotonic() - t0)

    def start_prewarm(self) -> Optional[threading.Thread]:
        """Fire prewarm() on a daemon thread unless BRAIN_PREWARM is off.

        Daemon so a still-running prewarm can never hold up shutdown. Returns
        the thread (tests join it); None when disabled.
        """
        if not BRAIN_PREWARM:
            logger.info("brain prewarm skipped (BRAIN_PREWARM off)")
            return None
        thread = threading.Thread(
            target=self.prewarm, name="brain-prewarm", daemon=True
        )
        thread.start()
        return thread

    def _roll_thread_if_new_day(self) -> None:
        """Adopt today's thread when the local date changes under a running
        process. Keeps the 'conversations reset overnight' contract true for
        a 24/7 process, not just one that happens to be restarted daily.
        No-op for pinned (explicitly passed) thread_ids."""
        if self._thread_pinned:
            return
        current = daily_thread_id()
        if current != self.thread_id:
            logger.info("daily thread rollover from=%s to=%s", self.thread_id, current)
            self.thread_id = current
            self.config = {
                **self.config,
                "configurable": {"thread_id": current},
            }

    @property
    def music_active(self) -> bool:
        return self._tm.music_active

    def clear_music_active(self) -> None:
        self._tm.music_active = False

    def _close_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.exception("failed to close checkpoint DB connection")
            self._conn = None

    def forget_session(self) -> str:
        """Wipe the current thread's checkpoints. Returns a status string
        the LLM can speak back ("Forgotten.")."""
        try:
            self.memory.delete_thread(self.thread_id)
            logger.info("forgot session thread_id=%s", self.thread_id)
            return "Forgotten."
        except Exception as e:
            logger.exception("forget_session failed")
            return f"Couldn't forget: {e.__class__.__name__}"

    def _make_forget_session_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.forget_session,
            name="forget_session",
            description=(
                "Wipe everything you remember about the current conversation. "
                "Use only when the user explicitly asks you to forget. Examples: "
                "'forget this', 'start fresh', 'clear your memory'."
            ),
        )

    def recall(self, query: str) -> str:
        """Search the durable episodic log for past turns matching `query`.

        Returns a short, newline-joined digest the LLM can read and speak from,
        or a plain "nothing found" string. Never raises into the agent loop —
        a memory miss must not break the conversation (failure-UX principle).
        """
        try:
            episodes = self.memory_store.search(query, limit=5)
        except Exception as e:
            logger.exception("recall failed")
            return f"Couldn't search memory: {e.__class__.__name__}"
        if not episodes:
            return "I don't have anything in memory about that."
        return "\n".join(ep.as_recall_line() for ep in episodes)

    def _make_recall_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.recall,
            name="recall",
            description=(
                "Search your memory of past conversations (across days) for "
                "things the user told you or that you discussed before. Use this "
                "ONLY when the user refers to the past or to personal context — "
                "'what did we talk about', 'do you remember', 'last time', 'my "
                "usual', 'you said'. Do NOT use it for simple commands like "
                "playing music or setting a reminder. The query should be the "
                "topic or keywords to look for."
            ),
        )

    def append_episode(self, user_text: str, assistant_text: str) -> str:
        """Record one completed turn to the durable episodic log.

        Best-effort: a write failure is logged but never propagated — losing
        one episode must not break the live loop. Returns the new episode id
        ("" if skipped/failed).
        """
        try:
            return self.memory_store.append(user_text, assistant_text, self.thread_id)
        except Exception:
            logger.exception("append_episode failed")
            return ""

    def stream(self, input_text: str):
        """Yield assistant token strings as the LLM produces them.

        Tool-call messages and non-text content parts are skipped so the
        generator is safe to feed straight into TextToSpeech.speak().

        On completion, the full turn is written to the durable episodic log.
        The write happens after the last token is yielded — the user has
        already heard the whole response, so it never adds perceived latency.
        Brain-side on purpose: episodic memory lives with the Brain, so the
        Edge/transport seam stays clean for the Phase 8 Pi/Mac split.
        """
        self._roll_thread_if_new_day()
        # Record the turn before the model runs. Tools that need proof of a
        # real human turn (the DoorDash order gate) read this, not the model's
        # claims about what was said.
        self._tm.note_user_turn(input_text)
        input_message = HumanMessage(content=input_text)
        parts: list[str] = []
        # stream_mode="updates", NOT "messages". Token-level streaming emits
        # text straight from the model as it is generated, which means it is
        # already in the speaker before the assistant node can judge it — a
        # fabricated "[calling query_entries] you ate 480 calories" got spoken
        # in full on 2026-08-26 while the log said "refusing to speak it", and
        # the corrective retry then spoke its version too. Emitting the node's
        # returned message instead costs the token-by-token head start (~400ms
        # on qwen2.5:14b) and buys the guarantee that nothing unvalidated is
        # ever voiced, and that a retry replaces the first answer rather than
        # appending to it.
        for update in self.graph.stream(
            {"messages": [input_message]},
            self.config,
            stream_mode="updates",
        ):
            for node, payload in update.items():
                if node != "assistant" or not payload:
                    continue
                messages = payload.get("messages") or []
                if not messages:
                    continue
                text = _content_to_text(getattr(messages[-1], "content", None))
                if text:
                    parts.append(text)
                    yield text
        self.append_episode(input_text, "".join(parts))

    def run(self, input_text: str) -> str:
        """Non-streaming convenience wrapper: returns the full response string."""
        return "".join(self.stream(input_text))

    def build_graph(self):
        def assistant(state: MessagesState):
            # Only the last MAX_HISTORY_MESSAGES go to the model; the full
            # thread is still returned below and checkpointed.
            history = _trim_history(state["messages"], MAX_HISTORY_MESSAGES)
            # Time the model round-trip so a hang here (e.g. a slow/unreachable
            # OpenAI call) is distinguishable from a hang inside a tool: you'll
            # see 'llm start' with no 'llm done'. messages=sent/total shows the
            # trim at work.
            t0 = time.monotonic()
            logger.info(
                "llm start messages=%d/%d", len(history), len(state["messages"])
            )
            response = self.llm.invoke([self._system_message()] + history)

            # Catch a reply that only *describes* doing the work. One corrective
            # retry usually lands a real call; a second failure is answered
            # honestly rather than spoken as if it succeeded.
            reason = _fabrication_reason(response, history)
            if reason is not None:
                logger.warning("fabricated tool call (%s); retrying", reason)
                retried = None
                try:
                    retried = self.llm.invoke(
                        [self._system_message()]
                        + history
                        + [SystemMessage(content=_FABRICATION_NUDGE)],
                        timeout=FABRICATION_RETRY_TIMEOUT,
                    )
                except Exception:
                    # A retry that hangs is worse than no retry: the user is
                    # waiting on an answer we already know was wrong.
                    logger.warning("corrective retry failed; answering honestly")

                if retried is not None and _fabrication_reason(retried, history) is None:
                    response = retried
                else:
                    logger.error(
                        "fabricated tool call again (%s); refusing to speak it",
                        reason,
                    )
                    response.content = _fallback_for(reason)
                    response.tool_calls = []

            # The model sometimes emits the same tool call twice in one turn;
            # for log_entry that double-writes the user's data. Drop exact dupes
            # before ToolNode runs them.
            response = _dedupe_tool_calls(response)
            # A guessed entry_id destroys a real row, so this runs before the
            # tools do, not as a check afterwards.
            response = _drop_unverified_destructive_calls(response, history)
            n_calls = len(getattr(response, "tool_calls", None) or [])
            logger.info(
                "llm done  elapsed=%.2fs tool_calls=%d", time.monotonic() - t0, n_calls
            )
            return {"messages": state["messages"] + [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        return builder.compile(checkpointer=self.memory)


def _content_to_text(content) -> str:
    # LangChain message content can be a string OR a list of parts (dicts/strings)
    # for tool-using/multimodal models. Collapsing it here avoids the
    # `msg.type + ":" + msg.content` TypeError when content is a list.
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get("text")
                if t:
                    parts.append(t)
        return "".join(parts)
    return str(content)
