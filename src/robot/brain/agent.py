"""LangGraph agent. Streams tokens, dispatches tool calls, persists
conversation memory to sqlite so the process survives restart.
"""

from __future__ import annotations

import atexit
import json
import logging
import sqlite3
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
        input_message = HumanMessage(content=input_text)
        parts: list[str] = []
        for chunk, metadata in self.graph.stream(
            {"messages": [input_message]},
            self.config,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") != "assistant":
                continue
            text = _content_to_text(getattr(chunk, "content", None))
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
            # The model sometimes emits the same tool call twice in one turn;
            # for log_entry that double-writes the user's data. Drop exact dupes
            # before ToolNode runs them.
            response = _dedupe_tool_calls(response)
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
