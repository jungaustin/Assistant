"""LangGraph agent. Streams tokens, dispatches tool calls, persists
conversation memory to sqlite so the process survives restart.
"""

from __future__ import annotations

import atexit
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from robot.config import (
    MEMORY_DB_PATH,
    STATE_DB_PATH,
    daily_thread_id,
    load_persona,
    make_llm,
)
from robot.memory import MemoryStore
from robot.tools.manager import ToolManager

logger = logging.getLogger(__name__)


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
    ):
        self.thread_id = thread_id or daily_thread_id()
        self.system_message = SystemMessage(content=load_persona())

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
        tm = ToolManager()
        self.tools = tm.get_tools() + [
            self._make_forget_session_tool(),
            self._make_recall_tool(),
        ]

        llm = llm if llm is not None else make_llm()
        self.llm = llm.bind_tools(self.tools)
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.graph = self.build_graph()

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
            response = self.llm.invoke([self.system_message] + state["messages"])
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
