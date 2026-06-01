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

from robot.config import STATE_DB_PATH, daily_thread_id, load_persona, make_llm
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
    ):
        self.thread_id = thread_id or daily_thread_id()
        self.system_message = SystemMessage(content=load_persona())

        # Memory: SqliteSaver by default; tests can inject MemorySaver.
        # Own the sqlite connection so we can close it cleanly at exit.
        self._owns_connection = checkpointer is None
        if checkpointer is None:
            self._conn = _open_checkpoint_db(STATE_DB_PATH)
            self.memory = SqliteSaver(self._conn)
            atexit.register(self._close_connection)
        else:
            self._conn = None
            self.memory = checkpointer

        # Tools: built AFTER memory so forget_session can call back into
        # the agent's checkpointer. Closure capture is the simplest binding;
        # ToolManager doesn't need to know about agent-specific tools.
        tm = ToolManager()
        self.tools = tm.get_tools() + [self._make_forget_session_tool()]

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

    def stream(self, input_text: str):
        """Yield assistant token strings as the LLM produces them.

        Tool-call messages and non-text content parts are skipped so the
        generator is safe to feed straight into TextToSpeech.speak().
        """
        input_message = HumanMessage(content=input_text)
        for chunk, metadata in self.graph.stream(
            {"messages": [input_message]},
            self.config,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") != "assistant":
                continue
            text = _content_to_text(getattr(chunk, "content", None))
            if text:
                yield text

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
