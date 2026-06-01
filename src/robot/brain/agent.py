from robot.tools.manager import ToolManager
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from robot.config import load_persona, make_llm
import uuid

class Agent:
    def __init__(self, llm=None, thread_id=None):
        self.tools = ToolManager().get_tools()
        llm = llm if llm is not None else make_llm()
        self.llm = llm.bind_tools(self.tools)
        self.system_message = SystemMessage(content=load_persona())
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        self.graph = self.build_graph()
    
    def stream(self, input_text):
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

    def run(self, input_text):
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
