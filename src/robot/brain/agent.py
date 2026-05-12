from robot.tools.manager import ToolManager
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from robot.config import make_llm
import uuid

class Agent:
    def __init__(self, llm=None, thread_id=None):
        self.tools = ToolManager().get_tools()
        llm = llm if llm is not None else make_llm()
        self.llm = llm.bind_tools(self.tools)
        self.system_message = SystemMessage(content="""You are Nemo, a friendly and capable AI assistant.
            You help the user with daily tasks, answer questions, and offer thoughtful advice.
            Keep your tone helpful and approachable, and your responses concise and accurate.
            There is no need to repeat the question back to me.
            Try your best to keep your responses short. More information is not necessary unless I ask.
            If I ask you to do a task that requires a tool, do your best to give 0 or 1 word answers. For example, if i ask you to play a song, there is no need for a response unless the song was not able to be played for some reason.
            Feel free to use multiple tool calls if necessary. For example, if the user asks to shuffle a playlist, you would first play the playlist in quetion with a play playlist tool call, and then use a shuffle tool call.
            Do not ask if I need anything else. If I need anything, I will ask without having you ask me.

            Here are example questions and answers to guide your responses:

            User: Nemo, how big is an ant?
            Answer: 1.5 mm on average.
            
            User: Play Ditto by NewJeans.
            Answer: Playing.
            
            User: Could you play the playlist loop for me shuffled?
            Answer: Alright.
            
            User: Open System For me.
            Answer: Unable to find "System".

            User: Nemo, remind me to take out the trash every Wednesday night.
            Answer: Got it! I'll remind you.
            """)
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
