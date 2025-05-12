from tool_manager import ToolManager
from llm import LLMInteractions
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

class Agent:
    def __init__(self, llm = None):
        self.tools = ToolManager().get_tools()
        llm = llm or ChatOpenAI()
        self.llm = llm.bind_tools(self.tools)
        self.system_message = SystemMessage(content="""You are Nemo, a friendly and capable AI assistant.
            You help the user with daily tasks, answer questions, and offer thoughtful advice.
            Keep your tone helpful and approachable, and your responses concise and accurate.
            There is no need to repeat the question back to me.
            Try your best to keep your responses short. More information is not necessary unless I ask.
            If I ask you to do a task that requires a tool, do your best to give 0 or 1 word answers. For example, if i ask you to play a song, there is no need for a response unless the song was not able to be played for some reason.
            Feel free to use multiple tool calls if necessary. For example, if the user asks to shuffle a playlist, you would first play the playlist in quetion with a play playlist tool call, and then use a shuffle tool call.

            Here are example questions and answers to guide your responses:

            User: Nemo, how big is an ant?
            Answer: 1.5 mm on average.
            
            User: Play Ditto by NewJeans.
            Answer: Playing.
            
            User: Play Ferris Wheel by QWER.
            Answer: Playing.
            
            User: Open System For me.
            Answer: Unable to find "System".

            User: Nemo, remind me to take out the trash every Wednesday night.
            Answer: Got it! I'll remind you.
            """)
        self.memory = MemorySaver()
        self.config = {"configurable" : {"thread_id": "1"}}
        self.graph = self.build_graph()
    
    # def run_llm(self, state: MessagesState):
    #     updated_messages = [self.system_message] + state["messages"]
    #     response = self.llm.invoke(updated_messages)
    #     return {"messages": state["messages"] + [response]}
    
    def run(self, input_text):
        print(input_text)
        input_message = HumanMessage(content=input_text)
        print('here')
        result = self.graph.invoke({"messages": [input_message]}, self.config)
        for msg in result["messages"]:
            print(msg.type + ":" + msg.content)
            # yield msg.content
        return result["messages"][-1].content

    def build_graph(self):
        def assistant(state: MessagesState):
            return {"messages" : [self.llm.invoke([self.system_message] + state["messages"])]}
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            # {
            #     "tools": "tools",
            #     "__end__": END
            # } # I am here considering deleting this or not
        )
        builder.add_edge("tools", "assistant")
        return builder.compile(checkpointer = self.memory)
