from tool_manager import ToolManager
from llm import LLMInteractions
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI

class Agent:
    def __init__(self, llm = None):
        self.tool_manager = ToolManager()
        self.tools = self.tool_manager.get_tools()
        llm = llm or ChatOpenAI()
        self.llm = llm.bind_tools(self.tools)
        self.system_message = SystemMessage(content="""You are Nemo, a friendly and capable AI assistant.
            You help the user with daily tasks, answer questions, and offer thoughtful advice.
            Keep your tone helpful and approachable, and your responses concise and accurate.
            There is no need to repeat the question back to me.
            Try your best to keep your responses short. More information is not necessary unless I ask.
            If I ask you to do a task, do your best to give 0 or 1 word answers. For example, if i ask you to play a song, there is no need for a response unless the song was not able to be played for some reason.

            Here are example questions and answers to guide your responses:

            User: Nemo, how big is an ant?
            Answer: 1.5 mm on average.
            
            User: Play Ditto by NewJeans.
            Answer: 
            
            User: Play Ferris Wheel by QWER.
            Answer: 

            User: Nemo, remind me to take out the trash every Wednesday night.
            Answer: Got it! I'll remind you.

            User: Nemo, I'm feeling burnt out lately. What should I do?
            Answer: It's okay to feel that way. Try to take short breaks during the day, get plenty of rest, and talk to someone you trust.""")
        self.graph = self.build_graph()
    
    def run_llm(self, state: MessagesState):
        updated_messages = [self.system_message] + state["messages"]
        response = self.llm.invoke(updated_messages)
        return {"messages": state["messages"] + [response]}
    
    def run(self, input_text):
        print(input_text)
        input_message = HumanMessage(content=input_text)
        state = {"messages": [input_message]}
        print('here')
        result = self.graph.invoke(state)
        for msg in result["messages"]:
            print(msg.content)
            yield msg.content
        

    def build_graph(self):
        builder = StateGraph(MessagesState)
        builder.add_node("run_llm", self.run_llm)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "run_llm")
        builder.add_conditional_edges(
            "run_llm",
            tools_condition,
            {
                "tools": "tools",
                "__end__": END
            }
        )
        builder.add_edge("tools", "run_llm")
        return builder.compile()
