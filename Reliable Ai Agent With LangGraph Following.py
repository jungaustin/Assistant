from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a (int): first int
        b (int): second int
    """
    return a+b
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a (int): first int
        b (int): second int
    """
    return a*b
def divide(a: int, b: int) -> float:
    """Divides a and b.

    Args:
        a (int): first int
        b (int): second int
    """
    return a/b


tools = [add, multiply, divide]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful asssitant tasked with writing performing arithmetic on 2 numbers.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

graph = builder.compile()

input_message = HumanMessage(content="What is 1+2")
state = {"messages": [input_message]}
result = graph.invoke(state)
for msg in result["messages"]:
    print(msg.content)