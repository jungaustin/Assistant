# learning of langgraph agents

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
import test2

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def play_song(query : str) -> str:
    """
    This tool plays a song by sending a query string to the Spotify API.
    It accepts one argument: a formatted search string built from available fields — track, artist, album, year, and ISRC.
    Here are some artists to remember. If the input is similar to these, replace the artist with these ones.
    List:
        NewJeans
        Yorushika
        Ai Tomioka
        Reona
        Riria
        Query construction rules:
        Include all provided fields in the query.
        Field format: <field>:<value> (e.g., track:Ferris Wheel).
        Omit any field if its information is not provided.
    Examples:
        Input: "Play Ferris Wheel by QWER" → Query: track:Ferris Wheel artist:QWER
        Input: "Play Ferris Wheel from the album Summer Nights" → Query: track:Ferris Wheel album:Summer Nights
        Input: "Play Ferris Wheel by QWER from the album Summer Nights" → Query: track:Ferris Wheel artist:QWER album:Summer Nights
        Input: "Play any song by Yorushika" → Query: artist:Yorushika
        Input: "Play track with ISRC code USUM71703861" → Query: isrc:USUM71703861

    Args:
       query (string): the search query the spotify api will use to play a song
"""
    return test2.play_song(query)



tools = [play_song]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful asssitant everyday assistant. The user can tell you to do a variety of tasks, and you choose tools to help with those tasks.")

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

input_message = HumanMessage(content="Play the song Ferris Wheel by QWER")
state = {"messages": [input_message]}
result = graph.invoke(state)
for msg in result["messages"]:
    print(msg.content)