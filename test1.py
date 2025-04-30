from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool

import test2


llm = ChatOpenAI()

agent = initialize_agent(llm = llm,
                         verbose = True,
                         agent = AgentType.OPENAI_FUNCTIONS,
                         tools = [
                             StructuredTool.from_function(
                                 func=test2.play_song,
                                 name="play_song",
                                 description="""
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
                                """
                             )
                         ],
                         handle_parsing_errors=True,
                         )

prompt = "Play Ferris Wheel by QWER for me."

agent.invoke(prompt)