from spotify_client import SpotifyClient
from langchain_core.tools import StructuredTool, BaseTool
import subprocess

class SpotifyTools:
    def __init__(self, spotify_client: SpotifyClient):
        self.spotify_client = spotify_client
    def create_play_song_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.spotify_client.play_song,
            name="play_song",
            description="""
            This tool plays a song by sending a query string to the Spotify API.
            It accepts one argument: a formatted search string built from available fields — track, artist, album, year, and ISRC.
            Here are some artists to remember. If the input is similar to these, replace the artist with these ones.
            List:
                NewJeans
                Yorushika
                Ai Tomioka
                LiSA
                ReoNa
                Riria
                Query construction rules:
                Include all provided fields in the query.
                Field format: <field>:<value> (e.g., track:Ferris Wheel).
                Omit any field if its information is not provided.
            Examples:
                Input: "Play Dido by NewGenes" → Query: track:"Ditto" artist:"NewJeans"
                Input: "Play Dido by NewGames" → Query: track:"Ditto" artist:"NewJeans"
                Input: "Play Dear Sunrise" → Query: track:"Dear sunrise"
                Input: "Play Ferris Wheel from the album Summer Nights" → Query: track:"Ferris Wheel" album:"Summer Nights"
                Input: "Play Ferris Wheel by QWER from the album Summer Nights" → Query: track:"Ferris Wheel" artist:"QWER" album:"Summer Nights"
                Input: "Play any song by Reona" → Query: artist:"ReoNa"
                Input: "Play track with ISRC code KRA382401950" → Query: "isrc:KRA382401950"
        """
        )
    def create_get_my_playlists_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.spotify_client.get_my_playlists,
            name="get_my_playlists",
            description="Use this tool to refresh and retrieve the playlists saved on your Spotify client. This is to be used when the user wants to refresh the save of the playlists."
        )
    def create_play_playlist_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.spotify_client.play_playlist,
            name="play_playlist",
            description="Use this tool to play a specific playlist from your Spotify client. This tool wants the name of the playlist as the input, and it will play the playlist from Spotify."
        )
    def create_shuffle_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.spotify_client.shuffle,
            name='shuffle',
            description='Turns on or off Shuffling for the current playback on spotify. The function wants true to turn on shuffling, and false to turn off shuffling.'
        )
