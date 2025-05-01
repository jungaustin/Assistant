from spotify_client import SpotifyClient
from tools.spotify_tools import SpotifyTools

class ToolManager:
    def __init__(self):
        self.spotify_client = SpotifyClient()
        self.spotify_tools = SpotifyTools(self.spotify_client)
        self.tools = [
            self.spotify_tools.create_play_song_tool(),
        ]
        
    def get_tools(self):
        return self.tools
