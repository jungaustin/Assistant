from spotify_client import SpotifyClient
from tools.spotify_tools import SpotifyTools
from tools.generic_tools import GenericTools
class ToolManager:
    def __init__(self):
        self.spotify_client = SpotifyClient()
        self.spotify_tools = SpotifyTools(self.spotify_client)
        self.generic_tools = GenericTools()
        self.tools = [
            self.spotify_tools.create_play_song_tool(),
            self.generic_tools.create_open_app_tool(),
            self.spotify_tools.create_play_playlist_tool(),
            self.spotify_tools.create_get_my_playlists_tool(),
            self.spotify_tools.create_shuffle_tool(),
        ]
        
    def get_tools(self):
        return self.tools
