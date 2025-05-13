from spotify_client import SpotifyClient
from tools.spotify_tools import SpotifyTools
from tools.generic_tools import GenericTools
class ToolManager:
    def __init__(self):
        spotify_client = SpotifyClient()
        self.spotify_tools = SpotifyTools(spotify_client)
        self.generic_tools = GenericTools()
        self.tools = self.initialize_tools()
    def initialize_tools(self):
        all_tools = []
        all_tools.append(self.spotify_tools.create_play_song_tool())
        all_tools.append(self.generic_tools.create_open_app_tool())
        all_tools.append(self.spotify_tools.create_play_playlist_tool())
        all_tools.append(self.spotify_tools.create_get_my_playlists_tool())
        all_tools.append(self.spotify_tools.create_shuffle_tool())
        all_tools.append(self.spotify_tools.create_pause_tool())
        all_tools.append(self.spotify_tools.create_play_tool())
        return all_tools
    def get_tools(self):
        return self.tools
