from langchain.tools import StructuredTool

class SpotifyTools:
    def __init__(self, spotify_client):
        self.spotify_client = spotify_client
    def create_play_song_tool(self):
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
