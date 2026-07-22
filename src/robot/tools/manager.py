from typing import Callable

from robot.config import TRACKER_DB_PATH
from robot.tools.inner.calendar_tools import CalendarTools
from robot.tools.inner.clip_tools import ClipTools
from robot.tools.inner.discord_tools import DiscordTools
from robot.tools.inner.generic_tools import GenericTools
from robot.tools.inner.google_calendar import GoogleCalendar
from robot.tools.inner.log import LogTools, TrackerDB
from robot.tools.inner.math_tools import MathTools
from robot.tools.inner.spotify_client import SpotifyClient
from robot.tools.inner.spotify_tools import SpotifyTools
from robot.tools.inner.timer_tools import TimerTools
from robot.tools.inner.web_tools import WebTools


class ToolManager:
    def __init__(self, clip_service=None):
        # music_active: set True when playback starts, False when paused.
        # The Edge reads this after each turn to decide whether to open the
        # follow-up listen window (skip it while music is playing so STT
        # doesn't transcribe lyrics as user speech).
        self.music_active: bool = False

        spotify_client = SpotifyClient()
        self.spotify_tools = SpotifyTools(spotify_client)
        self.generic_tools = GenericTools()
        # Calendar client is lazy: the network/disk work happens on first
        # tool call, not at construction. So importing ToolManager doesn't
        # require Google credentials to be set up.
        self.calendar_tools = CalendarTools(GoogleCalendar())
        self.log_tools = LogTools(TrackerDB(TRACKER_DB_PATH))
        self.math_tools = MathTools()
        # Web search client is lazy too: Tavily client builds on first call,
        # so a missing TAVILY_API_KEY doesn't break boot.
        self.web_tools = WebTools()
        self.timer_tools = TimerTools()
        # Discord client is lazy too: no HTTP until a tool call, so a
        # missing DISCORD_BOT_TOKEN doesn't break boot.
        self.discord_tools = DiscordTools()
        # Clip service is constructor-injected (clip plan 4A): the same
        # instance the Edge snapshots/fast-paths through. None = clipping
        # disabled; the save_clip tool is then not registered at all, so a
        # disabled feature costs zero tool-budget tokens per turn.
        self.clip_tools = ClipTools(clip_service)
        self.tools = self.initialize_tools()

    def _starts_music(self, fn: Callable) -> Callable:
        """Wrap a tool function so it sets music_active=True on success."""

        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            self.music_active = True
            return result

        wrapper.__name__ = fn.__name__
        return wrapper

    def _pauses_music(self, fn: Callable) -> Callable:
        """Wrap a tool function so it sets music_active=False on success."""

        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            self.music_active = False
            return result

        wrapper.__name__ = fn.__name__
        return wrapper

    def initialize_tools(self):
        sc = self.spotify_tools.spotify_client
        all_tools = []
        all_tools.append(self.spotify_tools.create_play_song_tool())
        all_tools.append(self.generic_tools.create_open_app_tool())
        all_tools.append(self.spotify_tools.create_play_playlist_tool())
        all_tools.append(self.spotify_tools.create_get_my_playlists_tool())
        all_tools.append(self.spotify_tools.create_shuffle_tool())
        all_tools.append(self.spotify_tools.create_pause_tool())
        all_tools.append(self.spotify_tools.create_play_tool())
        all_tools.append(self.calendar_tools.create_list_calendar_events_tool())
        all_tools.append(self.calendar_tools.create_add_calendar_event_tool())
        all_tools.append(self.calendar_tools.create_delete_calendar_event_tool())
        all_tools.append(self.log_tools.create_log_entry_tool())
        all_tools.append(self.log_tools.create_query_entries_tool())
        all_tools.append(self.log_tools.create_entry_stats_tool())
        all_tools.append(self.math_tools.create_calculate_tool())
        all_tools.append(self.log_tools.create_update_entry_tool())
        all_tools.append(self.log_tools.create_delete_entry_tool())
        all_tools.append(self.log_tools.create_upsert_period_note_tool())
        all_tools.append(self.log_tools.create_get_period_note_tool())
        all_tools.append(self.web_tools.create_web_search_tool())
        all_tools.append(self.timer_tools.create_set_timer_tool())
        all_tools.append(self.timer_tools.create_list_timers_tool())
        all_tools.append(self.timer_tools.create_cancel_timer_tool())
        all_tools.append(self.discord_tools.create_send_discord_message_tool())
        all_tools.append(self.discord_tools.create_catch_up_discord_tool())
        all_tools.append(self.discord_tools.create_mark_discord_read_tool())
        all_tools.append(self.discord_tools.create_list_discord_channels_tool())
        if self.clip_tools.available:
            all_tools.append(self.clip_tools.create_save_clip_tool())

        # Patch the Spotify tools that start or stop playback so music_active
        # stays in sync. We do this after creating the StructuredTools and
        # swap their underlying func rather than recreating the tools so that
        # all LangChain metadata (name, description, schema) is preserved.
        _STARTS = {"play_song", "play_playlist", "play_playback"}
        _PAUSES = {"pause_playback"}
        for tool in all_tools:
            if tool.name in _STARTS:
                tool.func = self._starts_music(tool.func)
            elif tool.name in _PAUSES:
                tool.func = self._pauses_music(tool.func)

        return all_tools

    def get_tools(self):
        return self.tools
