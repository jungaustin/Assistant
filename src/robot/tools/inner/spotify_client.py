import requests
import urllib.parse
import functools
import logging
import threading

from datetime import datetime, timedelta
from rapidfuzz import process

from robot.tools.inner.spotify_auth import (
    DEFAULT_REDIRECT_URI,
    SpotifyAuthFlow,
    SpotifyAuthUnavailable,
)
# from flask import Flask, redirect, request, jsonify, session

import os
from dotenv import load_dotenv, set_key, find_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class SpotifyReauthRequired(Exception):
    """The stored Spotify refresh token is no longer valid.

    Spotify refresh tokens expire after six months (policy effective
    2026-07-20). An expired or revoked token makes the token endpoint return
    {"error": "invalid_grant"}. Per Spotify's guidance the only fix is to
    discard the dead token (no retry) and run the OAuth bootstrap again to
    mint a fresh one — see setup/spotify_oauth_bootstrap.py.
    """


# Fallback wording for when the browser flow can't run at all (no client
# credentials, callback port taken, or SPOTIFY_AUTO_REAUTH turned off).
_REAUTH_MESSAGE = (
    "Spotify needs to be re-authorized — the saved login has expired. "
    "Run the Spotify setup again (python setup/spotify_oauth_bootstrap.py), "
    "log in once, and a new token will be saved automatically."
)

_BROWSER_OPEN_MESSAGE = (
    "Spotify's login expired, so I opened the Spotify login page in your "
    "browser. Approve it there and then ask me again."
)

_NO_BROWSER_MESSAGE = (
    "Spotify's login expired and I couldn't open a browser on this machine. "
    "The login link is in the log — open it, approve it, and ask me again."
)

_REAUTH_FAILED_MESSAGE = (
    "That Spotify login didn't go through. Ask me to reconnect Spotify and "
    "I'll open it again."
)

# Seconds a tool call blocks waiting for the user to click "Agree". The flow
# outlives this — the wait only decides whether *this* turn can still finish
# what the user actually asked for, or has to answer and let them ask again.
_AUTH_WAIT_SECONDS = float(os.getenv("SPOTIFY_AUTH_WAIT_SECONDS", "60"))

_FALSEY = {"false", "0", "no", "off"}


def _auto_reauth_enabled() -> bool:
    """Set SPOTIFY_AUTO_REAUTH=false to go back to the old behaviour of just
    telling the user to run the bootstrap script by hand."""
    return os.getenv("SPOTIFY_AUTO_REAUTH", "true").strip().lower() not in _FALSEY


def _reauth_guard(method):
    """Recover from an expired Spotify login without ending the turn.

    Tool methods are handed straight to the LLM, so an unhandled exception
    would surface as a stack trace. On expiry we open the Spotify login in
    the user's browser and wait briefly for the one click only a human can
    make; if they take it, the call they originally asked for is retried and
    they never have to repeat themselves. If they don't, we fall back to a
    calm, speakable instruction.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except SpotifyReauthRequired:
            pass

        reconnected, message = self.await_reauth()
        if not reconnected:
            return message
        try:
            return method(self, *args, **kwargs)
        except SpotifyReauthRequired:
            # Fresh token, still refused: not something another browser
            # round-trip will fix.
            return _REAUTH_MESSAGE
    return wrapper


class SpotifyClient:
    def __init__(self):
        self.CLIENT_ID = os.getenv("CLIENT_ID")
        self.CLIENT_SECRET = os.getenv("CLIENT_SECRET")
        self.REFRESH_TOKEN =  os.getenv("REFRESH_TOKEN")
        self.access_token = None
        self.device_id = None
        self.expires_at = None
        self.API_BASE_URL = 'https://api.spotify.com/v1'
        self.TOKEN_URL = 'https://accounts.spotify.com/api/token'
        self.playlists = None
        # Must match a Redirect URI registered on the Spotify app dashboard.
        self.REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", DEFAULT_REDIRECT_URI)
        # At most one browser login in flight. Two Spotify tool calls in the
        # same turn ("play my smoothie playlist shuffled") both hit the guard,
        # so they have to share a flow rather than race for the callback port.
        self._auth_flow: SpotifyAuthFlow | None = None
        self._auth_lock = threading.Lock()
    
    def is_token_valid(self):
        return self.access_token is not None and datetime.now().timestamp() < self.expires_at
    
    def get_headers(self):
        return {
            'Authorization': f"Bearer {self.access_token}"
        }
    
    def refresh_token(self):
        if self.is_token_valid():
            return

        if not self.REFRESH_TOKEN:
            raise SpotifyReauthRequired("No Spotify refresh token is stored.")

        req_body = {
            'grant_type' : 'refresh_token',
            'refresh_token' : self.REFRESH_TOKEN,
            'client_id' : self.CLIENT_ID,
            'client_secret' : self.CLIENT_SECRET
        }
        response = requests.post(self.TOKEN_URL, data = req_body)
        new_token_info = response.json()

        if response.status_code != 200:
            # A refresh token that has expired (six-month policy, effective
            # 2026-07-20) or been revoked comes back as invalid_grant. Discard
            # it and signal re-auth — do NOT retry, per Spotify's guidance.
            if new_token_info.get('error') == 'invalid_grant':
                self.REFRESH_TOKEN = None
                self.access_token = None
                self.expires_at = None
                raise SpotifyReauthRequired(
                    new_token_info.get('error_description', 'invalid_grant')
                )
            raise RuntimeError(
                f"Spotify token refresh failed "
                f"({response.status_code}): {new_token_info}"
            )

        self.access_token = new_token_info['access_token']
        self.expires_at = datetime.now().timestamp() + new_token_info['expires_in']

        # Spotify may rotate the refresh token on refresh; the old one then
        # stops working. Persist any replacement so the next process start
        # doesn't wrongly think it needs re-auth.
        rotated = new_token_info.get('refresh_token')
        if rotated and rotated != self.REFRESH_TOKEN:
            self.REFRESH_TOKEN = rotated
            self._persist_refresh_token(rotated)

    def _persist_refresh_token(self, token: str) -> None:
        # Best-effort write-back to .env. A failure here must not break
        # playback — we still hold a valid access token in memory.
        try:
            dotenv_path = find_dotenv(usecwd=True)
            if dotenv_path:
                set_key(dotenv_path, 'REFRESH_TOKEN', token)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Browser re-authorization (six-month refresh-token expiry)
    # ------------------------------------------------------------------

    def _apply_token_info(self, token_info: dict) -> None:
        """Adopt tokens from a completed browser login, mid-process.

        Called from the OAuth callback thread, which is why it does the whole
        job: without this the new token would only reach a *restarted* robot,
        and the point of the browser flow is that the turn in progress can
        carry on.
        """
        self.access_token = token_info["access_token"]
        self.expires_at = datetime.now().timestamp() + token_info["expires_in"]
        refresh = token_info.get("refresh_token")
        if refresh:
            self.REFRESH_TOKEN = refresh
            self._persist_refresh_token(refresh)
        # The new grant may be for a different Spotify account, and both of
        # these were cached under the old one.
        self.device_id = None
        self.playlists = None
        logger.info("spotify reauthorized; new refresh token saved")

    def begin_reauth(self) -> SpotifyAuthFlow:
        """Start (or reuse) the browser login. Returns without waiting.

        Raises SpotifyAuthUnavailable when the flow can't run here at all.
        """
        with self._auth_lock:
            flow = self._auth_flow
            if flow is not None and flow.pending:
                # Already waiting on the user — reusing it keeps a second tool
                # call in the same turn from stacking up browser tabs.
                return flow

            if not (self.CLIENT_ID and self.CLIENT_SECRET):
                raise SpotifyAuthUnavailable(
                    "CLIENT_ID/CLIENT_SECRET are missing from .env"
                )

            flow = SpotifyAuthFlow(
                client_id=self.CLIENT_ID,
                client_secret=self.CLIENT_SECRET,
                redirect_uri=self.REDIRECT_URI,
                on_token=self._apply_token_info,
            )
            try:
                flow.start()
            except OSError as exc:
                # Port already in use — most often macOS AirPlay Receiver
                # squatting on 5000, or a bootstrap script still running.
                raise SpotifyAuthUnavailable(
                    f"couldn't listen on {self.REDIRECT_URI}: {exc}"
                ) from exc

            self._auth_flow = flow
            return flow

    def await_reauth(self, wait_seconds: float | None = None) -> tuple[bool, str]:
        """Open the Spotify login and wait for the click.

        Returns (reconnected, message_for_the_user). On success the message is
        empty — the caller has something better to say.
        """
        if not _auto_reauth_enabled():
            return False, _REAUTH_MESSAGE

        try:
            flow = self.begin_reauth()
        except SpotifyAuthUnavailable as exc:
            logger.warning("spotify reauth unavailable: %s", exc)
            return False, _REAUTH_MESSAGE

        flow.wait(_AUTH_WAIT_SECONDS if wait_seconds is None else wait_seconds)

        if flow.succeeded:
            return True, ""
        if flow.done.is_set():
            return False, _REAUTH_FAILED_MESSAGE
        if not flow.browser_opened:
            return False, _NO_BROWSER_MESSAGE
        return False, _BROWSER_OPEN_MESSAGE

    def reauthorize(self) -> str:
        """Tool entry point: reconnect Spotify by asking the user to log in."""
        reconnected, message = self.await_reauth()
        if reconnected:
            return "Spotify's reconnected."
        return message

    def get_device_id(self):
        if not self.is_token_valid():
            self.refresh_token()
        
        response = requests.get(self.API_BASE_URL + '/me/player/devices', headers=self.get_headers())
        devices = response.json()
        for device in devices['devices']:
            #I can change this to whichever device I want later (currently using mac)
            if 'MacBook' in device['name']:
                self.device_id = device['id']
                break

    @_reauth_guard
    def play_song(self, query : str) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()

        search = requests.get(
            self.API_BASE_URL + '/search',
            headers=self.get_headers(),
            params={
                'q': query,
                'type': 'track',
                'limit': 1
            }
        )
        search_json = search.json()
        items = search_json.get('tracks', {}).get('items', [])
        if items and 'uri' in items[0]:
            track_uri = items[0]['uri']
        else:
            return "No matching track found."

        data = {
            'uris' : [track_uri],
            'device_id' : self.device_id
        }
        response = requests.put(self.API_BASE_URL + f'/me/player/play',
                                headers = self.get_headers(),
                                json = data
                                )
        if response.status_code == 204:
            return "Playing."
        else:
            return f"Failed to play song. Status code: {response.status_code}"
    
    @_reauth_guard
    def get_my_playlists(self):
        if not self.is_token_valid():
            self.refresh_token()

        if self.device_id is None:
            self.get_device_id()

        offset = 0
        params={
            'limit': 50,
            'offset': offset,
        }
        
        my_playlists = {}
        
        while(True):
            search = requests.get(
                self.API_BASE_URL + '/me/playlists',
                headers=self.get_headers(),
                params=params
            )
            search_res = search.json()
            items = search_res.get('items', {})
            if search.status_code != 200:
                return (f"Error fetching playlists: {search.status_code} - {search.text}")
            for item in items:
                my_playlists[item['name']] = item['uri']
            if(len(items) < 50):
                break
            offset += 50
        self.playlists = my_playlists
        # Return the actual names (was `self.playlists.keys` — a bound method
        # whose repr the LLM had to parse, which produced longer/odder turns).
        return list(self.playlists.keys())

    def get_best_playlist_match(self, user_input: str, threshold=70) -> str | None:
        result = process.extractOne(user_input, self.playlists.keys(), score_cutoff=threshold)
        return result[0] if result else None
    
    @_reauth_guard
    def play_playlist(self, input : str) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()

        if(self.playlists == None):
            _ = self.get_my_playlists()
        
        playlist_name = None
        
        for name in self.playlists.keys():
            if input.strip().lower() == name.strip().lower():
                playlist_name = name
                break

        if playlist_name is None:
            playlist_name = self.get_best_playlist_match(input)

        if playlist_name is None:
            return f"No matching playlist found for '{input}'."
        
        playlist_uri = self.playlists[playlist_name]
        data = {
            'context_uri': playlist_uri,
            'position_ms': 0,
        }
        response = requests.put(
            self.API_BASE_URL + '/me/player/play',
            params={
                'device_id': self.device_id
                },
            headers=self.get_headers(),
            json=data
        )
        if response.status_code == 204:
            return f"Playing '{playlist_name}'."
        else:
            return f"Failed to play playlist. Status code: {response.status_code}"

    @_reauth_guard
    def shuffle(self, state : bool) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()
            
        response = requests.put(
            self.API_BASE_URL + '/me/player/shuffle', 
            headers=self.get_headers(),
            params={'state': state, 'device_id': self.device_id}
        )
        if response.status_code > 199 and response.status_code < 300:
            return f"It worked."
        else:
            return f"Failed to shuffle"
    
    @_reauth_guard
    def pause_playback(self) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()
        
        response = requests.put(
            self.API_BASE_URL + '/me/player/pause',
            headers=self.get_headers(),
            params={
                'device_id': self.device_id
            }
        )
        
        if 200 <= response.status_code < 300:
            return "Paused"
        else:
            return f"An Error has Occured: Status Code {response.status_code}"
        
    @_reauth_guard
    def play_playback(self) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()
        
        response = requests.put(
            self.API_BASE_URL + '/me/player/play',
            headers=self.get_headers(),
            params={
                'device_id': self.device_id
            }
        )
        
        if 200 <= response.status_code < 300:
            return "Started playback"
        else:
            return f"An error has occured: Status Code {response.status_code}"


# def queue_song():
#     if 'access_token' not in session:
#         return redirect('/login')
    
#     if datetime.now().timestamp() > session['expires_at']:
#         return redirect('/refresh-token?next=/queue-song')
    
#     if 'device_id' not in session or session['device_id'] == None:
#         return redirect('/device?next=/queue-song')
    
#     headers = {
#         'Authorization': f"Bearer {session['access_token']}"
#     }
#     #change this later with song from voice to text, maybe use a llm to give json of this type
#     search = requests.get(
#         API_BASE_URL + '/search',
#         headers=headers,
#         params={
#             'q': 'track:Ferris Wheel artist:QWER',
#             'type': 'track',
#             'limit': 1
#         }
#     )
#     search_json = search.json()
#     track_uri = search_json['tracks']['items'][0]['uri']
#     response = requests.post(API_BASE_URL + f'/me/player/queue',
#                              headers = headers,
#                              params = {
#                                  'device_id' : session['device_id'],
#                                  'uri' : track_uri
#                                  }
#                              )
#     return redirect('/')
