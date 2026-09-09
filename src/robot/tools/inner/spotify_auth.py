"""Runtime Spotify re-authorization — pops the login in the user's browser.

Spotify expires refresh tokens after six months (policy effective
2026-07-20). Before this module the only fix was for the user to drop to a
terminal and run `setup/spotify_oauth_bootstrap.py` by hand, which is a
terrible thing to ask of someone talking to a voice robot. This runs the
same authorization-code exchange from inside the running process: open the
browser, wait for the one click only a human can make, write the new refresh
token back to `.env`, and hand the fresh tokens to the live SpotifyClient so
the turn in progress can just continue.

Why stdlib http.server and not Flask: the bootstrap script uses Flask, but
Flask is a setup-only dependency and is not installed in the runtime. The
callback needs exactly one route, so BaseHTTPRequestHandler is enough.

Why a fixed port: Spotify only redirects to a URI registered on the app's
dashboard, so we can't bind "any free port" the way the Google Calendar flow
does. The port has to stay the one the bootstrap script registered.
"""

from __future__ import annotations

import logging
import secrets
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

logger = logging.getLogger(__name__)

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"

# Must match a Redirect URI registered on the Spotify app dashboard — same
# one setup/spotify_oauth_bootstrap.py uses. Override with SPOTIFY_REDIRECT_URI
# only if you also add the new URI in the dashboard.
DEFAULT_REDIRECT_URI = "http://127.0.0.1:5000/callback"

SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "playlist-read-private "
    "playlist-read-collaborative"
)

# How long the callback server stays up after the browser opens. Well past any
# realistic "click Agree" delay; its job is to make sure an abandoned flow
# eventually frees the port instead of holding it for the process lifetime.
FLOW_TTL_SECONDS = 300.0

_SUCCESS_PAGE = (
    "<h2>Spotify reconnected.</h2>"
    "<p>You can close this tab — the robot already has the new token.</p>"
)


class _CallbackHandler(BaseHTTPRequestHandler):
    """Handles the single GET that Spotify redirects the browser to."""

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        flow: SpotifyAuthFlow = self.server.flow  # type: ignore[attr-defined]
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path != flow.callback_path:
            # Browsers ask for /favicon.ico off their own bat; answering 404
            # without touching the flow keeps that from ending it.
            self._reply(404, "<p>Not found.</p>")
            return

        params = urllib.parse.parse_qs(parsed.query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]
        error = (params.get("error") or [""])[0]

        # Compare with compare_digest: this port is open to anything on the
        # loopback interface, so the state check is the only thing stopping
        # another local process from feeding us a code.
        if not secrets.compare_digest(state, flow.state):
            self._finish(flow, "the login response didn't match this request",
                         "<p>That login didn't match the request. Ask again.</p>")
            return

        if error:
            self._finish(flow, f"Spotify returned '{error}'",
                         f"<p>Spotify said: {error}. You can close this tab.</p>")
            return

        if not code:
            self._finish(flow, "Spotify sent no authorization code",
                         "<p>No authorization code came back. Ask again.</p>")
            return

        try:
            token_info = flow.exchange_code(code)
        except Exception as exc:  # noqa: BLE001 - surfaced as flow.error
            logger.warning("spotify token exchange failed: %s", exc)
            self._finish(flow, str(exc),
                         "<p>Couldn't finish the token exchange. Ask again.</p>")
            return

        self._reply(200, _SUCCESS_PAGE)
        flow.succeed(token_info)

    def _finish(self, flow: "SpotifyAuthFlow", error: str, body: str) -> None:
        self._reply(400, body)
        flow.fail(error)

    def _reply(self, status: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args, **kwargs) -> None:
        """Silence the default stderr access log — it would interleave with
        the robot's structlog output mid-conversation."""


class SpotifyAuthFlow:
    """One browser login attempt. Start it, then wait on `done`.

    Lives entirely on background threads so the caller decides how long to
    block; a tool call that gives up waiting leaves the flow running, and the
    token still lands on the live client whenever the user gets around to
    clicking.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        on_token,
        ttl_seconds: float = FLOW_TTL_SECONDS,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.on_token = on_token
        self.ttl_seconds = ttl_seconds

        self.state = secrets.token_urlsafe(16)
        self.done = threading.Event()
        self.error: str | None = None
        self.browser_opened = False

        parsed = urllib.parse.urlparse(redirect_uri)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.callback_path = parsed.path or "/"

        self._server: HTTPServer | None = None

    # -- state ----------------------------------------------------------

    @property
    def succeeded(self) -> bool:
        return self.done.is_set() and self.error is None

    @property
    def pending(self) -> bool:
        """Still waiting on the human. A pending flow is reused rather than
        restarted, so repeated asks don't stack up browser tabs or fight over
        the callback port."""
        return self._server is not None and not self.done.is_set()

    # -- lifecycle ------------------------------------------------------

    def authorize_url(self) -> str:
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": SCOPES,
            "redirect_uri": self.redirect_uri,
            "state": self.state,
            # show_dialog: this only ever runs because the old grant died, so
            # skip Spotify's silent re-approve and let the user see what they
            # are re-granting.
            "show_dialog": "true",
        }
        return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    def start(self) -> None:
        """Bind the callback port and open the browser. Raises OSError if the
        port is taken (AirPlay Receiver squats on 5000 on some Macs)."""
        server = HTTPServer((self.host, self.port), _CallbackHandler)
        server.flow = self  # type: ignore[attr-defined]
        self._server = server

        threading.Thread(
            target=server.serve_forever,
            name="spotify-oauth-callback",
            daemon=True,
        ).start()
        # Shutting the server down from inside a handler would deadlock
        # (the handler runs on the serve_forever thread), so a closer thread
        # waits on the event and tears down from the outside. It also enforces
        # the TTL, which is what frees the port on an abandoned flow.
        threading.Thread(
            target=self._close_when_finished,
            name="spotify-oauth-closer",
            daemon=True,
        ).start()

        url = self.authorize_url()
        logger.info("spotify reauth: opening browser (%s)", self.redirect_uri)
        try:
            self.browser_opened = webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 - headless host, no browser
            logger.warning("spotify reauth: could not open browser: %s", exc)
            self.browser_opened = False
        if not self.browser_opened:
            # No browser on this machine (a Pi-shaped Edge, say). The flow is
            # still live, so logging the URL lets the user finish it from a
            # phone or another machine on the LAN.
            logger.warning("spotify reauth: open this URL manually: %s", url)

    def exchange_code(self, code: str) -> dict:
        response = requests.post(
            TOKEN_URL,
            data={
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        token_info = response.json()
        if response.status_code != 200:
            raise RuntimeError(
                f"token exchange failed ({response.status_code}): "
                f"{token_info.get('error_description') or token_info}"
            )
        if "refresh_token" not in token_info:
            raise RuntimeError("Spotify returned no refresh token")
        return token_info

    def succeed(self, token_info: dict) -> None:
        try:
            self.on_token(token_info)
        except Exception as exc:  # noqa: BLE001 - don't strand the waiter
            logger.warning("spotify reauth: applying token failed: %s", exc)
            self.error = str(exc)
        self.done.set()

    def fail(self, error: str) -> None:
        logger.warning("spotify reauth failed: %s", error)
        self.error = error
        self.done.set()

    def wait(self, timeout: float) -> bool:
        """Block up to `timeout` seconds for the user's click."""
        return self.done.wait(timeout)

    def _close_when_finished(self) -> None:
        if not self.done.wait(self.ttl_seconds):
            self.error = "timed out waiting for the Spotify login"
            self.done.set()
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()


class SpotifyAuthUnavailable(Exception):
    """The browser login can't be started on this machine right now.

    Raised for setup problems the user has to fix themselves — missing
    CLIENT_ID/CLIENT_SECRET, or the callback port already in use — as opposed
    to a flow that started fine and is merely waiting on a click.
    """
