"""WebSocket transport: the Phase 8 Pi/Mac split.

Two halves of one wire protocol, kept in a single module so the format can't
drift:

- ``BrainServer`` (Mac side) wraps a Brain and serves it over a WebSocket.
  Internally it reuses ``InProcessTransport`` so the blocking Agent generator
  runs in a thread exactly the way the in-process path does.
- ``WebSocketTransport`` (Edge side — eventually the Pi) satisfies the same
  ``Transport`` contract as ``InProcessTransport``: ``respond(utterance)``
  yields token strings, ``music_active``/``clear_music_active`` carry the
  one per-turn flag the Edge reads after speaking.

Wire protocol (JSON text frames, pydantic models from core/events.py):

    Edge → Brain:  TranscriptReady{text}
    Brain → Edge:  BrainToken{text} * N, then exactly one terminal event:
                   BrainDone{music_active} on success, Error{message} on a
                   brain-side failure.

One turn at a time per connection; the server also serializes turns across
connections with a lock, because the Agent is not safe to stream concurrently.

Liveness is the websockets library's built-in ping/pong (ping_interval) — a
dead link surfaces as ConnectionClosed rather than a silent hang, so the
custom Heartbeat events aren't needed on this seam.

Failure UX (plan §1.6, "never silent"): the Edge-side transport never raises
into the voice loop. If the brain is unreachable, or the link drops mid-turn,
``respond`` yields a short spoken phrase instead, logs the details, and
reconnects with exponential backoff on the next attempt.

Auth: shared-secret bearer token (TRANSPORT_TOKEN in .env on both machines),
checked during the HTTP handshake so an unauthenticated client never gets a
socket at all.
"""

from __future__ import annotations

import asyncio
import http
from typing import AsyncIterator, Optional

from pydantic import TypeAdapter, ValidationError
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus, WebSocketException

from robot.core.events import BrainDone, BrainToken, Error, Event, TranscriptReady
from robot.core.logging import get_logger
from robot.transport.inproc import Brain, InProcessTransport

log = get_logger(__name__)

_EVENT = TypeAdapter(Event)

# Spoken instead of silence when the brain can't answer (plan §1.6). Kept
# short and literal — the TTS voice is the error channel until the LED exists.
UNREACHABLE_PHRASE = "Sorry — I can't reach my brain right now."
LOST_MIDTURN_PHRASE = " — sorry, I lost my train of thought. Ask me again?"
BRAIN_ERROR_PHRASE = "Something went wrong in my head. Try that again?"


class WebSocketTransport:
    """Edge-side client. Same shape as InProcessTransport, network underneath.

    The connection is opened lazily on the first respond() and kept alive
    between turns. Connect failures back off exponentially (base * 2^n up to
    `backoff_max`) across `connect_attempts` tries per turn; if the link dies
    mid-turn before any token arrived, the utterance is re-sent once on a
    fresh connection (safe: the brain never saw it stream anything back).
    """

    def __init__(
        self,
        url: str,
        token: str,
        connect_attempts: int = 3,
        backoff_base: float = 0.5,
        backoff_max: float = 8.0,
    ):
        self.url = url
        self.token = token
        self.connect_attempts = connect_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._ws = None
        self._music_active = False

    # -- Transport contract ------------------------------------------------

    @property
    def music_active(self) -> bool:
        return self._music_active

    def clear_music_active(self) -> None:
        self._music_active = False

    async def respond(self, utterance: str) -> AsyncIterator[str]:
        # Two passes max: a second pass only happens when the link died
        # before the first token, where a resend can't duplicate anything.
        for attempt in (1, 2):
            ws = await self._ensure_connected()
            if ws is None:
                yield UNREACHABLE_PHRASE
                return
            tokens_seen = False
            try:
                await ws.send(
                    TranscriptReady(source="edge", text=utterance).model_dump_json()
                )
                async for raw in ws:
                    event = _EVENT.validate_json(raw)
                    if isinstance(event, BrainToken):
                        tokens_seen = True
                        yield event.text
                    elif isinstance(event, BrainDone):
                        self._music_active = event.music_active
                        return
                    elif isinstance(event, Error):
                        log.error("brain_error_event", message=event.message)
                        yield BRAIN_ERROR_PHRASE
                        return
                    else:
                        log.warning("unexpected_event", type=event.type)
                # Server closed cleanly without a terminal event: treat as a
                # drop; fall through to the shared recovery path below.
                raise WebSocketException("stream ended without BrainDone")
            except (WebSocketException, OSError, ValidationError) as e:
                await self._drop()
                log.warning(
                    "transport_turn_failed",
                    attempt=attempt,
                    tokens_seen=tokens_seen,
                    error=repr(e),
                )
                if tokens_seen:
                    # Part of the reply was already spoken — apologize and
                    # stop rather than replaying the turn and stuttering.
                    yield LOST_MIDTURN_PHRASE
                    return
        yield UNREACHABLE_PHRASE

    # -- plumbing ------------------------------------------------------------

    async def _ensure_connected(self):
        if self._ws is not None:
            return self._ws
        delay = self.backoff_base
        for attempt in range(1, self.connect_attempts + 1):
            try:
                self._ws = await connect(
                    self.url,
                    additional_headers={"Authorization": f"Bearer {self.token}"},
                )
                log.info("transport_connected", url=self.url, attempt=attempt)
                return self._ws
            except InvalidStatus as e:
                # The server answered but refused the handshake — almost
                # certainly a TRANSPORT_TOKEN mismatch. Retrying can't help.
                log.error(
                    "transport_rejected",
                    url=self.url,
                    status=e.response.status_code,
                )
                return None
            except (WebSocketException, OSError) as e:
                log.warning(
                    "transport_connect_failed",
                    url=self.url,
                    attempt=attempt,
                    error=repr(e),
                )
                if attempt < self.connect_attempts:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.backoff_max)
        return None

    async def _drop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def aclose(self) -> None:
        await self._drop()


class BrainServer:
    """Mac-side server: one Brain behind a WebSocket.

    Wraps the Brain in InProcessTransport so streaming runs in a worker
    thread and the event loop stays free to answer pings. `_turn_lock`
    serializes turns across connections — the Agent streams one conversation
    at a time.
    """

    def __init__(self, brain: Brain, host: str, port: int, token: str):
        if not token:
            raise ValueError(
                "TRANSPORT_TOKEN is required to serve the brain over a socket. "
                "Set it in .env on both machines "
                '(python -c "import secrets; print(secrets.token_hex(32))").'
            )
        self.host = host
        self.port = port
        self.token = token
        self._inproc = InProcessTransport(brain)
        self._turn_lock = asyncio.Lock()
        self._server = None

    async def serve(self) -> None:
        """Listen until cancelled. For scripts/tests that need the bound port
        (e.g. port=0), use `async with server.start(): ...` instead."""
        async with self._serve_ctx() as server:
            await server.serve_forever()

    def _serve_ctx(self):
        return serve(
            self._handle,
            self.host,
            self.port,
            process_request=self._check_auth,
        )

    def start(self):
        """Context manager form for tests: binds, yields the underlying
        websockets server (which exposes the ephemeral port), unbinds on exit."""
        return self._serve_ctx()

    def _check_auth(self, connection, request):
        if request.headers.get("Authorization") != f"Bearer {self.token}":
            log.warning("brain_server_auth_rejected")
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None

    async def _handle(self, ws) -> None:
        log.info("brain_server_client_connected", peer=str(ws.remote_address))
        async for raw in ws:
            try:
                event = _EVENT.validate_json(raw)
            except ValidationError as e:
                log.warning("brain_server_bad_frame", error=repr(e))
                await ws.send(
                    Error(source="brain", message="unparseable frame").model_dump_json()
                )
                continue
            if not isinstance(event, TranscriptReady):
                log.warning("brain_server_unexpected_event", type=event.type)
                await ws.send(
                    Error(
                        source="brain", message=f"unexpected event {event.type!r}"
                    ).model_dump_json()
                )
                continue
            async with self._turn_lock:
                await self._run_turn(ws, event.text)
        log.info("brain_server_client_disconnected", peer=str(ws.remote_address))

    async def _run_turn(self, ws, utterance: str) -> None:
        log.info("brain_turn_start", chars=len(utterance))
        try:
            async for token in self._inproc.respond(utterance):
                await ws.send(BrainToken(source="brain", text=token).model_dump_json())
        except (WebSocketException, OSError):
            # The client went away mid-turn; nothing to report to it. The
            # Agent generator has already run to whatever point it reached.
            raise
        except Exception:
            # Brain-side failure: report it over the wire instead of killing
            # the server. The Edge speaks BRAIN_ERROR_PHRASE (never silent).
            log.exception("brain_turn_failed")
            await ws.send(
                Error(
                    source="brain", message="brain raised during turn"
                ).model_dump_json()
            )
            return
        music = self._inproc.music_active
        self._inproc.clear_music_active()
        await ws.send(BrainDone(source="brain", music_active=music).model_dump_json())
        log.info("brain_turn_done", music_active=music)


__all__ = [
    "WebSocketTransport",
    "BrainServer",
    "UNREACHABLE_PHRASE",
    "LOST_MIDTURN_PHRASE",
    "BRAIN_ERROR_PHRASE",
]
