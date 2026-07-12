"""WebSocket transport (Phase 8): client/server pair over real loopback
sockets, with a fake Brain so no models or audio are involved.

Covers the contract main.Edge depends on — token streaming, music_active
hand-off — plus the failure UX guarantees: wrong token, server down, brain
exception, and reconnect-after-drop all end in a spoken phrase, never an
exception into the voice loop and never silence.
"""

from contextlib import asynccontextmanager

from robot.transport.websocket import (
    BRAIN_ERROR_PHRASE,
    UNREACHABLE_PHRASE,
    BrainServer,
    WebSocketTransport,
)

TOKEN = "test-shared-secret"


class FakeBrain:
    """Blocking-generator Brain, like Agent.stream. Optionally raises after
    yielding `fail_after` tokens, and reports music_active like ToolManager."""

    def __init__(self, tokens=("Hello", " there"), music=False, fail_after=None):
        self.tokens = tokens
        self.music_active = music
        self.fail_after = fail_after
        self.inputs = []

    def clear_music_active(self):
        self.music_active = False

    def stream(self, input_text):
        self.inputs.append(input_text)
        for i, tok in enumerate(self.tokens):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("synthetic brain failure")
            yield tok


@asynccontextmanager
async def running_server(brain, token=TOKEN):
    server = BrainServer(brain, "localhost", 0, token)
    async with server.start() as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        yield f"ws://localhost:{port}"


def fast_transport(url, token=TOKEN):
    """Client with sub-second backoff so failure tests don't sleep for real."""
    return WebSocketTransport(url, token, connect_attempts=2, backoff_base=0.01)


async def collect(transport, utterance):
    return [tok async for tok in transport.respond(utterance)]


async def test_round_trip_streams_tokens_in_order():
    brain = FakeBrain(tokens=("one", " two", " three"))
    async with running_server(brain) as url:
        transport = fast_transport(url)
        assert await collect(transport, "count") == ["one", " two", " three"]
        assert brain.inputs == ["count"]
        await transport.aclose()


async def test_music_active_crosses_the_wire_and_clears():
    brain = FakeBrain(music=True)
    async with running_server(brain) as url:
        transport = fast_transport(url)
        await collect(transport, "play something")
        assert transport.music_active is True
        transport.clear_music_active()
        assert transport.music_active is False
        # Server-side flag was cleared after the turn, so a second turn
        # (brain no longer reports music) comes back False.
        await collect(transport, "thanks")
        assert transport.music_active is False
        await transport.aclose()


async def test_multiple_turns_reuse_one_connection():
    brain = FakeBrain(tokens=("ok",))
    async with running_server(brain) as url:
        transport = fast_transport(url)
        await collect(transport, "first")
        ws = transport._ws
        await collect(transport, "second")
        assert transport._ws is ws
        assert brain.inputs == ["first", "second"]
        await transport.aclose()


async def test_wrong_token_speaks_unreachable_not_raises():
    async with running_server(FakeBrain()) as url:
        transport = fast_transport(url, token="wrong-secret")
        assert await collect(transport, "hello") == [UNREACHABLE_PHRASE]


async def test_server_down_speaks_unreachable_after_backoff():
    transport = fast_transport("ws://localhost:1")  # nothing listens here
    assert await collect(transport, "hello") == [UNREACHABLE_PHRASE]


async def test_brain_exception_becomes_spoken_error_phrase():
    brain = FakeBrain(fail_after=0)
    async with running_server(brain) as url:
        transport = fast_transport(url)
        assert await collect(transport, "hello") == [BRAIN_ERROR_PHRASE]
        # The turn failed but the server survived: the next turn still works.
        brain.fail_after = None
        assert await collect(transport, "again") == ["Hello", " there"]
        await transport.aclose()


async def test_partial_stream_then_brain_failure_apologizes_after_tokens():
    brain = FakeBrain(tokens=("Hel", "lo"), fail_after=1)
    async with running_server(brain) as url:
        transport = fast_transport(url)
        assert await collect(transport, "hello") == ["Hel", BRAIN_ERROR_PHRASE]
        await transport.aclose()


async def test_stale_connection_reconnects_and_resends():
    brain = FakeBrain(tokens=("back",))
    async with running_server(brain) as url:
        transport = fast_transport(url)
        await collect(transport, "warm up")
        # Kill the connection out from under the transport; the next turn
        # must notice, reconnect, and re-send (no tokens were lost).
        await transport._ws.close()
        assert await collect(transport, "still there?") == ["back"]
        assert brain.inputs == ["warm up", "still there?"]
        await transport.aclose()


async def test_server_requires_token():
    try:
        BrainServer(FakeBrain(), "localhost", 0, token="")
    except ValueError:
        pass
    else:
        raise AssertionError("BrainServer accepted an empty token")
