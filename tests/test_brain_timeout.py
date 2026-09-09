"""Tests for brain request timeouts and failure handling.

Two halves, and both are needed to be an improvement:

  1. The client must bound its own wait. The openai SDK defaults to 600s with
     2 retries, so a wedged server burns 30 minutes in total silence — the
     field failure that motivated this (three 10-minute 500s while a
     CPU-bound Ollama ground through the prompt at 4 tok/s).

  2. The Edge must survive the resulting exception. A timeout that crashes the
     robot mid-conversation is no better than the hang it replaced; the mic
     loop is the whole product.
"""

from __future__ import annotations

import asyncio

import pytest

from robot.brain.openai_compat import OpenAICompatChat
from robot.main import Edge
from robot.privacy import MicGate

# --- 1. the client bounds its own wait ----------------------------------


def test_timeout_and_retries_reach_the_client():
    chat = OpenAICompatChat(
        model="m",
        base_url="http://localhost:11434/v1",
        api_key="k",
        timeout=120.0,
        max_retries=0,
    )
    assert chat._client.timeout == 120.0
    assert chat._client.max_retries == 0


def test_client_defaults_are_left_alone_when_unset():
    """Unset means 'SDK default' — don't invent one and don't crash."""
    chat = OpenAICompatChat(model="m", base_url="http://x/v1", api_key="k")
    assert chat._client is not None


# --- 2. the Edge survives a brain failure -------------------------------


class BoomTransport:
    """A brain that fails the way a timeout does."""

    music_active = False

    def __init__(self):
        self.calls = 0

    def clear_music_active(self):
        pass

    def respond(self, utterance):
        self.calls += 1
        raise TimeoutError("request timed out")


class OneShotSTT:
    """Yields one utterance, then blocks so run() can be cancelled."""

    def __init__(self):
        self.utterances = ["what's my calorie count?"]
        self.is_recording = False

    def listen(self):
        if self.utterances:
            return self.utterances.pop(0)
        raise asyncio.CancelledError()

    def set_wake_word_bypass(self, seconds):
        pass

    def set_microphone(self, on):
        pass

    def abort(self):
        pass


class RecordingTTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, chunks):
        if isinstance(chunks, str):
            self.spoken.append(chunks)
            return
        for c in chunks:
            self.spoken.append(c)


async def test_brain_failure_is_spoken_and_does_not_crash_the_loop():
    transport = BoomTransport()
    tts = RecordingTTS()
    edge = Edge(
        transport, mic_gate=MicGate(), speech_to_text=OneShotSTT(), text_to_speech=tts
    )

    with pytest.raises(asyncio.CancelledError):
        await edge.run()

    assert transport.calls == 1, "the brain was asked exactly once (no retry)"
    said = "".join(tts.spoken)
    assert "brain" in said.lower(), f"expected a spoken failure notice, got {said!r}"
