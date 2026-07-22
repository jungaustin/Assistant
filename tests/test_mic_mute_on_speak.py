"""Tests that the Edge mutes the recorder's own audio capture while TTS is
playing, and unmutes it once playback finishes.

RealtimeSTT's wake-word/VAD thread runs continuously in the background,
independent of when the Edge calls listen() — so without an explicit mute it
can hear the robot's own TTS coming out of the speaker. `set_microphone`
gates that background capture at the source.
"""

from __future__ import annotations

from robot.main import Edge
from robot.privacy import MicGate


class RecordingSTT:
    """Records the sequence of set_microphone(on) calls it receives."""

    def __init__(self):
        self.calls: list[bool] = []

    def set_microphone(self, on: bool) -> None:
        self.calls.append(on)


class NoMicMethodSTT:
    """An Ear without set_microphone (older engine / bare test fake)."""


class RecordingTTS:
    def __init__(self, on_speak=None):
        self._on_speak = on_speak

    def speak(self, chunks):
        if self._on_speak:
            self._on_speak()
        for _ in chunks:
            pass


def _edge(stt, tts) -> Edge:
    return Edge(
        transport=None,
        mic_gate=MicGate(enabled=True),
        speech_to_text=stt,
        text_to_speech=tts,
    )


async def test_speak_stream_mutes_mic_during_playback_and_unmutes_after():
    stt = RecordingSTT()

    def during_speak():
        # By the time TTS playback runs, the mic must already be muted.
        assert stt.calls == [False]

    edge = _edge(stt, RecordingTTS(on_speak=during_speak))

    async def tokens():
        yield "hello"

    await edge._speak_stream(tokens())

    assert stt.calls == [False, True]  # muted, then unmuted


async def test_speak_stream_unmutes_even_if_tts_raises():
    stt = RecordingSTT()

    class ExplodingTTS:
        def speak(self, chunks):
            for _ in chunks:
                pass
            raise RuntimeError("engine died")

    edge = _edge(stt, ExplodingTTS())

    async def tokens():
        yield "hello"

    try:
        await edge._speak_stream(tokens())
    except RuntimeError:
        pass

    assert stt.calls == [False, True]


async def test_speak_stream_is_a_noop_when_ear_lacks_set_microphone():
    edge = _edge(NoMicMethodSTT(), RecordingTTS())

    async def tokens():
        yield "hello"

    # Must not raise even though the Ear has no set_microphone method.
    await edge._speak_stream(tokens())
