"""Tests for Edge._listen_followup's idle-timeout behavior.

The follow-up window bounds how long we wait for the user to *start*
speaking, not how long they may speak. These tests drive the real coroutine
with a fake STT (no audio): once recording starts, the deadline must no
longer apply, so a reply that runs past the window is captured in full
instead of being truncated and dropped (the old absolute-timeout bug).
"""

from __future__ import annotations

import threading
import time


from robot.main import Edge
from robot.privacy import MicGate


class FakeSTT:
    """Scriptable stand-in for SpeechToText.

    speech_delay: seconds after listen() begins before recording "starts".
        None means the user never speaks (pure silence path).
    speak_duration: how long recording stays active once it starts.
    """

    def __init__(self, *, speech_delay, speak_duration=0.0, result="follow up"):
        self.speech_delay = speech_delay
        self.speak_duration = speak_duration
        self.result = result
        self.is_recording = False
        self.bypass = 0.0
        self.aborted = False
        self._abort = threading.Event()

    def set_wake_word_bypass(self, seconds: float) -> None:
        self.bypass = seconds

    def abort(self) -> None:
        self.aborted = True
        self._abort.set()

    def listen(self):
        if self.speech_delay is None:
            self._abort.wait()  # silence: block until the window aborts us
            return None
        if self._abort.wait(timeout=self.speech_delay):
            return None  # aborted before speech began
        self.is_recording = True
        # Deliberately ignore abort here: once recording, the window must not
        # be able to cut us off.
        time.sleep(self.speak_duration)
        self.is_recording = False
        return self.result


def _edge(stt: FakeSTT) -> Edge:
    return Edge(
        transport=None,
        mic_gate=MicGate(enabled=True),
        speech_to_text=stt,
        text_to_speech=object(),
    )


async def test_speech_within_window_is_captured():
    stt = FakeSTT(speech_delay=0.1, speak_duration=0.2, result="yes please")
    edge = _edge(stt)
    text = await edge._listen_followup(timeout=0.5)
    assert text == "yes please"
    assert not stt.aborted


async def test_silence_times_out_and_aborts():
    stt = FakeSTT(speech_delay=None)
    edge = _edge(stt)
    text = await edge._listen_followup(timeout=0.2)
    assert text is None
    assert stt.aborted


async def test_long_reply_started_in_window_is_not_truncated():
    # Speech starts at 0.1s (inside the 0.3s window) but runs to 0.7s — well
    # past the window. The old absolute wait_for(0.3) would have aborted it
    # and returned None; the idle timeout must let it finish.
    stt = FakeSTT(speech_delay=0.1, speak_duration=0.6, result="a long answer")
    edge = _edge(stt)
    text = await edge._listen_followup(timeout=0.3)
    assert text == "a long answer"
    assert not stt.aborted


async def test_disabled_mic_returns_none_without_listening():
    stt = FakeSTT(speech_delay=0.1)
    edge = Edge(
        transport=None,
        mic_gate=MicGate(enabled=False),
        speech_to_text=stt,
        text_to_speech=object(),
    )
    assert await edge._listen_followup(timeout=0.2) is None
    assert stt.bypass == 0.0  # never even armed the bypass


async def test_bypass_is_reset_after_followup():
    stt = FakeSTT(speech_delay=0.05, speak_duration=0.05)
    edge = _edge(stt)
    await edge._listen_followup(timeout=0.3)
    assert stt.bypass == 0.0  # wake word required again after the window
