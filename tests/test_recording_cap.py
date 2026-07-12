"""Tests for the runaway-recording cap.

When the VAD never registers end-of-speech, the recorder keeps capturing
forever. Both listen paths must enforce MAX_UTTERANCE_SECONDS by force-stopping
the recorder so listen() returns the partial transcription instead of hanging.
"""

from __future__ import annotations

import threading

import robot.main as main_mod
from robot.main import Edge
from robot.privacy import MicGate


class StuckRecordingSTT:
    """Begins recording immediately, then never detects end-of-speech: listen()
    only returns once force_stop() (or abort()) releases it."""

    def __init__(self, result="partial capture"):
        self.result = result
        self.is_recording = False
        self.forced = False
        self._release = threading.Event()

    def listen(self):
        self.is_recording = True
        self._release.wait()  # never ends on its own
        self.is_recording = False
        return self.result if self.forced else None

    def force_stop(self):
        self.forced = True
        self._release.set()

    def abort(self):
        self._release.set()

    def is_healthy(self) -> bool:
        return True

    def restart(self) -> None:
        pass

    def set_wake_word_bypass(self, seconds: float) -> None:
        pass


def _edge(stt) -> Edge:
    return Edge(
        transport=None,
        mic_gate=MicGate(enabled=True),
        speech_to_text=stt,
        text_to_speech=object(),
    )


async def test_watchdog_caps_runaway_recording(monkeypatch):
    monkeypatch.setattr(main_mod, "STT_HEALTH_POLL_SECONDS", 0.01)
    monkeypatch.setattr(main_mod, "MAX_UTTERANCE_SECONDS", 0.05)
    stt = StuckRecordingSTT(result="captured words")
    edge = _edge(stt)
    text = await edge._listen_once()
    assert stt.forced  # the cap fired
    assert text == "captured words"  # partial transcription is returned


async def test_followup_caps_runaway_recording(monkeypatch):
    monkeypatch.setattr(main_mod, "MAX_UTTERANCE_SECONDS", 0.3)
    stt = StuckRecordingSTT(result="hello there friend")
    edge = _edge(stt)
    text = await edge._listen_followup(timeout=0.2)
    assert stt.forced
    assert text == "hello there friend"
