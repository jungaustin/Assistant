"""Tests for Edge._listen_with_watchdog's dead-recorder recovery.

The wake-word listen() blocks until RealtimeSTT's background capture thread
signals speech. That daemon thread can die silently, after which listen()
never returns. The watchdog must notice (via is_healthy()), abort the wedged
listen, rebuild the recorder (restart()), and return None so the run loop
listens again — instead of hanging for minutes.
"""

from __future__ import annotations

import threading

import robot.main as main_mod
from robot.main import Edge
from robot.privacy import MicGate


class DeadRecorderSTT:
    """A recorder whose capture thread has died: listen() wedges until aborted,
    and is_healthy() reports dead."""

    def __init__(self):
        self.healthy = False
        self.aborted = False
        self.restarted = False
        self._release = threading.Event()

    def listen(self):
        self._release.wait()  # wedged until abort() releases us
        return None

    @property
    def is_recording(self) -> bool:
        return False

    def is_healthy(self) -> bool:
        return self.healthy

    def abort(self) -> None:
        self.aborted = True
        self._release.set()

    def restart(self) -> None:
        self.restarted = True

    def set_wake_word_bypass(self, seconds: float) -> None:
        pass


class HealthySTT:
    def __init__(self, result="hello"):
        self.result = result
        self.restarted = False

    def listen(self):
        return self.result

    @property
    def is_recording(self) -> bool:
        return False

    def is_healthy(self) -> bool:
        return True

    def abort(self) -> None:
        pass

    def restart(self) -> None:
        self.restarted = True

    def set_wake_word_bypass(self, seconds: float) -> None:
        pass


def _edge(stt) -> Edge:
    return Edge(
        transport=None,
        mic_gate=MicGate(enabled=True),
        speech_to_text=stt,
        text_to_speech=object(),
    )


async def test_watchdog_recovers_dead_recorder(monkeypatch):
    monkeypatch.setattr(main_mod, "STT_HEALTH_POLL_SECONDS", 0.01)
    stt = DeadRecorderSTT()
    edge = _edge(stt)
    text = await edge._listen_once()
    assert text is None
    assert stt.aborted  # the wedged listen was unblocked
    assert stt.restarted  # a fresh recorder was built


async def test_watchdog_passes_through_when_healthy(monkeypatch):
    monkeypatch.setattr(main_mod, "STT_HEALTH_POLL_SECONDS", 0.01)
    stt = HealthySTT(result="play some music")
    edge = _edge(stt)
    text = await edge._listen_once()
    assert text == "play some music"
    assert not stt.restarted  # healthy recorder is never torn down
