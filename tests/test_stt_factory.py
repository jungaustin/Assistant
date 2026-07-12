"""Regression tests for the injected recorder factory (clip plan T3).

The clip service wires its snapshot hook into the recorder via
make_stt_recorder(on_recording_start=...). If the watchdog then rebuilds a
wedged recorder through SpeechToText.restart(), that wiring must survive —
restart() has to reuse the injected factory, not fall back to a bare default
recorder that would silently drop the hook.
"""

from __future__ import annotations

import sys
import types

import robot.config as config
from robot.ear.realtimestt import SpeechToText


class FakeRecorder:
    def __init__(self):
        self.shutdown_called = False

    def shutdown(self):
        self.shutdown_called = True


def test_restart_rebuilds_via_injected_factory():
    made = []

    def factory():
        recorder = FakeRecorder()
        made.append(recorder)
        return recorder

    stt = SpeechToText(recorder_factory=factory)
    assert stt.recorder is made[0]

    stt.restart()
    assert stt.recorder is made[1]  # fresh recorder from the same factory
    assert made[0].shutdown_called  # old one was torn down


def test_explicit_recorder_still_restarts_via_factory():
    initial = FakeRecorder()
    made = []

    def factory():
        recorder = FakeRecorder()
        made.append(recorder)
        return recorder

    stt = SpeechToText(recorder=initial, recorder_factory=factory)
    assert stt.recorder is initial
    assert made == []  # factory not called when a recorder is supplied

    stt.restart()
    assert initial.shutdown_called
    assert stt.recorder is made[0]


def test_no_factory_falls_back_to_make_stt_recorder(monkeypatch):
    # Today's behavior, unchanged: without an injected factory, restart()
    # rebuilds through config.make_stt_recorder.
    import robot.ear.realtimestt as ear_mod

    built = []

    def fake_make():
        recorder = FakeRecorder()
        built.append(recorder)
        return recorder

    monkeypatch.setattr(ear_mod, "make_stt_recorder", fake_make)
    stt = SpeechToText(recorder=FakeRecorder())
    stt.restart()
    assert stt.recorder is built[0]


class _CapturingRecorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_realtimestt(monkeypatch) -> None:
    module = types.ModuleType("RealtimeSTT")
    module.AudioToTextRecorder = _CapturingRecorder
    monkeypatch.setitem(sys.modules, "RealtimeSTT", module)


def test_make_stt_recorder_no_arg_is_unchanged(monkeypatch):
    _fake_realtimestt(monkeypatch)
    monkeypatch.setattr(config, "STT_PROVIDER", "realtimestt")
    recorder = config.make_stt_recorder()
    assert recorder.kwargs["on_recording_start"] is None
    # Spot-check that existing wiring didn't move.
    assert recorder.kwargs["wake_words"] == "nemo"
    assert recorder.kwargs["spinner"] is False


def test_make_stt_recorder_passes_on_recording_start(monkeypatch):
    _fake_realtimestt(monkeypatch)
    monkeypatch.setattr(config, "STT_PROVIDER", "realtimestt")

    def hook():
        pass

    recorder = config.make_stt_recorder(on_recording_start=hook)
    assert recorder.kwargs["on_recording_start"] is hook
