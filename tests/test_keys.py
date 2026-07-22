"""Tests for the hotkey controller (F17 wake toggle / F18 deafen toggle).

Drives HotkeyController with a fake ear and real MicGate; beeps are silenced
by monkeypatching so tests never touch an audio device.
"""

from __future__ import annotations

import pytest

import robot.keys as keys
from robot.keys import HotkeyController
from robot.privacy.gate import MicGate


class FakeEar:
    def __init__(self, is_recording: bool = False):
        self.is_recording = is_recording
        self.aborted = 0
        self.force_started = 0

    def abort(self):
        self.aborted += 1

    def force_start(self):
        self.force_started += 1


class FakeTTS:
    def __init__(self, is_speaking: bool = False):
        self.is_speaking = is_speaking
        self.stopped = 0

    def stop(self):
        self.stopped += 1
        self.is_speaking = False


@pytest.fixture(autouse=True)
def silence_beeps(monkeypatch):
    for name in ("wake_beep", "cancel_beep", "mute_beep", "ready_beep"):
        monkeypatch.setattr(keys, name, lambda: None)


def test_wake_idle_force_starts_listen():
    ear = FakeEar(is_recording=False)
    HotkeyController(MicGate(enabled=True), ear).wake_pressed()
    assert ear.force_started == 1
    assert ear.aborted == 0


def test_wake_mid_capture_aborts_instead():
    ear = FakeEar(is_recording=True)
    HotkeyController(MicGate(enabled=True), ear).wake_pressed()
    assert ear.aborted == 1
    assert ear.force_started == 0


def test_wake_while_deafened_does_nothing():
    ear = FakeEar()
    HotkeyController(MicGate(enabled=False), ear).wake_pressed()
    assert ear.aborted == 0
    assert ear.force_started == 0


def test_wake_while_speaking_stops_playback():
    ear = FakeEar(is_recording=False)
    tts = FakeTTS(is_speaking=True)
    HotkeyController(MicGate(enabled=True), ear, tts).wake_pressed()
    assert tts.stopped == 1
    # Barge-in is a distinct action: it doesn't also start/abort a listen.
    assert ear.force_started == 0
    assert ear.aborted == 0


def test_wake_while_speaking_stops_even_when_deafened():
    # Silencing the speaker is independent of the mic gate.
    ear = FakeEar()
    tts = FakeTTS(is_speaking=True)
    HotkeyController(MicGate(enabled=False), ear, tts).wake_pressed()
    assert tts.stopped == 1


def test_wake_not_speaking_falls_through_to_listen():
    ear = FakeEar(is_recording=False)
    tts = FakeTTS(is_speaking=False)
    HotkeyController(MicGate(enabled=True), ear, tts).wake_pressed()
    assert tts.stopped == 0
    assert ear.force_started == 1


def test_wake_barge_in_takes_precedence_over_mid_capture():
    # If somehow both talking and recording, stopping speech wins (no abort).
    ear = FakeEar(is_recording=True)
    tts = FakeTTS(is_speaking=True)
    HotkeyController(MicGate(enabled=True), ear, tts).wake_pressed()
    assert tts.stopped == 1
    assert ear.aborted == 0


def test_wake_tts_stop_error_never_escapes():
    class ExplodingTTS(FakeTTS):
        def stop(self):
            raise RuntimeError("boom")

    ear = FakeEar()
    HotkeyController(MicGate(enabled=True), ear, ExplodingTTS(is_speaking=True)).wake_pressed()
    # Swallowed: no force_start fallback either, the press was consumed by barge-in.
    assert ear.force_started == 0


def test_deafen_disables_gate_and_aborts_pending_listen():
    ear = FakeEar()
    gate = MicGate(enabled=True)
    HotkeyController(gate, ear).deafen_pressed()
    assert gate.enabled is False
    assert ear.aborted == 1


def test_deafen_again_reenables_without_abort():
    ear = FakeEar()
    gate = MicGate(enabled=False)
    HotkeyController(gate, ear).deafen_pressed()
    assert gate.enabled is True
    assert ear.aborted == 0


def test_press_dropped_while_another_action_in_flight():
    # RealtimeSTT/PortAudio control isn't concurrency-safe; a press that lands
    # while another action holds the lock is dropped, not run (prevents the
    # native double-free from a spammed button).
    ear = FakeEar(is_recording=False)
    controller = HotkeyController(MicGate(enabled=True), ear)
    assert controller._action_lock.acquire(blocking=False)  # simulate in-flight
    try:
        controller.wake_pressed()
        controller.deafen_pressed()
    finally:
        controller._action_lock.release()
    assert ear.force_started == 0
    assert ear.aborted == 0


def test_lock_released_after_action_so_next_press_runs():
    ear = FakeEar(is_recording=False)
    controller = HotkeyController(MicGate(enabled=True), ear)
    controller.wake_pressed()
    controller.wake_pressed()  # second press must not be blocked by a stuck lock
    assert ear.force_started == 2


def test_lock_released_even_when_action_raises():
    class ExplodingEar(FakeEar):
        def force_start(self):
            raise RuntimeError("boom")

    ear = ExplodingEar(is_recording=False)
    controller = HotkeyController(MicGate(enabled=True), ear)
    controller.wake_pressed()  # raises internally, but lock must be released
    assert controller._action_lock.acquire(blocking=False)
    controller._action_lock.release()


def test_ear_errors_never_escape_the_key_handler():
    class ExplodingEar(FakeEar):
        def abort(self):
            raise RuntimeError("boom")

        def force_start(self):
            raise RuntimeError("boom")

    gate = MicGate(enabled=True)
    HotkeyController(gate, ExplodingEar(is_recording=True)).wake_pressed()
    HotkeyController(gate, ExplodingEar()).wake_pressed()
    controller = HotkeyController(gate, ExplodingEar())
    controller.deafen_pressed()
    assert gate.enabled is False
