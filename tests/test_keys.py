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
