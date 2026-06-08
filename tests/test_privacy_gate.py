"""Tests for MicGate, including the hardware mute hook (Phase 6.2)."""

from __future__ import annotations

from robot.privacy.gate import MicGate


def test_default_enabled_state():
    assert MicGate(enabled=True).enabled is True
    assert MicGate(enabled=False).enabled is False


def test_toggle_flips_software_flag():
    gate = MicGate(enabled=True)
    assert gate.toggle() is False
    assert gate.enabled is False
    assert gate.toggle() is True
    assert gate.enabled is True


def test_hardware_mute_overrides_software_on():
    gate = MicGate(enabled=True)
    muted = {"v": True}
    gate.set_hardware_mute_pin(lambda: muted["v"])
    assert gate.enabled is False  # hardware wins
    assert gate.software_enabled is True  # raw flag unchanged
    muted["v"] = False
    assert gate.enabled is True


def test_no_hardware_pin_defaults_unmuted():
    gate = MicGate(enabled=True)
    assert gate.hardware_muted() is False
    assert gate.enabled is True


def test_flaky_pin_reader_fails_safe_to_muted():
    gate = MicGate(enabled=True)

    def boom():
        raise RuntimeError("gpio glitch")

    gate.set_hardware_mute_pin(boom)
    # Privacy default: uncertain → treat as muted.
    assert gate.hardware_muted() is True
    assert gate.enabled is False


def test_software_off_stays_off_regardless_of_hardware():
    gate = MicGate(enabled=False)
    gate.set_hardware_mute_pin(lambda: False)
    assert gate.enabled is False
