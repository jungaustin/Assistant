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


# --- Observer API (clip plan decision 3A) ---


def test_observer_fires_on_set_with_new_value():
    gate = MicGate(enabled=True)
    seen = []
    gate.subscribe(seen.append)
    gate.set(False)
    assert seen == [False]
    gate.set(True)
    assert seen == [False, True]


def test_observer_fires_on_toggle_with_new_value():
    gate = MicGate(enabled=True)
    seen = []
    gate.subscribe(seen.append)
    assert gate.toggle() is False
    assert gate.toggle() is True
    assert seen == [False, True]


def test_raising_observer_is_swallowed_and_others_still_fire():
    gate = MicGate(enabled=True)
    seen = []

    def boom(value):
        raise RuntimeError("subscriber bug")

    gate.subscribe(boom)
    gate.subscribe(seen.append)
    gate.set(False)  # must not raise
    assert gate.software_enabled is False  # state change went through
    assert seen == [False]  # later subscribers unaffected


def test_observer_runs_outside_the_lock():
    # An observer reading gate state would deadlock on the (non-reentrant)
    # lock if notification happened while holding it.
    gate = MicGate(enabled=True)
    seen = []
    gate.subscribe(lambda value: seen.append(gate.software_enabled))
    gate.set(False)
    assert seen == [False]


def test_subscribe_does_not_fire_retroactively():
    gate = MicGate(enabled=True)
    gate.set(False)
    seen = []
    gate.subscribe(seen.append)
    assert seen == []
