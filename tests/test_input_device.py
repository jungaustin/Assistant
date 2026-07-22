"""Tests for resolve_input_device_index (mic pinning via STT_INPUT_DEVICE).

PyAudio is faked so tests run without an audio device. The contract:
case-insensitive substring match over input-capable devices only, None on
empty fragment or no match (caller falls back to the system default input).
"""

from __future__ import annotations

import sys
import types

import pytest

from robot.config import resolve_input_device_index


class _FakePyAudio:
    def __init__(self, devices):
        self._devices = devices
        self.terminated = False

    def get_device_count(self):
        return len(self._devices)

    def get_device_info_by_index(self, i):
        return self._devices[i]

    def terminate(self):
        self.terminated = True


@pytest.fixture
def fake_pyaudio(monkeypatch):
    """Install a fake `pyaudio` module; yields a setter for the device list."""
    holder = {}

    module = types.ModuleType("pyaudio")

    def make(devices):
        instance = _FakePyAudio(devices)
        holder["instance"] = instance
        module.PyAudio = lambda: instance
        return instance

    monkeypatch.setitem(sys.modules, "pyaudio", module)
    return make


_DEVICES = [
    {"name": "Bose QC Ultra Earbuds", "maxInputChannels": 1},
    {"name": "MacBook Pro Speakers", "maxInputChannels": 0},
    {"name": "MacBook Pro Microphone", "maxInputChannels": 1},
]


def test_matches_by_name_substring(fake_pyaudio):
    fake_pyaudio(_DEVICES)
    assert resolve_input_device_index("MacBook Pro Microphone") == 2


def test_match_is_case_insensitive_and_partial(fake_pyaudio):
    fake_pyaudio(_DEVICES)
    assert resolve_input_device_index("macbook pro mic") == 2


def test_output_only_devices_never_match(fake_pyaudio):
    # "MacBook Pro" alone would hit the Speakers first by index — but that's
    # an output-only device, so the Microphone must win.
    fake_pyaudio(_DEVICES)
    assert resolve_input_device_index("macbook pro") == 2


def test_no_match_returns_none(fake_pyaudio):
    fake_pyaudio(_DEVICES)
    assert resolve_input_device_index("AirPods") is None


def test_empty_fragment_returns_none_without_touching_pyaudio(fake_pyaudio):
    instance = fake_pyaudio(_DEVICES)
    assert resolve_input_device_index("") is None
    assert instance.terminated is False  # early-out, PyAudio never built


def test_pyaudio_is_terminated_after_lookup(fake_pyaudio):
    instance = fake_pyaudio(_DEVICES)
    resolve_input_device_index("microphone")
    assert instance.terminated is True
