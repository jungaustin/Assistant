"""Audio cues must never overlap inside PortAudio.

`sd.play()` drives one module-global stream and stops whatever is already
playing, so two cues at once had one thread tearing down the CoreAudio stream
another was still inside wait() on. That crashed the robot with SIGSEGV when
the deafen key was pressed repeatedly (2026-08-25 session). HotkeyController's
action lock does not cover it — _fire() returns the moment the thread starts.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from robot.voice import beep


class _FakeSD:
    """Records overlap: play() that starts while another is unfinished."""

    def __init__(self, play_seconds: float = 0.05):
        self.play_seconds = play_seconds
        self.active = 0
        self.max_active = 0
        self.plays = 0
        self._lock = threading.Lock()

    def play(self, data, samplerate):
        with self._lock:
            self.active += 1
            self.plays += 1
            self.max_active = max(self.max_active, self.active)

    def wait(self):
        time.sleep(self.play_seconds)
        with self._lock:
            self.active -= 1


def _hammer(n: int = 12) -> _FakeSD:
    fake = _FakeSD()
    with patch.object(beep, "_sd", fake):
        threads = [
            threading.Thread(target=beep._play_blips, args=(440, 0.01, 2, 0.01))
            for _ in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
    return fake


def test_concurrent_cues_never_overlap():
    fake = _hammer()
    assert fake.max_active == 1, f"{fake.max_active} overlapping sd.play() calls"


def test_every_cue_still_plays_when_they_queue_up():
    # Serialising must not silently swallow cues at this rate — the deafen
    # blip is the only signal the mic just went off.
    fake = _hammer(n=12)
    assert fake.plays == 12


def test_missing_audio_device_is_not_fatal():
    with patch.object(beep, "_sd", None):
        beep._play_blips(440, 0.01, 1, 0.01)  # must simply return


def test_playback_error_releases_the_lock():
    class _Boom:
        def play(self, *a, **k):
            raise RuntimeError("device busy")

        def wait(self):
            pass

    with patch.object(beep, "_sd", _Boom()):
        beep._play_blips(440, 0.01, 1, 0.01)
    assert not beep._PLAY_LOCK.locked(), "lock leaked after a playback error"


def test_public_cues_are_fire_and_forget():
    fake = _FakeSD(play_seconds=0.01)
    with patch.object(beep, "_sd", fake):
        t0 = time.monotonic()
        beep.mute_beep()
        beep.ready_beep()
        elapsed = time.monotonic() - t0
        time.sleep(0.4)
    assert elapsed < 0.05, "cue helpers must not block the caller"
    assert fake.max_active == 1
