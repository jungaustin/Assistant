"""Short audio cues, played fire-and-forget on a daemon thread.

Cues:
  ready_beep()      — single low blip: the assistant booted and is
                      listening. Fired once at the top of the Edge loop,
                      after STT/TTS models have loaded. Also reused as the
                      un-deafen cue ("listening again").
  timer_done_beep() — double higher blip: a timer finished.
  wake_beep()       — single high blip: a listen started — wake word heard
                      or the wake key force-started one.
  cancel_beep()     — single low blip: the wake key cancelled a capture (or
                      was pressed while deafened and did nothing).
  mute_beep()       — double low blip: the mic was just deafened.

Generated sine waves via sounddevice rather than afplay/system sounds so the
same code runs on the Pi after the Phase 8 split. Each tone gets a 5ms linear
fade in/out — without it the hard edges click audibly.

Playback opens its own PortAudio stream, so cues mix fine over TTS speech or
Spotify; they never contend with the main voice pipeline.
"""

from __future__ import annotations

import threading

import numpy as np

_SAMPLE_RATE = 22050
_VOLUME = 0.3
_FADE_SECONDS = 0.005


def _tone(freq_hz: float, duration_s: float) -> np.ndarray:
    t = np.linspace(0, duration_s, int(_SAMPLE_RATE * duration_s), endpoint=False)
    wave = _VOLUME * np.sin(2 * np.pi * freq_hz * t)
    fade_n = int(_SAMPLE_RATE * _FADE_SECONDS)
    if fade_n > 0 and len(wave) > 2 * fade_n:
        ramp = np.linspace(0.0, 1.0, fade_n)
        wave[:fade_n] *= ramp
        wave[-fade_n:] *= ramp[::-1]
    return wave.astype(np.float32)


def _play_blips(freq_hz: float, duration_s: float, count: int, gap_s: float) -> None:
    import sounddevice as sd

    gap = np.zeros(int(_SAMPLE_RATE * gap_s), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i in range(count):
        if i:
            parts.append(gap)
        parts.append(_tone(freq_hz, duration_s))
    try:
        sd.play(np.concatenate(parts), _SAMPLE_RATE)
        sd.wait()
    except Exception:
        # A beep is a nicety; never let a missing/busy audio device take
        # down the thread that asked for it.
        pass


def _fire(freq_hz: float, duration_s: float, count: int = 1, gap_s: float = 0.07):
    threading.Thread(
        target=_play_blips,
        args=(freq_hz, duration_s, count, gap_s),
        daemon=True,
    ).start()


def ready_beep() -> None:
    """Single low blip — the assistant is booted and listening."""
    _fire(660, 0.08)


def timer_done_beep() -> None:
    """Double higher blip — a timer just finished."""
    _fire(1175, 0.12, count=2)


def wake_beep() -> None:
    """Single high blip — a listen started (wake word or wake key); talk now."""
    _fire(990, 0.08)


def cancel_beep() -> None:
    """Single low blip — the wake key cancelled/was refused."""
    _fire(330, 0.1)


def mute_beep() -> None:
    """Double low blip — the mic is now deafened."""
    _fire(330, 0.1, count=2)
