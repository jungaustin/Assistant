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

Cues mix fine over TTS speech or Spotify, but NOT over each other:
`sd.play()` drives a single module-global stream and its own docs say it
"cannot be used for multiple overlapping playbacks" — it calls stop() on
whatever is already playing. Two beeps landing together therefore had one
thread closing the CoreAudio stream another was still inside wait() on, which
surfaced as `PaMacCore ... err='!obj'` / `err='-50'` and then killed the
process with SIGSEGV. Spamming the deafen key was enough to do it: mute_beep
is 0.27s long and the toggles arrived ~0.3s apart. `_PLAY_LOCK` below keeps
exactly one cue in flight; HotkeyController's own lock does not cover this,
because _fire() returns as soon as the thread is spawned.
"""

from __future__ import annotations

import threading

import numpy as np

# Imported here, on the main thread at startup, rather than inside the worker.
# `import sounddevice` initialises PortAudio through cffi, and doing that for
# the first time on a throwaway beep thread is its own crash (SIGSEGV inside
# ffi_call during module exec). A machine with no audio device must still boot.
try:
    import sounddevice as _sd
except Exception:  # pragma: no cover - depends on host audio setup
    _sd = None

# One cue at a time. Held across play()+wait() so the next beep cannot stop the
# stream mid-playback. Beeps are ~0.1-0.3s, so a short block is imperceptible;
# a cue that cannot get the lock in time is dropped rather than queued, since a
# beep that arrives late is worse than no beep.
_PLAY_LOCK = threading.Lock()
_PLAY_LOCK_TIMEOUT = 1.5

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
    if _sd is None:
        return

    gap = np.zeros(int(_SAMPLE_RATE * gap_s), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i in range(count):
        if i:
            parts.append(gap)
        parts.append(_tone(freq_hz, duration_s))

    if not _PLAY_LOCK.acquire(timeout=_PLAY_LOCK_TIMEOUT):
        return
    try:
        _sd.play(np.concatenate(parts), _SAMPLE_RATE)
        _sd.wait()
    except Exception:
        # A beep is a nicety; never let a missing/busy audio device take
        # down the thread that asked for it.
        pass
    finally:
        _PLAY_LOCK.release()


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
