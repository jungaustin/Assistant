"""Minimal Ear protocol. Reflects what main.Edge actually calls today.

Listen for a single utterance, optionally bypass the wake word for a follow-up,
or abort an in-flight listen. Implementations may use RealtimeSTT, Whisper,
Parakeet, or anything else with this surface.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Ear(Protocol):
    def listen(self) -> Optional[str]:
        """Block until an utterance is captured. Return None on timeout/abort."""

    @property
    def is_recording(self) -> bool:
        """True while actively capturing speech (vs waiting to start)."""

    def set_wake_word_bypass(self, timeout: float) -> None:
        """Skip the wake word for the next listen() within `timeout` seconds."""

    def abort(self) -> None:
        """Interrupt an in-flight listen()."""
