"""Minimal Voice protocol. Reflects what main.Edge actually calls today.

speak() accepts a single string or an iterator of token strings; the latter
lets the agent stream tokens straight into the TTS pipeline.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Union, runtime_checkable


@runtime_checkable
class Voice(Protocol):
    def speak(self, text_or_stream: Union[str, Iterable[str]]) -> None:
        """Synthesize and play audio. Blocking; called via asyncio.to_thread."""
