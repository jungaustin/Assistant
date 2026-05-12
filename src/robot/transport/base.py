"""Minimal Transport protocol. Reflects how main.Edge consumes a Brain.

The Brain may be in-process or across a network boundary; the transport
hides that. Today: InProcessTransport wraps an Agent. Phase 8: a WebSocket
transport with the same shape.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    def respond(self, utterance: str) -> AsyncIterator[str]:
        """Yield brain response tokens for `utterance`."""
