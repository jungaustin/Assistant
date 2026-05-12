"""Minimal Brain protocol. Reflects what InProcessTransport.respond uses.

A Brain takes a user utterance and yields response tokens (and any tool
output as plain-text tokens). The current Agent class implements this; future
brains (a hand-rolled state machine, a remote brain over WebSocket) can
substitute without touching Edge.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Brain(Protocol):
    def stream(self, input_text: str) -> Iterator[str]:
        """Yield response tokens. Implementations may also drive tool calls."""
