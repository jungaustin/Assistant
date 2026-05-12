"""Transport seam between the Edge (mic/camera/speaker — eventually a Pi) and
the Brain (LLM + tools — eventually a Mac).

Today everything runs in one process via InProcessTransport. In Phase 5, a
WebSocketTransport can drop in here without changes to the Edge or the Brain,
as long as it satisfies the same `respond(utterance) -> async iterator of
tokens` contract.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol


class Brain(Protocol):
    """Anything that can turn an utterance into a stream of token strings.

    The Agent class satisfies this implicitly via its `stream()` generator."""

    def stream(self, input_text: str): ...


class Transport(Protocol):
    async def respond(self, utterance: str) -> AsyncIterator[str]: ...


class InProcessTransport:
    """Local, same-process transport. Wraps the Brain's blocking generator
    in `asyncio.to_thread` so the Edge's event loop stays responsive."""

    def __init__(self, brain: Brain):
        self.brain = brain

    async def respond(self, utterance: str) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        gen = self.brain.stream(utterance)
        sentinel = object()

        def next_token():
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        while True:
            token = await loop.run_in_executor(None, next_token)
            if token is sentinel:
                return
            yield token
