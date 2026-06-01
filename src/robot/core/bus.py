"""Minimal async pub/sub bus.

One bounded queue per subscriber, fan-out on publish. Slow subscribers
drop their oldest events rather than backpressuring the publisher — a
wedged consumer must never stall the voice loop.

Per the Phase 5 plan, components don't publish to this yet; the bus is
here so Phase 8's WebSocket transport has somewhere to plug into. The
shape is small on purpose — promote it to topic-per-queue or a real
broker (NATS, Redis pub/sub) when the use case actually demands it.

Usage::

    bus = Bus()

    async def consumer():
        async with bus.subscribe() as sub:
            async for event in sub:
                handle(event)

    asyncio.create_task(consumer())
    await bus.publish(WakeDetected(source="ear"))
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from robot.core.events import Event

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256


class Subscription:
    """One subscriber's view of the bus. Async-iterable; closes on context exit.

    The bus puts events into ``self.queue``. Iterating drains them in order.
    On ``aclose()`` (or context exit), the bus removes this subscription
    from its fan-out list.
    """

    def __init__(self, bus: "Bus", queue_size: int) -> None:
        self._bus = bus
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size)
        self._closed = False

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iterator()

    async def _iterator(self) -> AsyncIterator[Event]:
        while not self._closed:
            try:
                event = await self.queue.get()
            except asyncio.CancelledError:
                return
            if event is _SHUTDOWN_SENTINEL:
                return
            yield event

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus._remove(self)
        # Wake the iterator out of its await on queue.get().
        try:
            self.queue.put_nowait(_SHUTDOWN_SENTINEL)
        except asyncio.QueueFull:
            # Drain one and try again — the consumer will see the sentinel.
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(_SHUTDOWN_SENTINEL)


# Sentinel object used to wake a subscription's iterator out of queue.get()
# when aclose() runs. Typed as Event for the queue's sake; never delivered to
# consumers.
_SHUTDOWN_SENTINEL: Event = object()  # type: ignore[assignment]


class Bus:
    """Async pub/sub. Fan-out delivery, bounded per-subscriber queues,
    drop-oldest on subscriber backpressure (the publisher never blocks).

    Thread-safety: single-event-loop only. Don't share across loops.
    """

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._subs: List[Subscription] = []
        self._queue_size = queue_size

    def _remove(self, sub: Subscription) -> None:
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    async def publish(self, event: Event) -> None:
        """Deliver `event` to every current subscriber.

        Non-blocking: if a subscriber's queue is full, drop its oldest event
        and enqueue this one. Logs a warning so a chronically slow consumer
        is visible.
        """
        # Snapshot to avoid mutation-during-iteration if a subscriber's
        # consumer closes mid-publish.
        for sub in list(self._subs):
            self._deliver(sub, event)

    def _deliver(self, sub: Subscription, event: Event) -> None:
        try:
            sub.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                dropped = sub.queue.get_nowait()
                logger.warning(
                    "bus: dropped oldest event for slow subscriber "
                    "(dropped type=%s, new type=%s)",
                    getattr(dropped, "type", "?"),
                    getattr(event, "type", "?"),
                )
            except asyncio.QueueEmpty:
                pass
            # Try once more; under heavy contention with cancellation this
            # could still fail and we let it propagate.
            sub.queue.put_nowait(event)

    @asynccontextmanager
    async def subscribe(
        self, queue_size: int | None = None
    ) -> AsyncIterator[Subscription]:
        """Context manager that adds a subscription, yields it, and closes
        it on exit. Late subscribers do NOT see prior events (no replay)."""
        sub = Subscription(self, queue_size or self._queue_size)
        self._subs.append(sub)
        try:
            yield sub
        finally:
            await sub.aclose()

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)
