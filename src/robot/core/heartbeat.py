"""Heartbeat helper for long-running components.

Any component that survives multiple turns (Ear, Voice, Brain, Transport)
can run this as a background task and publish a Heartbeat every N
seconds. A separate watchdog (in Edge today, in the Conductor when it
exists) subscribes to Heartbeat events and warns if a source has gone
silent.

Per the Phase 5 plan, components don't actually wire this in yet — the
helper exists so Phase 8's WebSocket reconnect logic has the foundation:
when the link drops, missing heartbeats are how the conductor knows.

Usage::

    bus = Bus()
    asyncio.create_task(heartbeat_loop(bus, source="ear", interval_s=5.0))
    # ... and somewhere a watchdog:
    async with bus.subscribe() as sub:
        async for event in sub:
            if event.type == "heartbeat":
                last_seen[event.source] = event.ts
"""

from __future__ import annotations

import asyncio
import logging

from robot.core.bus import Bus
from robot.core.events import Heartbeat

logger = logging.getLogger(__name__)


async def heartbeat_loop(
    bus: Bus, *, source: str, interval_s: float = 5.0
) -> None:
    """Publish a Heartbeat to `bus` every `interval_s` seconds until
    cancelled. Cancellation is the normal shutdown signal.
    """
    if interval_s <= 0:
        raise ValueError(f"interval_s must be > 0, got {interval_s}")

    try:
        while True:
            await bus.publish(Heartbeat(source=source, interval_s=interval_s))
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        # Normal shutdown — no log noise.
        raise
    except Exception:
        # Anything else is unexpected; log loudly and re-raise so the
        # supervising task sees it.
        logger.exception("heartbeat_loop(%s) crashed", source)
        raise
