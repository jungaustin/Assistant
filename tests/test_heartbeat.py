"""Tests for the heartbeat loop."""

from __future__ import annotations

import asyncio

import pytest

from robot.core.bus import Bus
from robot.core.heartbeat import heartbeat_loop


@pytest.mark.asyncio
async def test_heartbeat_loop_emits_at_interval():
    bus = Bus()
    received = []

    async def consumer():
        async with bus.subscribe() as sub:
            async for event in sub:
                received.append(event)
                if len(received) == 3:
                    break

    consumer_task = asyncio.create_task(consumer())
    while bus.subscriber_count < 1:
        await asyncio.sleep(0)

    # 50ms interval → 3 heartbeats inside ~200ms.
    hb_task = asyncio.create_task(
        heartbeat_loop(bus, source="ear", interval_s=0.05)
    )
    await asyncio.wait_for(consumer_task, timeout=1.0)
    hb_task.cancel()
    try:
        await hb_task
    except asyncio.CancelledError:
        pass

    assert len(received) == 3
    assert all(e.type == "heartbeat" for e in received)
    assert all(e.source == "ear" for e in received)
    assert all(e.interval_s == 0.05 for e in received)


@pytest.mark.asyncio
async def test_heartbeat_loop_exits_on_cancellation():
    bus = Bus()
    hb_task = asyncio.create_task(
        heartbeat_loop(bus, source="brain", interval_s=10.0)
    )
    await asyncio.sleep(0)  # let the task hit its first sleep
    hb_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await hb_task


@pytest.mark.asyncio
async def test_heartbeat_loop_rejects_zero_interval():
    bus = Bus()
    with pytest.raises(ValueError, match="interval_s must be > 0"):
        await heartbeat_loop(bus, source="x", interval_s=0)
