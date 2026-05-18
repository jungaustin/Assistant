"""Tests for the pub/sub bus.

Covers: single + multiple subscribers, no-replay-for-late-subscribers,
unsubscribe-on-context-exit, slow-subscriber drop-oldest (publisher never
blocks), no deadlock under interleaved publish/consume.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from robot.core.bus import Bus
from robot.core.events import BrainToken, Heartbeat, TranscriptReady, WakeDetected


# ---------- Basic delivery ----------


@pytest.mark.asyncio
async def test_single_subscriber_receives_event():
    bus = Bus()
    received = []

    async def consumer():
        async with bus.subscribe() as sub:
            async for event in sub:
                received.append(event)
                if len(received) == 1:
                    break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let consumer enter subscribe()
    await bus.publish(WakeDetected(source="ear"))
    await asyncio.wait_for(task, timeout=1.0)

    assert len(received) == 1
    assert received[0].type == "wake_detected"
    assert received[0].source == "ear"


@pytest.mark.asyncio
async def test_multiple_subscribers_all_get_event():
    bus = Bus()
    a_received, b_received = [], []

    async def consumer(out):
        async with bus.subscribe() as sub:
            async for event in sub:
                out.append(event)
                if len(out) == 2:
                    break

    a = asyncio.create_task(consumer(a_received))
    b = asyncio.create_task(consumer(b_received))
    # Give both consumers time to subscribe before publishing.
    while bus.subscriber_count < 2:
        await asyncio.sleep(0)

    await bus.publish(BrainToken(source="brain", text="hi"))
    await bus.publish(BrainToken(source="brain", text="!"))
    await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)

    assert [e.text for e in a_received] == ["hi", "!"]
    assert [e.text for e in b_received] == ["hi", "!"]


@pytest.mark.asyncio
async def test_late_subscriber_does_not_replay():
    bus = Bus()
    received = []

    # Publish BEFORE anyone subscribes. Those events go nowhere.
    await bus.publish(WakeDetected(source="ear"))
    await bus.publish(TranscriptReady(source="ear", text="hello"))

    async def consumer():
        async with bus.subscribe() as sub:
            async for event in sub:
                received.append(event)
                break  # expect exactly one — the LATER publish

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    # Now publish — only this one should be received.
    await bus.publish(BrainToken(source="brain", text="new"))
    await asyncio.wait_for(task, timeout=1.0)

    assert len(received) == 1
    assert received[0].type == "brain_token"
    assert received[0].text == "new"


# ---------- Lifecycle ----------


@pytest.mark.asyncio
async def test_subscriber_count_decreases_on_exit():
    bus = Bus()

    async def consumer():
        async with bus.subscribe() as sub:
            assert bus.subscriber_count == 1
            # Use the queue once so the iterator is "live" then exit.
            return

    await consumer()
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_publish_with_no_subscribers_is_a_noop():
    bus = Bus()
    # Must not raise, must not block.
    await asyncio.wait_for(bus.publish(WakeDetected(source="ear")), timeout=0.5)


# ---------- Backpressure / no-deadlock ----------


@pytest.mark.asyncio
async def test_slow_subscriber_does_not_block_publisher(caplog):
    """The whole point of drop-oldest: one wedged consumer must never stall
    the voice loop. Fast consumer has a roomy queue (would receive all
    events if it actually got to run); slow consumer has a 2-deep queue
    and never reads, so it forces drop-oldest. The assertion that matters
    is that publish() never blocks despite the wedged consumer."""
    bus = Bus()
    fast_received = []

    async def fast_consumer():
        async with bus.subscribe(queue_size=64) as sub:
            async for event in sub:
                fast_received.append(event)
                if len(fast_received) == 10:
                    break

    # Slow subscriber that never reads from its queue.
    async def slow_consumer():
        async with bus.subscribe(queue_size=2) as sub:
            await asyncio.sleep(10)  # Will be cancelled
            _ = sub  # keep reference

    fast_task = asyncio.create_task(fast_consumer())
    slow_task = asyncio.create_task(slow_consumer())
    while bus.subscriber_count < 2:
        await asyncio.sleep(0)

    with caplog.at_level(logging.WARNING, logger="robot.core.bus"):
        # Publish 10 events with NO yields between them. Slow consumer's
        # 2-deep queue must drop-oldest. Each publish must return promptly.
        for i in range(10):
            await asyncio.wait_for(
                bus.publish(Heartbeat(source="test", interval_s=float(i))),
                timeout=0.5,
            )

        await asyncio.wait_for(fast_task, timeout=1.0)
        slow_task.cancel()
        try:
            await slow_task
        except asyncio.CancelledError:
            pass

    # Fast consumer got all 10 events in order (its queue had room).
    assert len(fast_received) == 10
    assert [e.interval_s for e in fast_received] == [float(i) for i in range(10)]
    # Drop-oldest fired for the slow consumer (8 times: 10 events into a
    # 2-deep queue with no reader).
    drop_msgs = [r for r in caplog.records if "dropped oldest" in r.message]
    assert len(drop_msgs) == 8, f"expected 8 drops, got {len(drop_msgs)}"


@pytest.mark.asyncio
async def test_interleaved_publish_consume_no_deadlock():
    """Stress-shape: alternate publishing and consuming many events without
    any single party getting ahead enough to wedge the other."""
    bus = Bus(queue_size=4)
    received = []

    async def consumer():
        async with bus.subscribe() as sub:
            async for event in sub:
                received.append(event)
                if len(received) == 50:
                    break

    consumer_task = asyncio.create_task(consumer())
    while bus.subscriber_count < 1:
        await asyncio.sleep(0)

    for i in range(50):
        await bus.publish(BrainToken(source="brain", text=str(i)))
        # Yield so the consumer can drain.
        await asyncio.sleep(0)

    await asyncio.wait_for(consumer_task, timeout=2.0)
    assert len(received) == 50
    assert [e.text for e in received] == [str(i) for i in range(50)]


# ---------- aclose semantics ----------


@pytest.mark.asyncio
async def test_aclose_wakes_blocked_iterator():
    """Iterator blocked on queue.get() must exit cleanly when aclose runs."""
    bus = Bus()

    async def consumer(sub):
        async for _ in sub:
            return  # any event wakes us
        return "exited cleanly"

    sub_cm = bus.subscribe()
    sub = await sub_cm.__aenter__()
    consumer_task = asyncio.create_task(consumer(sub))
    await asyncio.sleep(0)  # park consumer on queue.get()

    # Close without ever publishing. Consumer should wake and exit.
    await sub_cm.__aexit__(None, None, None)
    await asyncio.wait_for(consumer_task, timeout=1.0)
