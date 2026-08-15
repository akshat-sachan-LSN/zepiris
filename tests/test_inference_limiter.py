"""Admission control: run at full speed or be told no, never crawl."""

import asyncio

import pytest
from fastapi import HTTPException

from zepiris.ml_inference.concurrency import InferenceLimiter


def test_limits_concurrent_holders() -> None:
    limiter = InferenceLimiter(limit=3, queue_timeout=5.0)
    peak = 0
    active = 0

    async def worker() -> None:
        nonlocal peak, active
        async with limiter.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    asyncio.run(_gather(worker, 12))
    assert peak == 3


def test_sheds_when_queue_wait_exceeded() -> None:
    """A request that cannot get a slot in time is rejected, not left to crawl."""
    limiter = InferenceLimiter(limit=1, queue_timeout=0.05)

    async def scenario() -> None:
        async def hold() -> None:
            async with limiter.slot():
                await asyncio.sleep(0.3)

        holder = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        with pytest.raises(HTTPException) as exc:
            async with limiter.slot():
                pass
        assert exc.value.status_code == 503
        assert exc.value.detail["message"] == "inference_overloaded"
        await holder

    asyncio.run(scenario())


def test_slot_is_released_when_the_work_raises() -> None:
    """A failing inference must not leak its slot, or capacity bleeds away."""
    limiter = InferenceLimiter(limit=1, queue_timeout=0.5)

    async def scenario() -> None:
        with pytest.raises(RuntimeError):
            async with limiter.slot():
                raise RuntimeError("inference blew up")
        # The slot is free again, so this acquires immediately.
        async with limiter.slot():
            pass

    asyncio.run(scenario())


def test_waiting_count_tracks_backlog() -> None:
    limiter = InferenceLimiter(limit=1, queue_timeout=5.0)

    async def scenario() -> None:
        async def hold() -> None:
            async with limiter.slot():
                await asyncio.sleep(0.1)

        first = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        queued = [asyncio.create_task(hold()) for _ in range(4)]
        await asyncio.sleep(0.01)
        assert limiter.waiting == 4
        await asyncio.gather(first, *queued)
        assert limiter.waiting == 0

    asyncio.run(scenario())


def test_derives_limit_from_cpu_count_when_unset() -> None:
    limiter = InferenceLimiter.from_settings(0, 10.0)
    assert limiter.limit >= 1


async def _gather(worker, n: int) -> None:
    await asyncio.gather(*(worker() for _ in range(n)))
