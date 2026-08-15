"""Admission control for CPU-bound inference.

Model inference is CPU-bound, so the number of requests that can make progress
at once is the core count — not the number of connections the server will
accept. Without a limit, 100 simultaneous requests all enter the thread pool,
each gets a fraction of a core, and every one of them misses its deadline; the
server looks busy and returns nothing on time.

Bounding concurrency at roughly the core count means requests either run at full
speed or wait their turn, and the ones that would have missed their deadline
anyway are rejected quickly with 503 instead of consuming CPU they cannot use.
Total throughput is unchanged — it is fixed by the hardware — but latency stops
degrading for everyone at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class InferenceLimiter:
    """Caps in-flight inferences, shedding load past a bounded wait.

    Attributes:
        limit: Maximum concurrent inferences.
        queue_timeout: How long a request may wait for a slot before it is shed.
    """

    def __init__(self, limit: int, queue_timeout: float) -> None:
        self.limit = limit
        self.queue_timeout = queue_timeout
        self._sem = asyncio.Semaphore(limit)
        self._waiting = 0
        self._active = 0
        self._shed = 0

    @classmethod
    def from_settings(cls, limit: int, queue_timeout: float) -> InferenceLimiter:
        """Build a limiter, deriving the cap from the CPU count when unset.

        The default leaves one core's worth of headroom for the event loop and
        HTTP handling, so inference never fully starves the process of the
        ability to accept and shed traffic.
        """
        if limit <= 0:
            limit = max(1, (os.cpu_count() or 2) - 1)
        return cls(limit, queue_timeout)

    @property
    def waiting(self) -> int:
        """Requests currently queued for a slot — the backlog depth."""
        return self._waiting

    @property
    def active(self) -> int:
        """Requests currently holding a slot — inferences in flight."""
        return self._active

    def snapshot(self) -> dict[str, float | int]:
        """Current saturation, for health reporting and autoscaling.

        ``queue_depth`` is the signal worth scaling on: ``active`` saturates at
        the limit and stops rising however much load arrives, so it cannot
        distinguish "exactly busy" from "badly overloaded". Anything queued means
        requests are waiting on CPU that does not exist yet.
        """
        return {
            "limit": self.limit,
            "active": self._active,
            "queue_depth": self._waiting,
            "utilization": round(self._active / self.limit, 3) if self.limit else 0.0,
            "shed_total": self._shed,
        }

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold an inference slot, or raise 503 if none frees up in time."""
        self._waiting += 1
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.queue_timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            self._shed += 1
            logger.warning(
                "Shedding request: no inference slot within %.1fs (limit=%d, waiting=%d)",
                self.queue_timeout,
                self.limit,
                self._waiting,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "inference_overloaded",
                    "limit": self.limit,
                    "waiting": self._waiting,
                    "hint": "Reduce concurrency or add ML service capacity.",
                },
                headers={"Retry-After": "1"},
            ) from exc
        finally:
            self._waiting -= 1

        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            self._sem.release()
