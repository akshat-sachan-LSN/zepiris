"""The API's in-flight admission control (zepiris/api/concurrency.py).

These tests pin the semantics that uvicorn's limit_concurrency gets wrong:
the cap must count in-flight *requests*, so idle keep-alive connections can
never shed a healthy instance, and overflow must arrive as an immediate,
well-formed 503.
"""

import asyncio

import httpx
from fastapi import FastAPI

from zepiris.api.concurrency import ApiConcurrencyLimiter, ConcurrencyLimitMiddleware


def _app(limit: int, gate: asyncio.Event) -> tuple[FastAPI, ApiConcurrencyLimiter]:
    app = FastAPI()
    limiter = ApiConcurrencyLimiter(limit)
    app.state.api_limiter = limiter
    app.add_middleware(ConcurrencyLimitMiddleware, limiter=limiter)

    @app.get("/work")
    async def work() -> dict:
        await gate.wait()
        return {"ok": True}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app, limiter


def test_overflow_sheds_503_and_recovers() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        app, limiter = _app(1, gate)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            first = asyncio.create_task(client.get("/work"))
            while limiter.active < 1:  # let the first request occupy the slot
                await asyncio.sleep(0)

            shed = await client.get("/work")
            assert shed.status_code == 503
            assert shed.headers["retry-after"] == "1"
            assert shed.json()["detail"]["message"] == "api_overloaded"

            gate.set()
            assert (await first).status_code == 200
            # The slot freed: the same request now succeeds.
            assert (await client.get("/work")).status_code == 200
        assert limiter.shed == 1
        assert limiter.active == 0

    asyncio.run(scenario())


def test_health_bypasses_the_limiter() -> None:
    """An instance that is shedding is exactly the one the operator must be
    able to look at."""

    async def scenario() -> None:
        gate = asyncio.Event()
        app, limiter = _app(1, gate)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            first = asyncio.create_task(client.get("/work"))
            while limiter.active < 1:
                await asyncio.sleep(0)
            assert (await client.get("/healthz")).status_code == 200
            gate.set()
            await first

    asyncio.run(scenario())


def test_slot_is_released_when_the_handler_raises() -> None:
    async def scenario() -> None:
        app = FastAPI()
        limiter = ApiConcurrencyLimiter(1)
        app.add_middleware(ConcurrencyLimitMiddleware, limiter=limiter)

        @app.get("/boom")
        async def boom() -> dict:
            raise RuntimeError("kaboom")

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            assert (await client.get("/boom")).status_code == 500
        assert limiter.active == 0

    asyncio.run(scenario())


def test_zero_limit_is_disabled_via_create_app_wiring() -> None:
    """create_app only installs the middleware when the setting is > 0; the
    limiter itself treats limit=0 as permanently full, so wiring must gate it."""
    limiter = ApiConcurrencyLimiter(0)
    assert limiter.full
    assert limiter.snapshot()["utilization"] == 0.0
