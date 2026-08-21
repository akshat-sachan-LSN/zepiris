"""Admission control for the API process: bound in-flight requests, shed the rest.

This exists because the obvious knob — uvicorn's ``limit_concurrency`` — counts
the wrong thing. Its check is ``len(connections) >= limit or len(tasks) >=
limit``, and ``connections`` includes **idle keep-alive connections**: a load
balancer's warm pool, or a few hundred mobile clients that simply have a
connection open, trips the limit while the box is doing nothing, and every
request is refused. That is exactly how the one historical run that enabled it
produced a wall of errors with nothing served.

What actually needs bounding is in-flight *requests*: under overload the queue
otherwise forms in the connection backlog, where nothing times out and no
limiter can see it — measured there, 150 req/s offered gave p95 13.9s with
zero shed, and a 100 req/s soak showed p99 24s with zero 503s. Counting
requests at the ASGI layer bounds that queue precisely, whatever the
connection count, and turns overflow into an immediate, well-formed 503 that
costs the event loop almost nothing to produce.

Health and metrics endpoints bypass the limiter: an instance that is shedding
is exactly the instance the operator needs to be able to look at.
"""

from __future__ import annotations

import json


class ApiConcurrencyLimiter:
    """In-flight request counter shared between the middleware and /metrics.

    No lock: it is only touched from the event loop, and every update is a
    single increment between awaits.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self.shed = 0

    @property
    def full(self) -> bool:
        return self.active >= self.limit

    def snapshot(self) -> dict[str, float | int]:
        """Saturation for /metrics — ``active`` near ``limit`` with ``shed``
        rising is the signal that this instance needs a sibling, not tuning."""
        return {
            "limit": self.limit,
            "active": self.active,
            "utilization": round(self.active / self.limit, 3) if self.limit else 0.0,
            "shed_total": self.shed,
        }


#: Paths that must answer even while the limiter is shedding.
_BYPASS_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})

_SHED_BODY = json.dumps(
    {
        "detail": {
            "message": "api_overloaded",
            "hint": "The instance is at its in-flight request limit; retry shortly.",
        }
    }
).encode()

_SHED_HEADERS = [
    (b"content-type", b"application/json"),
    (b"content-length", str(len(_SHED_BODY)).encode()),
    (b"retry-after", b"1"),
]


class ConcurrencyLimitMiddleware:
    """Pure-ASGI in-flight request cap — cheap enough for the hot path.

    Deliberately not a ``BaseHTTPMiddleware``: that wrapper spawns a task and
    an anyio stream pair per request, which is real overhead at 100+ req/s for
    what is otherwise one integer compare.
    """

    def __init__(self, app, limiter: ApiConcurrencyLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"] in _BYPASS_PATHS:
            await self.app(scope, receive, send)
            return

        if self.limiter.full:
            self.limiter.shed += 1
            await send(
                {"type": "http.response.start", "status": 503, "headers": _SHED_HEADERS}
            )
            await send({"type": "http.response.body", "body": _SHED_BODY})
            return

        self.limiter.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.limiter.active -= 1
