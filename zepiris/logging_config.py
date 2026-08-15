"""Logging setup shared by both services.

Uvicorn configures only its own loggers, so application loggers propagate to a
root that has no handler and their records are silently dropped. That costs
nothing until an instance misbehaves in production and the lines that would
explain it — which tier loaded, what the concurrency limit resolved to, whether
warm-up ran — were never emitted.
"""

from __future__ import annotations

import logging
import os


def configure_logging(default_level: str = "INFO") -> None:
    """Attach a stdout handler to the root logger, once.

    Level comes from ``LOG_LEVEL`` so it can be raised on a single instance
    without a redeploy. Idempotent: uvicorn's own handlers are left alone.
    """
    level_name = os.environ.get("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, "_zepiris", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        handler._zepiris = True  # type: ignore[attr-defined]
        root.addHandler(handler)
