# ZepIris API — production runtime image (deps from poetry.lock).
# Local development uses Poetry (`.venv/`).
FROM python:3.12.3-slim

LABEL org.opencontainers.image.title="ZepIris API"
LABEL org.opencontainers.image.description="FastAPI face authentication service"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# OpenCV (headless wheel) runtime deps + curl for HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock poetry.toml README.md ./
COPY zepiris/ zepiris/

ARG POETRY_VERSION=2.1.3
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && poetry install --only main --no-root --no-ansi \
    && pip install --no-cache-dir . \
    && pip uninstall -y poetry

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=20s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# Console script rather than a bare uvicorn line so ZEPIRIS_API_WORKERS is
# honoured — a single worker caps this process at one core for the JSON/base64
# work on the request path.
CMD ["zepiris-api"]
