# ML inference service — PyTorch + InsightFace. Local dev: `poetry install --extras ml`.
# Build: docker build -f ml_inference.Dockerfile -t zepiris-ml-service .
#
# Runtime deps (including torch) come from poetry.lock. Place model weights in models/
# at build time (see README / ML_INFERENCE_SERVICE.md).

FROM python:3.12.3-slim

LABEL org.opencontainers.image.title="ZepIris ML Inference"
LABEL org.opencontainers.image.description="Face IQA, embedding, and detection microservice"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# OpenCV / InsightFace runtime libs + g++ for native wheels + curl for HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        g++ \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock poetry.toml README.md ./
COPY zepiris/ zepiris/

ARG POETRY_VERSION=2.1.3
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && poetry install --only main --extras ml --no-root --no-ansi \
    && pip install --no-cache-dir . \
    && pip uninstall -y poetry

RUN apt-get update \
    && apt-get purge -y g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# GPU builds: swap the CPU wheel for onnxruntime-gpu plus the CUDA 12 / cuDNN 9
# libraries it links against (the [cuda,cudnn] extras pull them from PyPI, so the
# base image needs no CUDA toolkit — only the host driver, via the NVIDIA
# container runtime). Empty, the default, leaves the CPU wheel alone.
#
# The CPU wheel has no CUDAExecutionProvider, so a GPU instance built without
# this silently runs inference on the host cores. See docs/CPU_VS_GPU.md §5.2.
ARG ONNXRUNTIME_GPU_VERSION=""
RUN if [ -n "$ONNXRUNTIME_GPU_VERSION" ]; then \
        pip uninstall -y onnxruntime \
        && pip install --no-cache-dir \
            "onnxruntime-gpu[cuda,cudnn]==${ONNXRUNTIME_GPU_VERSION}" \
        && find /usr/local/lib/python3.12/site-packages/nvidia -type d -name lib \
            > /etc/ld.so.conf.d/nvidia-pip-wheels.conf \
        && ldconfig; \
    fi

COPY models/ /app/models/

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /home/appuser/.insightface \
    && chown -R appuser:appuser /app /home/appuser/.insightface

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8001/healthz || exit 1

CMD ["python", "-m", "uvicorn", "zepiris.ml_inference.app:app", "--host", "0.0.0.0", "--port", "8001"]
