# ZepIris

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127%2B-green)](https://fastapi.tiangolo.com/)
[![Poetry](https://img.shields.io/badge/Poetry-2.x-blueviolet)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👁️ ZepIris — Face Authentication at Scale

**No OTPs. No registers. No buddy punching. Just a selfie.**

ZepIris is Zepto's purpose-built face authentication platform — open-sourced for teams running identity verification at operational scale.

It is a **stateless 1:1 face-verification** service: send a live selfie plus a reference image (as an S3 URL) and get a match decision back. It handles the full pipeline on the live photo — face detection, liveness/IQA, embedding generation — then embeds the reference image and compares the two with cosine similarity. Designed to work on budget smartphones, in low light, under high concurrency. **Nothing is persisted.**

If you're running attendance or identity workflows at scale and don't want to stitch together multiple vendors, this is it.

**Current version:** v1.0.0. URL paths use `/v1/...` for the HTTP API; OpenAPI `info.version` and the Python package follow semver (`pyproject.toml` / `zepiris.version`).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Support & Community](#support--community)

---

## Overview

ZepIris simplifies face verification workflows by providing:

- **Stateless 1:1 verification** — compare a live photo against a reference image; nothing is stored
- **Pre-integrated face embeddings** using InsightFace's buffalo_l model (512-dimensional vectors)
- **COSINE similarity matching** with a configurable decision threshold
- **Automated content safety checks**: nudity detection, anti-spoofing/liveness, blur detection — all via a dedicated ML inference service
- **Reference image by S3 URL** — supply a presigned/public URL; fetched over plain HTTP, no AWS credentials required
- **Production-ready microservice architecture** with independent scaling for ML inference
- **REST API** with OpenAPI/Swagger auto-documentation and `requestId` traceability

### Use Cases

- **Attendance Tracking** — Verify a live selfie against a stored reference photo
- **Onboarding Workflows** — Identity verification with liveness detection
- **Face-Based Access Control** — 1:1 face matching with content safety validation
- **Quality Assurance** — Automatic detection of low-quality, spoofed, or unsafe images

---

## Features


| Capability                    | Description                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| **Stateless 1:1 Verify**      | Compare a live photo against a reference image; no enrollment, no storage                |
| **Face Embedding**            | Extract 512-dimensional L2-normalized embeddings using InsightFace buffalo_l             |
| **COSINE Matching**           | Configurable per-request decision threshold (`ZEPIRIS_VERIFY_THRESHOLD`)                 |
| **Content Safety (ML)**       | Nudity, spoof/liveness, and blur detection via dedicated ML inference microservice       |
| **Reference by S3 URL**       | Reference image fetched over plain HTTP from a presigned/public URL (no AWS creds)       |
| **Microservice Architecture** | Separate ML inference service (port 8001) scales independently from main API (port 8000) |
| **REST API**                  | FastAPI with OpenAPI/Swagger docs, `requestId` on every response                         |
| **Docker Ready**              | Multi-stage Dockerfiles for both services + Docker Compose (2 containers only)           |
| **Configurable Thresholds**   | Fine-tune quality checks (blur sensitivity, spoof threshold, nudity confidence)          |


---

## Architecture

ZepIris consists of **two independent FastAPI microservices** that communicate via HTTP. There is **no** vector database, object storage, or metadata store — the deployment is just `api` + `ml-inference`.

```
┌─────────────────────────────────────────────────┐
│  Client / Application                           │
└──────────────┬──────────────────────────────────┘
               │ (REST API: live photo + reference S3 URL)
      ┌────────▼──────────────────────────────┐
      │  Main API (port 8000)                 │
      │  ├─ POST /v1/faces/verify             │
      │  ├─ POST /v1/faces/detect             │
      │  ├─ GET  /ui  (single verify page)    │
      │  ├─ GET  /healthz                     │
      │  └─ GET  /readyz                      │
      └───┬───────────────────────────┬───────┘
          │                           │
          │ (plain HTTP GET)          │ (base64 over HTTP)
   ┌──────▼────────┐        ┌─────────▼───────────────────────┐
   │ Reference img │        │ ML Inference (port 8001)         │
   │ via S3 URL    │        │ ├─ POST /v1/embed  (Embedding)   │
   │ (not stored)  │        │ ├─ POST /v1/nudity (Nudity)      │
   └───────────────┘        │ ├─ POST /v1/spoof  (Spoof)       │
                            │ ├─ POST /v1/blur   (Blur)        │
                            │ └─ POST /v1/assess (Combined IQA)│
                            └──────────────────────────────────┘
```

### Main API Service (Port 8000)

**Responsibilities:**

- Handle the stateless verify endpoint and the detect poll under `/v1/faces/`
- Validate the uploaded live photo (size ≤ 5MB; decodable)
- Coordinate with ML inference service for liveness/IQA and embedding extraction
- Fetch the reference image from the supplied `s3_url` over plain HTTP (no AWS creds)
- Compute cosine similarity and return the match decision with `requestId`
- Persist nothing

**Dependencies:**

- FastAPI, Uvicorn, Pydantic
- HTTPx (for ML service communication and reference-image fetch)

### ML Inference Service (Port 8001)

**Responsibilities:**

- Run independent, parallelizable ML workloads
- Maintain 4 PyTorch models in memory:
  - **Face Embedding** — InsightFace buffalo_l (640×640 input → 512-d output)
  - **Nudity Detection** — MobileNetV2 (2-class classifier)
  - **Spoof Detection** — MobileNetV3-Large (liveness detection)
  - **Blur Detection** — ResNet18 (image quality assessment)
- Expose HTTP endpoints for individual or combined inference
- Combined IQA runs all 3 quality checks in parallel via `ThreadPoolExecutor`

**Benefits:**

- Scale independently — run on GPU hardware if needed
- Reuse models across requests — no repeated loading
- Parallel execution — run all 3 quality checks simultaneously

---

## Quick Start

### Prerequisites

- **Python 3.10–3.14** (tested on 3.10–3.14)
- **Poetry 2.x** for dependency management ([install here](https://python-poetry.org/docs/#installation))
- **Docker & Docker Compose** (v1.29+; recommended for all-in-one setup)
- **2GB+ RAM**, **5GB+ free disk space** (for the ML models)

Check your versions:

```bash
python3 --version
poetry --version
```

### Docker Compose (One Command)

For a fully containerized local setup:

```bash
# Clone and navigate
git clone <repository-url>
cd zepiris

# Start both services (api, ml-inference)
docker-compose up -d

# Verify health
curl http://localhost:8000/healthz

# Open API documentation
# Visit: http://localhost:8000/docs

# Stop services
docker-compose down
```

The Compose file sets `name: zepiris`, so images are tagged `zepiris-api` and `zepiris-ml-inference` regardless of clone directory name.

This starts **2 containers**:

- Main API (port 8000)
- ML inference (port 8001)

---

## API Endpoints

### Interactive Documentation

Once running, visit these URLs in your browser:

- **Main API**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
- **ML Inference**: [http://localhost:8001/docs](http://localhost:8001/docs) (Swagger UI)

### Main API (`/v1/faces/`)

#### Health & Readiness

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}

curl http://localhost:8000/readyz
# {"status": "ok"}
```

#### Verify a Face (1:1)

Compare a live photo against a reference image supplied as an S3 URL. Nothing is persisted.

```bash
curl -X POST http://localhost:8000/v1/faces/verify \
  -F "file=@live.jpg" \
  -F "s3_url=https://your-bucket.s3.amazonaws.com/ref.jpg?X-Amz-Signature=..."
```

**Response:**

```json
{
  "requestId": "a1b2c3d4-e5f6-...",
  "imageQualityAssessment": {
    "passed": true,
    "nsfw": {"is_safe": true, "probability": 1.0},
    "spoof": {"is_live": true, "probability": 1.0},
    "blur": {"is_sharp": true, "probability": 0.85}
  },
  "verificationResult": {
    "isMatch": true,
    "score": 0.91,
    "threshold": 0.5
  },
  "faceDetected": true,
  "iqaPassed": true
}
```

**Parameters:**

- `file` (required, form) — live photo upload (JPEG/PNG, max 5 MB)
- `s3_url` (required, form) — presigned/public URL of the reference image
- `threshold` (optional, form) — cosine match threshold; defaults to `ZEPIRIS_VERIFY_THRESHOLD` (0.5)
- Returns `200 OK` with flags (and `score: null`) on early exit — live-photo decode failure, liveness failure, IQA not passed, or no face in the live photo
- Returns `400 Bad Request` for reference-image problems: `reference_image_fetch_failed`, `reference_image_decode_failed`, `reference_face_not_detected`

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for every early-exit and error variant.

#### Detect a Face (UI poll)

Lightweight face-presence check used by the verify UI capture ring:

```bash
curl -X POST http://localhost:8000/v1/faces/detect \
  -F "file=@frame.jpg"
```

**Response:**

```json
{
  "requestId": "f6a7b8c9-d0e1-...",
  "faceDetected": true
}
```

#### Verify UI

A single-page verify UI is served at [http://localhost:8000/ui](http://localhost:8000/ui).

### ML Inference API (`/v1/`)

All POST endpoints accept JSON bodies with base64-encoded images.

#### Base Payload Format

```json
{
  "image_b64": "<base64-encoded image bytes>"
}
```

**Example:** Encode an image to base64:

```bash
base64 -i face.jpg | pbcopy   # macOS
cat face.jpg | base64         # Linux
```

#### Health Check

```bash
curl http://localhost:8001/healthz
# {"status": "ok"}
```

#### Nudity Detection

Check if image contains nudity/NSFW content:

```bash
curl -X POST http://localhost:8001/v1/nudity \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "is_safe": true,
  "probability": 0.02
}
```

#### Spoof Detection

Check if face is real or spoofed/deepfake:

```bash
curl -X POST http://localhost:8001/v1/spoof \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "is_spoof": false,
  "probability": 0.05
}
```

#### Blur Detection

Check if face image is sharp enough:

```bash
curl -X POST http://localhost:8001/v1/blur \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "is_sharp": true,
  "probability": 0.10
}
```

#### Face Embedding

Generate a 512-dimensional face embedding:

```bash
curl -X POST http://localhost:8001/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "face_detected": true,
  "embedding": [0.123, -0.456, 0.789, "..."],
  "embedding_dim": 512
}
```

#### Combined Assessment (IQA)

Run all 3 quality checks in parallel:

```bash
curl -X POST http://localhost:8001/v1/assess \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "passed": true,
  "nudity": {"is_safe": true, "probability": 0.02},
  "spoof": {"is_spoof": false, "probability": 0.05},
  "blur": {"is_sharp": true, "probability": 0.10}
}
```

`passed` is `true` when: `nudity.is_safe AND (NOT spoof.is_spoof) AND blur.is_sharp`.

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize. All settings use environment variable prefixes.

#### Main API Service (`ZEPIRIS_`*)


| Variable | Default | Description |
|----------|---------|-------------|
| `ZEPIRIS_API_TITLE` | `ZepIris` | API title (shown in docs) |
| `ZEPIRIS_API_VERSION` | `1.0.0` | API version (OpenAPI `info.version`) |
| `ZEPIRIS_API_HOST` | `0.0.0.0` | Bind host |
| `ZEPIRIS_API_PORT` | `8000` | Bind port |
| `ZEPIRIS_VERIFY_THRESHOLD` | `0.5` | Default COSINE similarity match threshold for verify |
| `ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS` | `10.0` | HTTP timeout (s) when fetching the reference image from `s3_url` |
| `ZEPIRIS_REFERENCE_MAX_BYTES` | `5242880` | Max size (bytes, 5 MB) of the fetched reference image |
| `ZEPIRIS_ML_INFERENCE_SERVICE_URL` | *(required)* | URL of the ML inference service (e.g. `http://localhost:8001`) |

> **Note:** `ZEPIRIS_ML_INFERENCE_SERVICE_URL` is **required**. The main API will not start without it. Set it to `http://ml-inference:8001` in Docker Compose or `http://localhost:8001` when running locally.

#### ML Inference Service (`ML_SERVICE_`*)


| Variable                             | Default                        | Description                                  |
| ------------------------------------ | ------------------------------ | -------------------------------------------- |
| `ML_SERVICE_HOST`                    | `0.0.0.0`                      | Bind host                                    |
| `ML_SERVICE_PORT`                    | `8001`                         | Bind port                                    |
| `ML_SERVICE_ML_DEVICE`               | `cpu`                          | Inference device: `cpu`, `cuda:0`, `mps`     |
| `ML_SERVICE_NUDITY_LOCAL_MODEL_PATH` | `/app/models/nudity_model.pth` | Nudity model file                            |
| `ML_SERVICE_NUDITY_HF_REPO_ID`       | ``                             | HuggingFace repo for nudity model (optional) |
| `ML_SERVICE_NUDITY_THRESHOLD`        | `0.5`                          | Nudity classification threshold (0–1)        |
| `ML_SERVICE_SPOOF_LOCAL_MODEL_PATH`  | `/app/models/spoof_model.pth`  | Spoof model file                             |
| `ML_SERVICE_SPOOF_HF_REPO_ID`        | ``                             | HuggingFace repo for spoof model (optional)  |
| `ML_SERVICE_SPOOF_THRESHOLD`         | `0.5`                          | Spoof classification threshold (0–1)         |
| `ML_SERVICE_BLUR_LOCAL_MODEL_PATH`   | `/app/models/blur_model.pth`   | Blur model file                              |
| `ML_SERVICE_BLUR_HF_REPO_ID`         | ``                             | HuggingFace repo for blur model (optional)   |
| `ML_SERVICE_BLUR_THRESHOLD`          | `0.5`                          | Blur classification threshold (0–1)          |
| `ML_SERVICE_FACE_EMBEDDING_DIM`      | `512`                          | Face embedding dimension                     |
| `ML_SERVICE_FACE_DETECTION_WIDTH`    | `640`                          | Face detection input width                   |
| `ML_SERVICE_FACE_DETECTION_HEIGHT`   | `640`                          | Face detection input height                  |
| `ML_SERVICE_FACE_AREA_THRESHOLD`     | `0.01`                         | Minimum face area (fraction of image)        |


### Example: GPU Inference

To enable GPU inference, set:

```bash
export ML_SERVICE_ML_DEVICE=cuda:0
poetry run zepiris-ml-inference-api
```

Or in `.env`:

```bash
ML_SERVICE_ML_DEVICE=cuda:0
```

Ensure PyTorch CUDA version matches your GPU driver.

---

## Development

### Project Structure

```
zepiris/
├── pyproject.toml              # Project metadata, dependencies
├── poetry.lock                 # Locked dependency versions
├── poetry.toml                 # Poetry config (in-project .venv)
├── .env.example                # Environment variable template
├── Dockerfile                  # Main service container
├── ml_inference.Dockerfile     # ML service container
├── docker-compose.yml          # 2 services (api, ml-inference)
│
├── zepiris/
│   ├── main.py                 # Main FastAPI app factory + lifespan
│   ├── config.py               # Pydantic settings (ZEPIRIS_* prefix)
│   ├── deps.py                 # FastAPI dependency injection
│   ├── exceptions.py           # Domain exceptions (ReferenceImageError, etc.)
│   ├── exception_handlers.py   # Error response formatting
│   │
│   ├── api/routes/
│   │   ├── __init__.py         # build_api_router()
│   │   ├── face.py             # /v1/faces/ verify, detect
│   │   └── health.py           # GET /healthz, /readyz
│   │
│   ├── ml_inference/
│   │   ├── app.py              # ML service FastAPI app + MLServiceSettings
│   │   ├── routes.py           # ML endpoints (/v1/nudity, /spoof, /blur, /embed, /assess)
│   │   ├── deps.py             # ML service dependency injection (503 if model missing)
│   │   ├── base.py             # Base ModelService + ModelServiceConfig
│   │   ├── face_embedding.py   # FaceEmbeddingService (InsightFace)
│   │   ├── nudity_detection.py # NudityDetectionService (MobileNetV2)
│   │   ├── spoof_detection.py  # SpoofDetectionService (MobileNetV3)
│   │   ├── blur_detection.py   # BlurDetectionService (ResNet18)
│   │   ├── image_quality_assessment.py # Combined IQA (ThreadPoolExecutor)
│   │   └── models/             # PyTorch model definitions
│   │
│   ├── services/
│   │   ├── embedding.py        # FaceEmbeddingProvider (ABC) + MLInferenceEmbeddingService
│   │   ├── iqa.py              # MLInferenceIQAService → HTTP /v1/assess
│   │   ├── s3_fetcher.py       # S3ImageFetcher: fetch reference image from s3_url (plain HTTP)
│   │   ├── similarity.py       # In-process cosine similarity
│   │   └── ml_client.py        # MLInferenceClient (httpx, sync)
│   │
│   └── schemas/
│       ├── face.py             # VerifyResponse, DetectResponse
│       └── ml_inference.py     # FaceEmbeddingResult, IQA result schemas
│
└── models/                     # Pre-trained model weights
    ├── nudity_model.pth
    ├── spoof_model.pth
    └── blur_model.pth
```

### Development Setup

```bash
# Activate virtual environment
source .venv/bin/activate
# or use Poetry shell
poetry shell
```

### Running with Auto-Reload

For local development with hot-reloading:

```bash
# Terminal 1 — ML service with reload
uvicorn zepiris.ml_inference.app:app --reload --host 0.0.0.0 --port 8001

# Terminal 2 — Main API with reload
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001 \
  uvicorn zepiris.main:app --reload --host 0.0.0.0 --port 8000
```

### Adding Dependencies

```bash
# Add a runtime dependency
poetry add httpx-oauth

# Add a development-only tool
poetry add --group dev black ruff pytest

# After changing dependencies, commit both files
git add pyproject.toml poetry.lock
git commit -m "chore: add new dependencies"
```

---

## Testing

ZepIris includes test scripts for both unit and integration testing.

### End-to-End ML Service Test

Tests all ML endpoints with real model inference:

```bash
python scripts/test_ml_service.py \
  --test-image path/to/image.jpg \
  --nudity-model ./models/nudity_model.pth \
  --spoof-model ./models/spoof_model.pth \
  --blur-model ./models/blur_model.pth
```

Or test against a running service:

```bash
python scripts/test_ml_service.py \
  --test-image path/to/image.jpg \
  --no-server --port 8001
```

### Direct Model Testing

Load and test models in-process without HTTP:

```bash
python scripts/test_models.py \
  --test-image path/to/image.jpg \
  --nudity-model ./models/nudity_model.pth \
  --spoof-model ./models/spoof_model.pth \
  --blur-model ./models/blur_model.pth
```

### Manual API Testing

Use `curl`, `httpie`, or Postman to test endpoints. Interactive Swagger docs available at `/docs` on both services.

---

## Troubleshooting

### Poetry Installation Issues

**Problem:** `poetry install` fails with "No file/folder found"

**Solution:** This is expected on first run. Poetry will resolve dependencies. Try again.

**Problem:** Wrong Python version

**Solution:** Point Poetry to the correct interpreter:

```bash
poetry env use /path/to/python3.10
poetry install
```

### Service Connection Issues

**Problem:** Main API can't reach the ML inference service

**Solution:** Verify the ml-inference container is running and healthy:

```bash
# Check both services
docker compose ps

# Check ML inference directly
curl http://localhost:8001/healthz             # ML inference
```

**Problem:** Verify returns `reference_image_fetch_failed`

**Solution:** The `s3_url` may be expired, unreachable, returned 404, exceeded
`ZEPIRIS_REFERENCE_MAX_BYTES`, or timed out (`ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS`).
Re-generate a fresh presigned URL or raise the timeout.

**Problem:** Main API fails to start with "ML_INFERENCE_SERVICE_URL required"

**Solution:** Set the required environment variable:

```bash
export ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001
```

**Problem:** ML service doesn't load models on startup

**Solution:** Check logs for model loading errors:

```bash
docker compose logs ml-inference  # if using Docker Compose
# or
poetry run zepiris-ml-inference-api       # if running locally
```

Ensure model files exist at configured paths. Default: `/app/models/{nudity,spoof,blur}_model.pth`. If a model fails to load, that endpoint returns **503 Service Unavailable**.

### GPU Not Detected

**Problem:** ML service ignores `ML_SERVICE_ML_DEVICE=cuda`

**Solution:**

1. Verify PyTorch CUDA version matches your GPU driver:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

1. If unavailable, reinstall PyTorch with correct CUDA version:

```bash
poetry remove torch torchvision
poetry add torch torchvision --platform linux --python "^3.10"
```

### API Timeouts

**Problem:** Requests hang or timeout to ML service

**Solution:**

- Ensure ML inference service is running and healthy (`GET /healthz`)
- Check if models are still loading (can take 30-60s on first startup)
- Check network connectivity between services
- Monitor resource usage (disk space for model downloads, RAM for inference)

---

## Documentation

- **[LOCAL_SETUP_AND_TEST.md](docs/LOCAL_SETUP_AND_TEST.md)** — Step-by-step local testing guide (20-30 min)
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** — Complete API endpoint reference
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — Environment variables & tuning
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Code guidelines & contribution workflow
- **[.env.example](.env.example)** — Complete environment variable template

### Additional Resources

- [InsightFace Documentation](https://github.com/deepinsight/insightface) — Face embedding & detection
- [FastAPI Best Practices](https://fastapi.tiangolo.com/deployment/concepts/) — Web framework
- [HTTPX Documentation](https://www.python-httpx.org/) — Async/sync HTTP client

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

> The project is released under the [LICENSE](LICENSE) with contribution expectations described in [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Citation

If you use ZepIris in research or production, please cite:

```bibtex
@software{zepiris2026,
  title={ZepIris: Open-source face embedding and content safety microservice},
  author={Zepto Data Science Team},
  year={2026},
  url={https://github.com/zepto-labs/zepiris}
}
```

---

## Acknowledgments

ZepIris stands on the shoulders of excellent open-source projects:

- **[InsightFace](https://github.com/deepinsight/insightface)** — State-of-the-art face embedding models
- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern async Python web framework
- **[PyTorch](https://pytorch.org/)** — Deep learning framework
- **[HTTPX](https://www.python-httpx.org/)** — HTTP client for service-to-service calls

---

## Support & Community

- **Issues:** [GitHub Issues](https://github.com/zepto-labs/zepiris/issues)
- **Discussions:** [GitHub Discussions](https://github.com/zepto-labs/zepiris/discussions)
- **Email:** [opensource@zepto.com](mailto:opensource@zepto.com)

---

**Last Updated:** April 2026
**Status:** v1.0.0
