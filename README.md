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
- **Pre-integrated face embeddings** using InsightFace `buffalo_l` (ResNet50, 512-dimensional vectors)
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
| **Face Embedding**            | 512-d L2-normalized embeddings via InsightFace `buffalo_l` (ResNet50)                    |
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
  - **Face Embedding** — InsightFace buffalo_l (640×640 detect → 512-d output)
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

ZepIris runs as **two services**: the main API (port 8000) and an ML inference
microservice (port 8001). You can run them with **Docker Compose** (recommended)
or **locally with Poetry**.

### Prerequisites

| Requirement   | Notes                                                                       |
| ------------- | --------------------------------------------------------------------------- |
| **Hardware**  | 2 vCPU / **8 GB RAM** minimum, **30 GB+ disk** (PyTorch image + models)     |
| **Docker**    | Docker Engine 24+ and the Compose v2 plugin (`docker compose`)              |
| **Python**    | 3.10–3.14 (only for the local/Poetry path)                                  |
| **Poetry**    | 2.x (only for the local/Poetry path) — [install](https://python-poetry.org/docs/#installation) |
| **Internet**  | Outbound required: the ML service downloads InsightFace `buffalo_l` (~330 MB) on first run; the API fetches S3 reference URLs at request time |

> The recognition model (`buffalo_l`, ResNet50) and all ML deps (PyTorch, OpenCV,
> InsightFace, ONNX Runtime) are installed **inside the Docker images** — you do not
> install them by hand for the Docker path.

---

### Option A — Run locally with Docker Compose (recommended)

```bash
# 1. Clone
git clone <repository-url>
cd zepiris

# 2. Build + start both services (first build pulls PyTorch — a few minutes)
docker compose up -d --build

# 3. Wait for health (ml-inference downloads buffalo_l on first boot)
docker compose logs -f ml-inference     # Ctrl+C when "Application startup complete"
curl http://localhost:8000/healthz       # {"status":"ok"}

# 4. Open the API docs
#    Docs:  http://localhost:8000/docs

# Stop
docker compose down
```

Images are tagged `zepiris-api` and `zepiris-ml-inference` (the Compose file sets
`name: zepiris`).

---

### Option B — Run locally with Poetry (no Docker)

Installs Python deps into a local venv and runs both services. Useful for development.

```bash
# 1. System libraries OpenCV/InsightFace need (macOS: skip — wheels are self-contained)
#    Debian/Ubuntu:
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 g++ curl

# 2. Install Python dependencies (incl. ML extras: torch, insightface, onnxruntime)
poetry install --extras ml

# 3. Run BOTH services with one script (starts ml-inference, waits, starts api)
chmod +x run_local.sh
./run_local.sh
#    API: http://localhost:8000/docs   ·   stop with Ctrl+C (kills both)
```

`run_local.sh` points the model paths at the local `./models/` directory and wires
the API to the local ML service automatically. To run the services by hand instead:

```bash
# terminal 1 — ML inference
make run-ml
# terminal 2 — main API (points at the local ML service)
make run-api-local
```

---

### Option C — Deploy on EC2 (Docker Compose)

```bash
# --- Launch an instance ---
# Ubuntu 22.04 LTS · t3.large (8 GB RAM) min · 30–40 GB gp3 root disk · public IP
# Security Group inbound: 22 (your IP), 8000 (API). Leave 8001 closed (internal).

# --- 1. Install Docker + Compose plugin ---
sudo apt-get update && sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

# --- 2. Clone + build (build serially to stay light on disk) ---
git clone <repository-url> zepiris && cd zepiris
ls models/                              # must contain *.pth AND *.onnx
docker compose build ml-inference
docker compose build api
docker compose up -d

# --- 3. Verify ---
docker compose ps                       # both "healthy"
docker compose logs -f ml-inference     # buffalo_l downloads once (~330 MB)
curl http://localhost:8000/healthz      # {"status":"ok"}
# From your laptop: curl http://<EC2_PUBLIC_IP>:8000/healthz
```

`restart: always` is set, so both containers survive reboots. To update later:
`git pull && docker compose up -d --build`.

**EC2 gotchas** (full guide in [docs/EC2_DEPLOY.md](docs/EC2_DEPLOY.md)):

- **Disk too small** — the default 8 GB root volume can't fit PyTorch. Use **30–40 GB**;
  if you hit `No space left on device`, resize the EBS volume then
  `sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1`.
- **InsightFace permission error** — the `insightface_cache` volume is created
  root-owned on older deploys; if the ML log shows `Permission denied: .../.insightface`,
  run `docker compose exec -u root ml-inference chown -R appuser:appuser /home/appuser/.insightface`
  then `docker compose restart ml-inference`.

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

#### Face Match (1:1) — selfie vs a reference photo

The selfie is sent as **base64** (`selfie_b64`); the reference is an **S3 URL** (`s3_url`).

```bash
# selfie_b64 = base64 of the live capture; s3_url = reference photo
curl -X POST http://localhost:8000/v1/faces/facematch/verify \
  -F "selfie_b64=$(base64 -i selfie.jpg)" \
  -F "s3_url=https://your-bucket.s3.amazonaws.com/ref.jpg?X-Amz-Signature=..."
```

#### Doc Match (1:1) — selfie vs the photo on an Aadhaar/PAN

```bash
# selfie_b64 = base64 of the live capture; s3_url = ID document image
curl -X POST http://localhost:8000/v1/faces/docmatch/verify \
  -F "selfie_b64=$(base64 -i selfie.jpg)" \
  -F "s3_url=https://your-bucket.s3.amazonaws.com/aadhaar.jpg"
```

**Response (both endpoints):**

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

| Field        | facematch | docmatch | Notes                                                                       |
| ------------ | --------- | -------- | --------------------------------------------------------------------------- |
| `selfie_b64` | required  | required | Live selfie as base64 (bare or `data:` URI), max 5 MB — liveness/IQA run here |
| `s3_url`     | required  | required | Reference/document image; presigned/public URL, fetched server-side         |
| `threshold`  | optional  | optional | Cosine cutoff. Defaults: face `0.5`, doc `0.4`                               |

- The selfie is always a live, online capture (base64), so liveness/quality gates
  always run on it; the reference/document (S3 URL) is only embedded.
- `200 OK` with flags (and `score: null`) on early exit — decode failure, liveness
  failure, IQA not passed, or no face in the selfie.
- `400 Bad Request` for missing/invalid inputs and reference problems:
  `invalid_base64`, `reference_image_fetch_failed`, `reference_image_decode_failed`,
  `reference_face_not_detected`.
- `POST /v1/faces/verify` remains as a deprecated alias of `facematch/verify`.

> **Accuracy & speed:** the recognition model defaults to `antelopev2` (ResNet100,
> higher accuracy). Set `ML_SERVICE_FACE_MODEL_NAME=buffalo_l` for faster CPU
> inference at slightly lower accuracy. Document photos embed weaker than selfies,
> which is why docmatch uses a more lenient default threshold.

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for every early-exit and error variant.

#### Detect a Face

Lightweight face-presence check — is a face visible, and where:

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
| `ZEPIRIS_VERIFY_THRESHOLD` | `0.5` | Default COSINE match threshold for **facematch** |
| `ZEPIRIS_DOC_VERIFY_THRESHOLD` | `0.4` | Default COSINE match threshold for **docmatch** (more lenient) |
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
| `ML_SERVICE_FACE_MODEL_NAME`         | `buffalo_l`                    | Recognition model. `buffalo_l` loads reliably; `antelopev2` is more accurate but fails to load with this image's InsightFace build |
| `ML_SERVICE_FACE_DET_THRESH`         | `0.4`                          | Detector confidence; lower detects small/printed faces (Aadhaar/PAN) |
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
- **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** — Throughput, model tiers, and capacity sizing
- **[docs/EC2_AUTOSCALING.md](docs/EC2_AUTOSCALING.md)** — EC2 autoscaling deployment plan
- **[docs/CPU_VS_GPU.md](docs/CPU_VS_GPU.md)** — Compute choice and cost at 100 / 500 / 1000 req/s
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
