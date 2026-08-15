# ZepIris — Configuration Guide

Complete configuration reference for ZepIris services.

## Environment Variables

All main-API configuration uses the `ZEPIRIS_` prefix. Values can be set via:
1. Environment variables: `export ZEPIRIS_VERIFY_THRESHOLD=0.5`
2. `.env` file: Create `.env` in project root
3. Python code: Edit `zepiris/config.py`

## API Configuration

### ZEPIRIS_API_TITLE
**Type**: `str`
**Default**: `"ZepIris"`
**Description**: API service name (displayed in OpenAPI docs)

```env
ZEPIRIS_API_TITLE=ZepIris Face Auth
```

### ZEPIRIS_API_VERSION
**Type**: `str`
**Default**: `"1.0.0"`
**Description**: API version (follows semantic versioning)

```env
ZEPIRIS_API_VERSION=1.0.0
```

## Verification Configuration

### ZEPIRIS_VERIFY_THRESHOLD
**Type**: `float`
**Default**: `0.5`
**Description**: Default cosine-similarity match threshold for `POST /v1/faces/verify`.
A request is a match when `score >= threshold`. Can be overridden per request via
the `threshold` form field.

```env
ZEPIRIS_VERIFY_THRESHOLD=0.5
```

### ZEPIRIS_DOC_MIN_SHARPNESS
**Type**: `float`
**Default**: `0.0` (disabled — report only, never reject)
**Description**: Minimum sharpness (variance-of-Laplacian) for the face extracted
from a document in `POST /v1/faces/docmatch/verify`. Blurry phone captures of a
card embed poorly and silently drag the match score down. The extracted face's
sharpness is always reported in the response `documentFace` block; when this is
set `> 0`, a face below it is rejected with `422 document_too_blurry` so the
caller can request a clearer photo. A crisp ID photo typically scores `> 100`;
observed blurry captures fall in the `5–20` range, so a threshold around `25–40`
is a reasonable starting point.

```env
ZEPIRIS_DOC_MIN_SHARPNESS=30
```

### ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS
**Type**: `float`
**Default**: `10.0`
**Description**: HTTP timeout (seconds) when fetching an image from a supplied
S3 URL (any of `source_selfie_s3`, `face_check_s3`, `doc_check_s3`). If the
fetch exceeds this, the request fails with `reference_image_fetch_failed`
(HTTP 400).

```env
ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS=10.0
```

### ZEPIRIS_REFERENCE_MAX_BYTES
**Type**: `int`
**Default**: `5242880` (5 MB)
**Description**: Maximum allowed size (bytes) of the fetched reference image. A
larger body fails with `reference_image_fetch_failed` (HTTP 400).

```env
ZEPIRIS_REFERENCE_MAX_BYTES=5242880
```

## ML Inference Configuration

### ZEPIRIS_ML_INFERENCE_SERVICE_URL
**Type**: `str` (**required**)
**Description**: Base URL of the ML inference microservice. The main API **always**
calls it for **`POST /v1/iqa/assess`** (NSFW, spoof, blur) and **`POST /v1/face/embed`**
(face embedding) after encoding images as base64 JPEG.

If `ZEPIRIS_ML_INFERENCE_SERVICE_URL` is empty, **`ML_INFERENCE_SERVICE_URL`**
(no `ZEPIRIS_` prefix) is used instead. One of these must be set or **startup fails**.

```env
# Docker Compose (service name)
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://ml-inference:8001

# API on host, ML published on 8001
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001
```

### ZEPIRIS_ML_INFERENCE_TIMEOUT_SECONDS
**Type**: `float`
**Default**: `60.0`
**Description**: Per-request HTTP timeout (seconds) for calls to the ML inference
service. CPU embedding (antelopev2/ResNet100) plus the multi-pass face-detection
fallback cascade can far exceed httpx's 5s default on hard document images, so
keep this generous.

```env
ZEPIRIS_ML_INFERENCE_TIMEOUT_SECONDS=60.0
```

## Adaptive Learning (online threshold calibration)

The recognition model is never retrained in production; what keeps learning is
the **decision threshold**, per document type (`aadhaar`, `pan`, ...). Every
scored verification logs its match score (no images, no PII); operator
feedback via `POST /v1/faces/feedback` labels those scores, and the threshold
for that document type is re-fit to meet the false-accept-rate target. State
is plain JSONL/JSON under `ZEPIRIS_LEARNING_DIR`.

### ZEPIRIS_LEARNING_ENABLED
**Type**: `bool`
**Default**: `true`
**Description**: Master switch. When `false`, nothing is logged and learned
thresholds are ignored.

### ZEPIRIS_LEARNING_DIR
**Type**: `str`
**Default**: `learning`
**Description**: Directory for `samples.jsonl`, `feedback.jsonl`, and
`thresholds.json`. Delete it to reset all learned state.

### ZEPIRIS_LEARNING_MIN_GENUINE / ZEPIRIS_LEARNING_MIN_IMPOSTOR
**Type**: `int`
**Default**: `20` / `20`
**Description**: Labelled genuine / impostor outcomes required for a document
type before its threshold is learned; until then the configured default
threshold applies.

### ZEPIRIS_LEARNING_FAR_TARGET
**Type**: `float`
**Default**: `0.01`
**Description**: Maximum false-accept rate the fitted threshold may allow on
the labelled data. The fit maximizes true-accepts subject to this cap, and the
result is always clamped to `[0.25, 0.70]` so bad labels cannot push the
system into absurd behavior.

```env
ZEPIRIS_LEARNING_ENABLED=true
ZEPIRIS_LEARNING_DIR=learning
ZEPIRIS_LEARNING_MIN_GENUINE=20
ZEPIRIS_LEARNING_MIN_IMPOSTOR=20
ZEPIRIS_LEARNING_FAR_TARGET=0.01
```

## Image quality (IQA)

There are **no** `ZEPIRIS_IQA_*` knobs on the main API. Image quality is
determined entirely by the **ml-inference** service (`POST /v1/iqa/assess`). Tune
blur, NSFW, and spoof behavior with **`ML_SERVICE_*`** environment variables on
that container (see `zepiris/ml_inference/app.py` and `.env.example`).

## Configuration Presets

### Development Preset
```env
# .env for local development
ZEPIRIS_VERIFY_THRESHOLD=0.5
ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS=10.0
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001
```

### Production Preset
```env
# .env for production
ZEPIRIS_VERIFY_THRESHOLD=0.6
ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS=8.0
ZEPIRIS_REFERENCE_MAX_BYTES=5242880
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://ml-inference.internal:8001
```

### Kubernetes Preset
```env
# .env for Kubernetes deployment
ZEPIRIS_VERIFY_THRESHOLD=0.5
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://ml-inference.default.svc.cluster.local:8001
```

## Configuration Validation

On startup, ZepIris validates:
- ✓ `ZEPIRIS_ML_INFERENCE_SERVICE_URL` (or `ML_INFERENCE_SERVICE_URL`) is set
- ✓ Verify threshold and reference limits are positive

**Validation Errors** appear in logs:
```
[ERROR] ML_INFERENCE_SERVICE_URL is required but not set
```

## Performance Tuning

### Stricter matching
```env
ZEPIRIS_VERIFY_THRESHOLD=0.7   # require higher cosine similarity to accept
```

### Looser matching
```env
ZEPIRIS_VERIFY_THRESHOLD=0.4   # accept lower cosine similarity
```

### Image quality
Adjust blur/NSFW/spoof sensitivity via `ML_SERVICE_*` on the ml-inference
container, e.g.:

```env
ML_SERVICE_BLUR_THRESHOLD=0.85
```
(Set on the `ml-inference` service / its `.env`, not on the main API.)

## Troubleshooting Configuration

### Issue: Main API fails to start
```bash
# ML inference URL is required — make sure it is set
echo $ZEPIRIS_ML_INFERENCE_SERVICE_URL
export ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001
```

### Issue: `reference_image_fetch_failed`
- The supplied S3 URL (`source_selfie_s3` / `face_check_s3` / `doc_check_s3`)
  may be expired, unreachable, returned 404, or the body exceeded
  `ZEPIRIS_REFERENCE_MAX_BYTES`.
- The fetch may have exceeded `ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS`. Increase
  the timeout if your reference store is slow:

```env
ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS=20.0
```

### Issue: Live photo failing IQA
IQA is enforced by **ml-inference** (`/v1/iqa/assess`). Loosen thresholds there, e.g.:

```env
ML_SERVICE_BLUR_THRESHOLD=0.85
```
(Set on the `ml-inference` service / its `.env`, not on the main API.)

## Environment Variable Priority

Settings are loaded in order (highest priority first):
1. Environment variables (`export ZEPIRIS_...`)
2. `.env` file
3. Docker secrets (Kubernetes)
4. Hardcoded defaults in `config.py`

Example:
```bash
# Override with environment variable
export ZEPIRIS_VERIFY_THRESHOLD=0.6
docker-compose up
```

## Configuration File Format

### .env File Format
```env
# Comments start with #
ZEPIRIS_API_TITLE=ZepIris

# Quote if value contains spaces
ZEPIRIS_API_TITLE="My ZepIris Service"

# Floats
ZEPIRIS_VERIFY_THRESHOLD=0.5
ZEPIRIS_REFERENCE_FETCH_TIMEOUT_SECONDS=10.0

# Integers
ZEPIRIS_REFERENCE_MAX_BYTES=5242880
```

## Loading Custom Configuration

### Programmatically
```python
from zepiris.config import Settings

settings = Settings(
    verify_threshold=0.6,
    ml_inference_service_url="http://localhost:8001",
)
```

### From File
```bash
# Create custom.env
source custom.env
poetry run uvicorn zepiris.main:app
```

## Security Best Practices

1. **Never commit `.env` with real credentials**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use presigned, short-lived S3 URLs** for reference images so they cannot be
   replayed after expiry.

3. **Use TLS/HTTPS in production** for both the API and the reference image URLs.

4. **Restrict network access**
   - ml-inference: only reachable from the main API.

## See Also

- [README.md](../README.md) - Project overview
- [SETUP_GUIDE.md](../SETUP_GUIDE.md) - Setup instructions
- [API_REFERENCE.md](API_REFERENCE.md) - API endpoints
