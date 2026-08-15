# ZepIris — API Reference

Complete API specification for the ZepIris stateless 1:1 face-verification service.

**Base URL**: `http://localhost:8000`

---

## Overview

ZepIris is a **stateless 1:1 face-verification** service. It compares a freshly
captured live photo against a single reference image and returns a match
decision. **Nothing is persisted** — there is no enrollment, no database, and no
object storage.

All responses include:
- `requestId` (UUID) for traceability
- `imageQualityAssessment` (when the live photo reaches the IQA stage)
- Standardized flags / error messages

**Image pipeline:** each request compares an **incoming image being verified**
(the *probe*) against the user's **enrolled selfie** (`source_selfie`, the
source of truth stored in your DB). The probe is a live face for `facematch`
(`face_check`) or an ID document for `docmatch` (`doc_check`). Each side accepts
**either inline base64 or an S3 URL** (supply exactly one per side; sending both
is rejected with `400 image_source_invalid`):

| Role                              | base64 field        | S3 URL field        |
|-----------------------------------|---------------------|---------------------|
| Enrolled selfie (source of truth) | `source_selfie_b64` | `source_selfie_s3`  |
| Probe — live face (`facematch`)   | `face_check_b64`    | `face_check_s3`     |
| Probe — ID document (`docmatch`)  | `doc_check_b64`     | `doc_check_s3`      |

base64 may be a bare string or a `data:` URI. The **enrolled selfie is only ever
embedded** (the trusted reference). The liveness/anti-spoof gate runs on the
**probe**, and only for `facematch` — a `docmatch` document is a printed photo,
not a live face, so it is **not** liveness-checked. The API:
1. Resolves the probe (decode base64 or HTTP GET the S3 URL) and validates it (≤ 5MB; must decode).
2. **facematch only:** runs **liveness + IQA** (NSFW, spoof, blur) on the probe
   via the **ML inference service** (`POST /v1/iqa/assess`).
3. Embeds the probe (`POST /v1/face/embed`; for `docmatch` the face is first
   split out of the document, then embedded).
4. Resolves the enrolled selfie (base64 or plain **HTTP GET**, no AWS
   credentials) and embeds it.
5. Computes **cosine similarity** between the two embeddings and applies the
   match threshold.

The ML inference service URL is configured via **`ZEPIRIS_ML_INFERENCE_SERVICE_URL`**.

---

## Health & Readiness

### GET /healthz

Liveness probe — check if service is running.

**Response (200 OK)**
```json
{
  "status": "ok"
}
```

---

### GET /readyz

Readiness probe — check if service is ready to handle requests.

**Response (200 OK)**
```json
{
  "status": "ready"
}
```

---

## Face Operations

### POST /v1/faces/verify

Stateless 1:1 face verification. Compares an incoming live face (the probe)
against the user's enrolled selfie (the source of truth). Nothing is persisted.

Aliased by `POST /v1/faces/facematch/verify`; `POST /v1/faces/docmatch/verify`
behaves identically but the probe is an ID document: the face photo is
auto-extracted from the document (detected, cropped with margin, upscaled)
before embedding, the liveness gate is **not** applied (a document isn't a live
face), and the default threshold is more lenient. `docmatch` also accepts an
optional `doc_type` form field (e.g. `aadhaar`, `pan`) used to bucket the
request for adaptive threshold learning (see `POST /v1/faces/feedback`).

**Request**
```
Content-Type: application/json

JSON fields (supply exactly one input per side — base64 OR S3 URL):
Enrolled selfie / source of truth (required, one of) — only embedded:
- source_selfie_b64 : Enrolled selfie as base64 (bare string or data: URI), max 5MB
- source_selfie_s3  : Presigned or public URL of the enrolled selfie
Probe being verified (required, one of):
- face_check_b64 / face_check_s3  : Incoming live face   (facematch, /verify; liveness-gated)
- doc_check_b64  / doc_check_s3   : Incoming ID document  (docmatch; face extracted, no liveness)
Other:
- threshold (optional): Cosine match threshold (number). Defaults to
                        ZEPIRIS_VERIFY_THRESHOLD (0.5; docmatch default 0.4)
- doc_type  (optional, docmatch only): e.g. aadhaar, pan — buckets the request
                        for adaptive threshold learning
```

Sending both the `_b64` and `_s3` input for the same side returns
`400 image_source_invalid` (`reason: "ambiguous"`); sending neither returns
`reason: "missing"`.

**Flow**: resolve probe (base64 or S3 GET) → *(facematch only)* liveness + IQA on
probe → embed probe (docmatch: extract document face first) → resolve enrolled
selfie (base64 or S3 GET, no AWS creds) → embed it → cosine similarity → match
decision. For `docmatch` there is no liveness gate, so `imageQualityAssessment`
is `null` in the response.

**Response (200 OK — success)**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "imageQualityAssessment": {
    "passed": true,
    "nsfw": { "is_safe": true, "probability": 1.0 },
    "spoof": { "is_live": true, "probability": 1.0 },
    "blur": { "is_sharp": true, "probability": 0.85 }
  },
  "verificationResult": {
    "isMatch": true,
    "score": 0.91,
    "threshold": 0.5,
    "thresholdSource": "learned"
  },
  "scores": {
    "matchScore": 0.91,
    "threshold": 0.5,
    "margin": 0.41,
    "livenessScore": 1.0,
    "blurScore": 0.85,
    "nsfwSafeScore": 1.0
  },
  "faceDetected": true,
  "iqaPassed": true
}
```

- `verificationResult.score` (= `scores.matchScore`) is the cosine similarity in
  `[0.0, 1.0]` (higher = more similar).
- `verificationResult.isMatch` is `true` when `score >= threshold`.
- `verificationResult.thresholdSource` shows how the threshold was chosen —
  `"explicit"` (per-request), `"learned"` (adaptively calibrated from
  `/feedback` for this `doc_type`), or `"default"`.
- `scores` is a flat summary of the key numbers gathered in one place:
  `matchScore`, `margin` (= score − threshold), and the quality scores
  `livenessScore` (live probability), `blurScore` (sharpness probability), and
  `nsfwSafeScore` (safe probability). The quality scores are `null` when no
  liveness/IQA gate ran (e.g. `docmatch`).
- `documentFace` (docmatch only; `null` for facematch) reports how the face was
  pulled off the document: `faceDetected`, `detScore` (detector confidence),
  `sharpness` (variance-of-Laplacian of the extracted face — a crisp ID photo is
  `> 100`, blurry phone captures `5–20`), `usedCrop` (true when the extracted
  crop embedded successfully, false when it fell back to the whole card), and
  `lowQuality` (true when `sharpness` is below `ZEPIRIS_DOC_MIN_SHARPNESS`). When
  that env is `> 0`, a low-quality document is rejected with `422
  document_too_blurry` instead of returning a silently weak match.

#### Early-exit responses (still HTTP 200)

When the live photo cannot be processed past a given stage, the response returns
**HTTP 200** with flags instead of a full scored `verificationResult`. In these
cases `verificationResult.score` is `null`.

**Live photo decode failed**
```json
{
  "requestId": "...",
  "decodeFailed": true,
  "faceDetected": false,
  "iqaPassed": false,
  "verificationResult": {
    "isMatch": false,
    "score": null,
    "threshold": 0.5
  }
}
```

**Liveness failed** (adds `livenessFailed`, includes `imageQualityAssessment`)
```json
{
  "requestId": "...",
  "livenessFailed": true,
  "imageQualityAssessment": {
    "passed": false,
    "nsfw": { "is_safe": true, "probability": 1.0 },
    "spoof": { "is_live": false, "probability": 0.10 },
    "blur": { "is_sharp": true, "probability": 0.85 }
  },
  "faceDetected": true,
  "iqaPassed": false,
  "verificationResult": {
    "isMatch": false,
    "score": null,
    "threshold": 0.5
  }
}
```

**IQA failed** (not passed — e.g. blur/NSFW)
```json
{
  "requestId": "...",
  "imageQualityAssessment": {
    "passed": false,
    "nsfw": { "is_safe": true, "probability": 1.0 },
    "spoof": { "is_live": true, "probability": 1.0 },
    "blur": { "is_sharp": false, "probability": 0.15 }
  },
  "iqaPassed": false,
  "verificationResult": {
    "isMatch": false,
    "score": null,
    "threshold": 0.5
  }
}
```

**No face detected in live photo**
```json
{
  "requestId": "...",
  "faceDetected": false,
  "iqaPassed": true,
  "verificationResult": {
    "isMatch": false,
    "score": null,
    "threshold": 0.5
  }
}
```

#### Error responses (HTTP 400)

Problems with the **reference image** return **HTTP 400** with a `detail` object.

**`reference_image_fetch_failed`** — S3 GET timed out, returned 404, returned a
body larger than `ZEPIRIS_REFERENCE_MAX_BYTES`, or returned empty bytes.
```json
{
  "detail": {
    "error": "reference_image_fetch_failed"
  }
}
```

**`reference_image_decode_failed`** — fetched bytes are not a decodable image.
```json
{
  "detail": {
    "error": "reference_image_decode_failed"
  }
}
```

**`reference_face_not_detected`** — no face found in the reference image.
```json
{
  "detail": {
    "error": "reference_face_not_detected"
  }
}
```

---

### POST /v1/faces/feedback

Report the confirmed outcome of a past verification so the service keeps
calibrating itself on real traffic (especially Indian Aadhaar/PAN documents).
Every scored verification logs its match score (scores only — never images or
personal data). When downstream KYC/manual review confirms whether the pair was
genuinely the same person, post that label here; once enough labelled genuine
and impostor outcomes accumulate for a document type, its decision threshold is
re-fit automatically and used for subsequent requests (unless an explicit
`threshold` is sent).

**Request**
```
Content-Type: multipart/form-data

Form fields:
- request_id (required): The requestId returned by the original verify call
- genuine    (required): true if the selfie and document really were the same
                         person, false if it was an impostor
```

**Response (200 OK)**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "recorded": true,
  "matched_sample": true,
  "doc_type": "aadhaar",
  "thresholds": { "aadhaar": 0.47 }
}
```

Tunables: `ZEPIRIS_LEARNING_*` (see CONFIGURATION.md). Learned state lives in
`ZEPIRIS_LEARNING_DIR` as plain JSONL/JSON — auditable and resettable by
deleting the directory.

### POST /v1/faces/detect

Lightweight face-detection poll used by the verify UI to drive the capture ring.
Returns whether a face is present in the uploaded frame.

**Request**
```
Content-Type: multipart/form-data

Form fields:
- file (required): Image frame (JPEG/PNG, etc.)
```

**Response (200 OK)**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "faceDetected": true
}
```

---

## Data Models

### ImageQualityAssessmentResult

Image quality assessment of the live photo.

```json
{
  "passed": true,
  "nsfw": { "is_safe": true, "probability": 1.0 },
  "spoof": { "is_live": true, "probability": 1.0 },
  "blur": { "is_sharp": true, "probability": 0.85 }
}
```

**Fields**:
- `passed` (bool): True if all quality checks pass
- `nsfw` (NSFWDetectionResult): NSFW check
- `spoof` (SpoofDetectionResult): Liveness/spoof check
- `blur` (BlurDetectionResult): Sharpness/blur check

### NSFWDetectionResult

```json
{ "is_safe": true, "probability": 1.0 }
```

- `is_safe` (bool): True if content is safe (no NSFW detected)
- `probability` (float): Confidence [0.0, 1.0]

### SpoofDetectionResult

```json
{ "is_live": true, "probability": 1.0 }
```

- `is_live` (bool): True if image is genuine/live
- `probability` (float): Liveness confidence [0.0, 1.0]

### BlurDetectionResult

```json
{ "is_sharp": true, "probability": 0.85 }
```

- `is_sharp` (bool): True if image is sharp (not blurry)
- `probability` (float): Sharpness confidence [0.0, 1.0]

### VerificationResult

```json
{
  "isMatch": true,
  "score": 0.91,
  "threshold": 0.5
}
```

- `isMatch` (bool): True if `score >= threshold`
- `score` (float | null): Cosine similarity [0.0, 1.0]; `null` on early exit
- `threshold` (float): Match threshold used for this request

---

## Match Similarity Score (Distance Metric)

**Metric**: COSINE similarity (higher = more similar)

- `1.0` = identical faces
- `0.9` = high-confidence match
- `0.5` = default decision threshold
- `0.0` = completely different faces

The decision threshold is configurable per request (`threshold` form field) and
defaults to `ZEPIRIS_VERIFY_THRESHOLD` (`0.5`).

---

## Error Handling

### Common Error Codes

| Code | Condition | Example |
|------|-----------|---------|
| 200 | Live-photo early exit (decode/liveness/IQA/no-face) | See early-exit responses above |
| 400 | Missing/ambiguous/invalid image source | `{"detail": {"message": "image_source_invalid", "field": "source_selfie", "reason": "ambiguous"}}` |
| 422 | Image too large (> 5 MB) | `{"detail": "image_too_large_..."}` |
| 500 | Server error (ML inference unreachable, etc.) | `{"detail": "..."}` |

---

## Examples

Requests are JSON (`Content-Type: application/json`).

### cURL — Face match (S3 vs S3, the simplest case)

```bash
# probe = incoming live face (S3 URL); source_selfie = enrolled selfie in your DB (S3 URL)
curl -X POST http://localhost:8000/v1/faces/facematch/verify \
  -H "Content-Type: application/json" \
  -d '{
    "face_check_s3": "https://your-bucket.s3.amazonaws.com/incoming.jpg?X-Amz-Signature=...",
    "source_selfie_s3": "https://your-bucket.s3.amazonaws.com/enrolled.jpg?X-Amz-Signature=..."
  }'
```

Either side may instead be supplied as inline base64. Embedding base64 in JSON
from the shell is easiest via `jq`:

```bash
jq -n --arg face "$(base64 -i live.jpg)" \
  '{face_check_b64:$face, source_selfie_s3:"https://your-bucket.s3.amazonaws.com/enrolled.jpg?X-Amz-Signature=...", threshold:0.6}' \
  | curl -X POST http://localhost:8000/v1/faces/facematch/verify \
      -H "Content-Type: application/json" -d @-
```

### cURL — Doc match (S3 vs S3)

```bash
# probe = uploaded ID document (S3 URL); source_selfie = enrolled selfie (S3 URL)
# doc_type buckets adaptive learning; no liveness gate runs on the document
curl -X POST http://localhost:8000/v1/faces/docmatch/verify \
  -H "Content-Type: application/json" \
  -d '{
    "doc_check_s3": "https://your-bucket.s3.amazonaws.com/aadhaar.jpg?X-Amz-Signature=...",
    "source_selfie_s3": "https://your-bucket.s3.amazonaws.com/enrolled.jpg?X-Amz-Signature=...",
    "doc_type": "aadhaar"
  }'
```

### Python — Verify

```python
import base64

import requests

BASE_URL = "http://localhost:8000"

with open("live.jpg", "rb") as f:
    face_check_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{BASE_URL}/v1/faces/facematch/verify",
    json={
        # incoming live face being verified
        "face_check_b64": face_check_b64,
        # enrolled selfie (source of truth) stored in your DB
        "source_selfie_s3": "https://your-bucket.s3.amazonaws.com/enrolled.jpg?X-Amz-Signature=...",
        "threshold": 0.5,
    },
)
result = response.json()
print(result.get("verificationResult"))
```

---

## Rate Limiting

Currently not implemented. For production, planned:
- 100 requests/minute per IP
- 1000 requests/hour per API key

---

## Authentication

Currently no authentication. For production:
- JWT tokens recommended
- API key management
- See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## API Versioning

Current: `v1`

Future versions will use: `/v2/faces/...`

---

## See Also

- [README.md](../README.md) - Project overview
- [SETUP_GUIDE.md](../SETUP_GUIDE.md) - Setup instructions
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [LOCAL_SETUP_AND_TEST.md](../LOCAL_SETUP_AND_TEST.md) - Testing guide
