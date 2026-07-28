# ZepIris — Deployment Guide

Face verification service (go-online selfie vs enrolled selfie). Two processes:
**API** (FastAPI, orchestration + S3 fetch + cosine) → **ml_inference** (FastAPI,
CPU models: face detection + recognition). Runs pure 1:1 face match (liveness
gate off).

---

## 1. Measured performance

Response time, measured locally on an 8-core CPU against **real prod images**
(current config: liveness off, flip-TTA off, detector 512×512):

| Metric | Facematch |
|---|---|
| **Median (p50)** | **~0.75 s** |
| p90 | ~1.0 s |
| avg | ~0.77 s |
| min | ~0.22 s |
| n | 50 diverse prod pairs |

- Of the ~0.75 s, **~0.15 s is the S3 image fetch** (2 images, in parallel); actual model compute is **~0.6 s**.
- **Prod projection:** the prod EC2/Fargate CPU is slower than this laptop, but S3 is same-region (fetch ~30–50 ms), so the two roughly offset → expect **~0.5–1 s median on prod**.

### How it got here

| Version | Median (local) |
|---|---|
| Original | ~1.9 s (prod was 10–11 s) |
| detect-once + parallel S3 fetch + off-event-loop | ~0.7 s |
| + liveness gate OFF | ~0.4 s (tiny images) / ~1.1 s (real images) |
| **+ flip-TTA off + detector 512** (current) | **~0.75 s (real images)** |

Throughput anchor for sizing: **~3–4 req/s per 4 vCPU**.

---

## 2. Workload

| | Value |
|---|---|
| Baseline | ~10 req/min (~0.17 req/s) |
| Daily spikes | 05:00, 09:00, 17:00, 20:00 IST — rider go-online |
| Spike size | up to ~1000 req/min (~17 req/s) |
| Spike duration | ~15 min each |

Spiky, at fixed clock times → **scheduled autoscaling** (pre-warm before each
spike), with reactive target-tracking as a safety net.

---

## 3. Recommended infra — AWS `ap-south-1` (same region as the S3 image bucket)

> Deploy in AWS Mumbai, same account/VPC as `earn-de-docs`. The hot path fetches
> images from S3 on every request — cross-cloud (GCP/serverless) would add S3
> egress cost + latency + PII/DPDP concerns. CPU is sufficient; **no GPU** at this
> volume.

### Option A — ECS Fargate (recommended)

| Service | Task size | Baseline | Spike (15 min) | Throughput |
|---|---|---|---|---|
| **ml_inference** | **4 vCPU / 8 GB** | 1 task | **6 tasks** | ~3–4 req/s / task |
| **API** | **0.5 vCPU / 1 GB** | 1 task | 2 tasks | I/O only |
| ALB | — | 1 | 1 | — |

Models need ~1–2 GB resident — keep memory ≥ 8 GB/task. 6 ml tasks ≈ 21 req/s > 17 req/s peak (headroom).
At this volume API + ml_inference may be **collapsed into one 4 vCPU / 8 GB task** for simplicity.

### Option B — EC2 (ECS-on-EC2 or plain ASG)

| Role | Instance | Baseline | Spike |
|---|---|---|---|
| ml_inference | `c7i.2xlarge` (8 vCPU / 16 GB, ~7 req/s) | 1 | 3 |
| — finer unit — | `c7i.xlarge` (4 vCPU / 8 GB, ~3.5 req/s) | 1 | 6 |
| API | `c7i.large` (2 vCPU / 4 GB) | 1 | 2 |

**Graviton alt:** `c7g.*` (ARM) — arm64 wheels work, ~15–20% cheaper. Use `c7i` (x86) for the most-tested wheel path.

---

## 4. Autoscaling (scheduled + reactive)

| Trigger | Action | Why |
|---|---|---|
| **Scheduled** 04:50 / 08:50 / 16:50 / 19:50 IST | ml_inference min 1 → **6** | Pre-warm ~10 min before spike (model load ~15–20 s; reacting at spike-start misses the window) |
| **Scheduled** +20 min (05:10 / 09:10 / 17:10 / 20:10) | scale **6 → 1** | Back to baseline |
| **Target tracking** (backup) | ALB `RequestCountPerTarget` or CPU 60% → add tasks | Unexpected/early bursts |

- Cooldowns: scale-out 60 s (fast), scale-in 300 s (slow) — avoids flapping mid-spike.
- **min ≥ 1** always (or 2 for AZ HA) — **never scale to zero**, cold model load would blow the first requests.
- Run **spike tasks on Fargate Spot** (~70% cheaper; the bursts are interruption-tolerant); keep baseline on on-demand.

Example (Application Auto Scaling, one per window):
```bash
aws application-autoscaling put-scheduled-action --service-namespace ecs \
  --resource-id service/zepiris/ml-inference \
  --scalable-dimension ecs:service:DesiredCount \
  --scheduled-action-name prewarm-0450 \
  --schedule "cron(20 23 * * ? *)"   `# 04:50 IST = 23:20 UTC` \
  --scalable-target-action MinCapacity=6,MaxCapacity=8
# +20 min: scale back to MinCapacity=1,MaxCapacity=2
```

---

## 5. Cost (ap-south-1, on-demand, approx)

| | Baseline 24/7 | + 4 daily 15-min spikes |
|---|---|---|
| Fargate (1× 4vCPU/8GB + 1× 0.5vCPU) | **~$160/mo** | +~$5/mo |
| EC2 (1× c7i.2xlarge + 1× c7i.large) | **~$400/mo** | +~$8/mo |

Fargate wins here — you pay for 1 baseline task 24/7; the 6-task spike runs ~1 hr/day total. A GPU (`g5.xlarge`) would be idle 99% of the time — skip it.

---

## 6. Runtime config (env)

| Env | Value | Effect |
|---|---|---|
| `ZEPIRIS_ML_INFERENCE_SERVICE_URL` | `http://ml-inference:8001` | API → ml_inference |
| `ML_SERVICE_FACE_DETECTION_WIDTH/HEIGHT` | `512` | faster detection (default) |
| `ML_SERVICE_FACE_ENABLE_FLIP_TTA` | `false` | ~halves recognition (default) |
| liveness gate | removed in code | pure 1:1 match |

Tradeoffs already applied for speed: **liveness off** (a printed photo / screen replay of the
enrolled selfie passes on match score alone) and **TTA off** (~0.02–0.03 lower genuine
scores, still well clear of the 0.5 threshold). Set `ML_SERVICE_FACE_ENABLE_FLIP_TTA=true`
to trade some speed back for robustness on blurry document photos.

---

## 7. Further optimization (before scaling hardware)

- **Pre-compute enrolled-selfie embeddings** (store per user in Redis / Milvus). The reference
  never changes per go-online event — embedding it every time is wasted work + one extra S3
  fetch. Cuts per-request compute ~50% and removes one fetch → roughly doubles capacity.
- Load-test with k6/Locust against one task to get the true per-node req/s before finalizing counts.
