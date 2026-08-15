# CPU vs GPU: choosing the compute for face matching

Capacity and cost at 100, 500, and 1000 req/s, and how to deploy on either.

Read [PERFORMANCE.md](PERFORMANCE.md) for where the CPU numbers come from and
[EC2_AUTOSCALING.md](EC2_AUTOSCALING.md) for the fleet that runs them.

---

## 1. The short version

Right-sized, **GPU is cheaper at every scale** — including 100 req/s, where a
small T4 costs half the CPU fleet and needs no batching at all.

| Scale | Recommended | Fleet | Cost/mo | CPU alternative |
|---|---|---|---|---|
| **100 req/s** | GPU · T4 | 2 × `g4dn.xlarge` | **~$876** | 6 × `c7i.2xlarge` · $1,840 |
| **500 req/s** | GPU · A10G | 2 × `g5.2xlarge` | **~$2,117** | 29 × `c7i.2xlarge` · $8,891 |
| **1000 req/s** | GPU · A10G | 3 × `g5.2xlarge` | **~$3,176** | 58 × `c7i.2xlarge` · $17,783 |

What CPU buys is not economy — it is **shipping this week with no unvalidated
assumptions**. Every GPU figure on this page is estimated; every CPU figure is
measured.

**Recommendation at the 100 req/s launch target:**

- **If you have ~a week before launch, go GPU.** Cheaper, keeps the `accurate`
  recognition weights, and it is the architecture you will want at 500+, so the
  validation is work you would do eventually — done while the stakes are low.
- **If you need to ship sooner, go CPU** (`balanced` tier, 6 instances). Nothing
  about it is a dead end; run the GPU validation in parallel and cut over later.

---

## 2. Why GPU wins on cost

CPU capacity comes in small linear increments — every ~17.5 req/s needs another
instance, forever. GPU capacity comes in steps large enough that the *redundancy
floor* (two instances) already exceeds what you need:

```
$18k ┤                                                        ● CPU 58 inst
     │                                              ╭─────────╯
 $9k ┤                        ● CPU 29 inst ────────╯
     │              ╭─────────╯
     │    ● CPU 6 inst
   0 ┤    ●─────────────────● GPU 2 inst ────────────● GPU 3 inst
     └────┴──────────────────┴──────────────────────┴────────────
        100 req/s          500 req/s              1000 req/s
```

The GPU line barely leaves the axis because the card is never the constraint at
these rates — instance counts are set by redundancy and host decode capacity.

---

## 3. Which GPU

At 100 req/s the card is barely working. Pick the cheapest instance whose **host**
can decode fast enough, then check the card has headroom for where you are going.

| Instance | Card | vCPU | $/mo each | Unbatched est. | Batched ceiling | Use at |
|---|---|---|---|---|---|---|
| **`g4dn.xlarge`** | T4 | 4 | $438 | ~110 req/s | ~770 req/s | **100 req/s** |
| `g6.xlarge` | L4 | 4 | $730 | ~210 req/s | ~1,430 req/s | 100–300 req/s |
| `g6.2xlarge` | L4 | 8 | $876 | ~210 req/s | ~1,430 req/s | 500 req/s |
| `g5.2xlarge` | A10G | 8 | $1,058 | ~215 req/s | ~1,480 req/s | 500–1000 req/s |
| `g5.xlarge` | A10G | 4 | $869 | ~215 req/s | ~1,480 req/s | ⚠ avoid past ~300 req/s |

**The `g5.xlarge` trap:** the same A10G as the 2xlarge for $190/mo less, but only
4 vCPU. Decode needs 6.7 cores at 1000 req/s, so the host starves a card that is
otherwise the fastest in the table. Cheap GPU, wasted GPU.

### The constraint that sizes a GPU host

JPEG decode stays on the CPU whichever architecture you pick, and it scales with
request rate. Measured: one core decodes **299 images/sec** at 1200×1600.

| Scale | Decodes/sec | Cores for decode alone | Host needed |
|---|---|---|---|
| 100 req/s | 200 | 0.7 | 4 vCPU is fine |
| 500 req/s | 1,000 | 3.3 | 8 vCPU |
| 1000 req/s | 2,000 | 6.7 | 8 vCPU × 3 instances |

---

## 4. Batching: required above ~200 req/s, not at 100

A GPU fed one image at a time is mostly idle. The 500 and 1000 req/s plans assume
a **micro-batching queue** — gather requests for ~5 ms, run them as a batch of
16–32, scatter the results — in front of the recognizer. That is the difference
between roughly 5× a CPU core and 50×.

**That queue does not exist in this codebase**, and it cannot be validated without
GPU hardware. Budget it as real engineering with a validation cycle, not a config
change.

The 100 req/s plan does not need it. That is what puts GPU on a one-week timeline
at the launch target rather than a multi-week one.

> Batching is a **GPU-specific** optimisation. Measured on CPU it was *slower* —
> 3.5 vs 4.8 req/s for batch-2 versus two batch-1 runs — because with one thread
> per inference a batch just serializes the work while holding the session
> longer. This is why the current code deliberately does not batch.

---

## 5. Deploying on GPU

Everything below is additive to [EC2_AUTOSCALING.md](EC2_AUTOSCALING.md) — same
ASG, same ALB, same warm pool, same scaling signal. Only the instance type, the
image, and two settings change.

### 5.1 Base AMI

Start from the **AWS Deep Learning Base AMI (Ubuntu 22.04)**, which ships the
NVIDIA driver and container toolkit already. It saves a fragile driver install and
keeps cold start short.

```bash
aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'reverse(sort_by(Images,&CreationDate))[:1].[ImageId,Name]' --output text
```

Verify on a booted instance before baking:

```bash
nvidia-smi                                    # driver + card visible
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 5.2 ONNX Runtime with CUDA

The CPU wheel silently ignores `ML_SERVICE_ML_DEVICE=cuda` — it has no CUDA
provider to select, so it falls back and you get CPU speed on a GPU bill. Swap the
wheel in `ml_inference.Dockerfile`:

```dockerfile
# onnxruntime and onnxruntime-gpu conflict — remove the CPU wheel first.
RUN pip uninstall -y onnxruntime \
    && pip install --no-cache-dir onnxruntime-gpu
```

Confirm the provider is actually present at runtime:

```bash
docker compose exec ml-inference python -c \
  "import onnxruntime; print(onnxruntime.get_available_providers())"
# must include CUDAExecutionProvider
```

The engine selects the provider automatically and **logs the one actually in
use** — not the one requested. A missing CUDA wheel degrades to CPU with only a
warning, which on a GPU instance looks exactly like a disappointing GPU, so the
startup line names the real provider:

```
Face engine ready: tier=accurate det=det_10g.onnx@512x512 rec=w600k_r50.onnx \
  intra_op=1 device=cuda provider=CUDAExecutionProvider
```

If it falls back, the service logs an error you can alarm on:

```
ERROR device=cuda was requested but inference is running on CPUExecutionProvider —
      check that onnxruntime-gpu is installed and the GPU is visible to the container
```

### 5.3 Expose the GPU to the container

```yaml
# docker-compose.yml — ml-inference service
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 5.4 Settings

```bash
ML_SERVICE_ML_DEVICE=cuda
# The accurate tier is the point of moving to GPU — ResNet50 recognition is
# trivial work for a GPU and keeps every existing threshold valid.
ML_SERVICE_FACE_TIER=accurate
# intra_op governs CPU threads and is irrelevant once inference runs on the card;
# leave it at 1 so the host's cores stay free for JPEG decode.
ML_SERVICE_FACE_INTRA_OP_THREADS=1
# Raise the in-flight cap: it defaults to CPU count - 1, which on a 4-vCPU GPU
# host would throttle to 3 concurrent inferences and leave the card idle.
ML_SERVICE_MAX_CONCURRENT_INFERENCES=32
```

> `MAX_CONCURRENT_INFERENCES` is the setting most likely to be missed. Its default
> is derived from the CPU count, which is the right bound for CPU inference and
> far too low for a GPU. A GPU host that mysteriously plateaus is usually this.

### 5.5 Cost note

GPU Spot pools are thin and reclaimed often. Run the baseline on-demand; if you
want Spot, list several GPU types in the mixed-instances policy rather than
relying on one.

---

## 6. What is measured and what is not

**Measured** (8-core Apple Silicon, distinct images per request, full API path):

- Per-tier CPU throughput: **52 / 17.5 / 9.8 req/s** per 8 vCPU
  (`fast` / `balanced` / `accurate`)
- JPEG decode: **299 images/sec/core** at 1200×1600
- Genuine/impostor margins per tier on the sample gallery
- End-to-end gain from optimisation: **3.4 → 41.4 req/s**

**Estimated** (no GPU has run this workload):

- All GPU throughput, from ~25.4 GFLOPs per request at 30% of FP16 peak
- Unbatched figures, scaled from ~16 ms of GPU time per request on a T4
- Every GPU instance count and cost on this page

**Unbatched throughput is the softest number here.** It is latency-bound rather
than compute-bound, so it depends on kernel-launch and transfer overhead that
arithmetic predicts poorly — and it is exactly what the 100 req/s GPU plan rests
on. Mixed performance/efficiency cores also make the CPU figures conservative for
uniform x86; you are more likely to need fewer CPU instances than more.

---

## 7. Validate before committing

Stand up **one** instance of the type you intend to buy and measure it:

```bash
python scripts/loadtest.py --target api --concurrency 16 --requests 300
```

Then `instances = ceil(target / measured)`.

For the GPU path, the number to watch is unbatched throughput on one
`g4dn.xlarge`:

| Measured | Verdict |
|---|---|
| ≥ 100 req/s | 2 instances as planned (~$876/mo) |
| 50–100 req/s | 3 instances (~$1,314/mo) — still cheaper than CPU |
| < 50 req/s | Build the batching queue, or move to L4 (`g6.xlarge`) |

Also confirm the card is actually being used — a silent CPU fallback looks exactly
like a disappointing GPU:

```bash
nvidia-smi dmon -s u -c 10     # utilisation should be non-zero under load
```

---

## 8. Free capacity on either path

Capturing selfies at **640×854 instead of 1800×2400 is ~2.4× throughput** at no
accuracy cost. The detector letterboxes to 512×512 and the recognizer works from a
112×112 crop, so a full-resolution phone capture spends its extra pixels on JPEG
decode and nothing else.

| Probe resolution | Payload | Throughput |
|---|---|---|
| 640×854 | ~74 KB | **37.7 req/s** |
| 900×1200 | ~121 KB | 29.5 req/s |
| 1200×1600 | ~183 KB | 30.8 req/s |
| 1800×2400 | ~332 KB | 15.6 req/s |

On CPU that takes the 100 req/s fleet from 6 instances to 3. On GPU it cuts the
decode load that sizes the host, which keeps a 4-vCPU instance viable further up
the ladder. If the mobile client can be changed, do this before buying anything.
