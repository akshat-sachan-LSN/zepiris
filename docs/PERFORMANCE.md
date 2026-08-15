# Performance and capacity

How to make a verification fast, and how much hardware 100 concurrent
verifications under 500 ms actually needs.

---

## The one number that decides everything

Face matching is CPU-bound. A box can do a fixed amount of model work per
second, and that ceiling — not the number of connections it accepts — decides
how many requests it can hold under a deadline.

Little's Law gives the whole answer:

```
sustainable concurrency = throughput (req/s) x deadline (s)
```

So **100 concurrent requests under 500 ms requires 200 req/s of sustained
throughput.** Nothing else in this document matters more than that number. If
the fleet cannot do 200 req/s, no amount of tuning will hold 100 requests under
500 ms — they will queue, and every one of them will miss.

Conversely, a fleet that *does* 200 req/s will hold 100 concurrent under 500 ms
almost automatically.

---

## What changed, and what it bought

Measured end to end (API → ML service) on one 8-core machine, 100 concurrent
requests, **distinct images per request** so no cache can shortcut the work:

| Configuration | Throughput | p95 latency @ 100 concurrent |
|---|---|---|
| Before | 3.4 req/s | 30,175 ms |
| After, `balanced` tier (default) | 9.8 req/s | 10,935 ms |
| After, `fast` tier | 41.4 req/s | 4,833 ms |

**2.9x on the default, 12x on the fast tier**, same hardware, same accuracy on
the default.

Where the deadline actually breaks on that machine (`fast` tier, full path):

| Concurrency | Throughput | p95 | Within 500 ms |
|---|---|---|---|
| 8 | 55.8 req/s | 164 ms | 100% |
| 16 | 52.3 req/s | 368 ms | 100% |
| 24 | 47.8 req/s | 704 ms | 71% |
| 100 | 41.4 req/s | 4,833 ms | 1% |

Throughput plateaus around 52 req/s and latency then grows linearly with
concurrency, exactly as Little's Law predicts. This box holds **~16 concurrent**
under 500 ms. Reaching 100 needs about **6x the CPU**.

Use distinct images when you benchmark this yourself. Replaying one image lets
content-keyed caches answer from memory and inflates the result — the same test
with a single repeated image reported 7.1 req/s for the "before" case, roughly
double its real capacity.

### Where the gains came from

**One round trip instead of two, carrying bytes instead of base64.** The old path
sent each image to the ML service separately, and did so by decoding the upload,
re-encoding it as lossless PNG, and base64-ing the result — then the ML service
undid all of that. Per image, measured on a 1200x1600 capture:

| Step | Cost | Status |
|---|---|---|
| PNG encode | 31.5 ms | removed |
| base64 encode | 2.2 ms | removed |
| base64 decode | 3.1 ms | removed |
| PNG decode | 22.4 ms | removed |
| detection-cache hash | 5.5 ms | removed |
| **per image** | **64.7 ms** | |
| **per request (two images)** | **~130 ms** | |

The PNG payload was also ~1.5 MB versus a 149 KB JPEG — 10x the bytes across the
network, twice per request. Now both original images go in one multipart call
and only a float comes back; the 512-dimension vectors never touch JSON.

**ONNX Runtime threading.** InsightFace builds its sessions with default options,
which size each inference's thread pool to the whole machine. That is right for
one request at a time and wrong under concurrency — N requests each fanning out
across every core oversubscribe the CPU. Pinning `intra_op_num_threads=1` and
taking parallelism from request concurrency instead measured **~1.5x** throughput
at 8-way concurrency in isolation.

**A cheaper detector.** Detection, not recognition, dominated: swapping SCRFD-10G
for SCRFD-500M while keeping the same ResNet50 recognition weights roughly
doubles throughput and left the genuine/impostor margin unchanged (see below).

**Admission control.** Past the core count, extra in-flight requests add queueing
latency and no throughput. The ML service now caps concurrent inferences and
sheds the excess with `503` after a bounded wait, so overload costs the requests
that were already doomed rather than every request at once. Measured: throughput
stays flat at ~53 req/s from 16-way to 100-way concurrency instead of collapsing.
With the queue timeout tightened to 0.5 s and 100 requests in flight, 68 of 200
were shed immediately and the remaining 132 were served at full speed — the
service degrades by refusing work, not by making everyone wait.

**Event-loop hygiene.** Two things blocked the API's event loop on every request:
base64-decoding large payloads, and the adaptive-learning write (a file append
under a process-wide lock). Both now run off the loop. The base64 fix alone took
the API path from 34 to 44 req/s at 100 concurrent.

---

## Choosing a tier

`ML_SERVICE_FACE_TIER` selects the detector/recognizer pairing.

| Tier | Detector | Recognizer | Relative speed | Threshold |
|---|---|---|---|---|
| `accurate` | SCRFD-10G | ResNet50 | 1x | unchanged |
| `balanced` **(default)** | SCRFD-500M | ResNet50 | ~3x | unchanged |
| `fast` | SCRFD-500M | MobileFaceNet | ~13x | **must be recalibrated** |

`balanced` keeps the ResNet50 recognition weights — the model that produces the
embedding and therefore decides match scores — so existing thresholds carry over.

`fast` replaces the recognition model, which **moves the embedding space**. Its
scores are not comparable to the other tiers and the decision threshold must be
recalibrated against your own labelled pairs before it goes near production
traffic.

### Measured separation

`scripts/compare_tiers.py` scores every tier on your own images. Genuine pairs
come from realistic capture variation (rescaling, JPEG recompression, exposure
shifts, rotation, blur); impostor pairs are every cross-person combination.

On the 6-identity sample gallery, at threshold 0.5:

| Tier | ms/pair | worst genuine | best impostor | margin | false accepts | false rejects |
|---|---|---|---|---|---|---|
| `accurate` | 722 | 0.884 | 0.204 | 0.680 | 0 | 0 |
| `balanced` | 367 | 0.888 | 0.192 | **0.697** | 0 | 0 |
| `fast` | 63 | 0.856 | 0.208 | 0.648 | 0 | 0 |

`balanced` is free: it is twice as fast and its margin is, if anything, slightly
wider. `fast` gives up about 0.05 of margin for another ~6x.

**Run this on your own data before trusting it.** Six identities with synthetic
variation is a smoke test, not a validation set. Real genuine pairs — a different
day, different lighting, a different phone — are harder than any augmentation,
and the margins will be narrower for every tier.

---

## Sizing for 100 concurrent under 500 ms

The target needs **200 req/s**. Measured ML-service capacity on 8 cores, distinct
images, was ~53 req/s on the `fast` tier — about **6.6 req/s per core**:

```
cores needed = 200 req/s / (req/s per core)
```

| Tier | req/s per core (measured) | Cores for 200 req/s |
|---|---|---|
| `fast` | ~6.6 | **~30** |
| `balanced` | ~1.2 | ~160 |
| `accurate` | ~0.7 | ~280 |

Equivalently: this 8-core machine held 16 concurrent under 500 ms, and 100 is
about 6x that.

**On CPU, 100 concurrent under 500 ms is only practical on the `fast` tier**, at
roughly 30 vCPU of ML capacity — for example 4 x `c7i.2xlarge`, or 8 ECS tasks of
4 vCPU. On `balanced` the same target needs a fleet several times larger, and is
usually not worth it.

Two caveats before you buy instances:

- **These numbers come from an 8-core Apple Silicon laptop with mixed
  performance/efficiency cores, running the API and the ML service side by side
  on the same CPUs.** Uniform x86 server cores behave differently. Treat the
  per-core figures as a starting estimate, not a quote.
- **Measure on the instance type you will actually run.** `scripts/loadtest.py`
  does exactly this test; point it at one warmed task and read the throughput.

### If you need accuracy *and* the deadline

Use a GPU. ResNet50 ArcFace at 112x112 is trivial work for even a small
inference GPU, and one `g5.xlarge` clears 200 req/s while keeping `accurate`-tier
weights. Set `ML_SERVICE_ML_DEVICE=cuda` and install `onnxruntime-gpu`.

### Is 100 concurrent the real requirement?

Worth checking against `DEPLOYMENT.md`: measured production traffic peaks at
**~24 req/s** in the busiest minute of the day. At 24 req/s, holding p95 under
500 ms needs about 12 concurrent in flight, which the `balanced` tier serves on
roughly 4 vCPU — a fraction of what the 100-concurrent target implies. Size for
100 only if the burst profile has genuinely changed.

---

## Verifying it

Start both services, then:

```bash
python scripts/loadtest.py --target api --concurrency 100 --requests 500 --deadline-ms 500
```

It builds one distinct image pair per request, warms the models, and reports the
percentile distribution with a pass/fail against the deadline. `--target ml`
points at the ML service directly, which separates model time from API overhead —
useful when the two run on the same box and compete for cores.

To find the concurrency a given fleet actually sustains, sweep it:

```bash
for c in 8 16 32 64 100; do
  python scripts/loadtest.py --target api --concurrency $c --requests $((c*8))
done
```

Throughput plateaus at the hardware ceiling and latency then grows linearly with
concurrency. The last level that passes is the real capacity of that fleet.

---

## Settings reference

**ML service** (`ML_SERVICE_` prefix)

| Setting | Default | Notes |
|---|---|---|
| `FACE_TIER` | `balanced` | `accurate` / `balanced` / `fast` |
| `FACE_INTRA_OP_THREADS` | `1` | Raise only for low-concurrency, latency-critical use |
| `FACE_MAX_INPUT_SIDE` | `1600` | Downscale oversized captures before detection |
| `FACE_ENABLE_DET_CACHE` | `false` | Only useful if one request detects the same pixels twice |
| `MAX_CONCURRENT_INFERENCES` | `0` (CPU count - 1) | In-flight inference cap |
| `INFERENCE_QUEUE_TIMEOUT_SECONDS` | `20` | Wait before shedding with 503 |
| `FACE_UPSCALE_FACTOR` | `2.0` | Each retry is a full extra detection pass |

**API** (`ZEPIRIS_` prefix)

| Setting | Default | Notes |
|---|---|---|
| `MATCH_MODE` | `remote` | `local` runs models in-process — no HTTP hop at all |
| `ML_MAX_CONNECTIONS` | `200` | Must cover peak in-flight requests |
| `S3_MAX_CONNECTIONS` | `200` | Two reference fetches per request |
| `API_WORKERS` | `2` | API is I/O-bound; CPU cost lives in the ML service |
| `THREAD_POOL_SIZE` | `64` | Backs the few remaining sync call sites |

### `MATCH_MODE=local`

Loads the models into the API process, removing the HTTP hop entirely. It needs
the ML extras installed alongside the API (`poetry install --extras ml`); a
failed load falls back to `remote` rather than leaving the service unable to
match at all.

**It is not the faster option under load, despite removing a network hop.**
Measured on the same 8-core machine, `fast` tier:

| Mode | Throughput | p95 @ 16 concurrent |
|---|---|---|
| Two processes on one host | 52.3 req/s | 368 ms |
| `local` (single process) | 39.6 req/s | 652 ms |

Everything ends up in one Python process, so HTTP handling, JSON, base64 and the
Python side of inference all contend for one GIL. Splitting them across two
processes buys more than the localhost hop costs. `local` also has no admission
control — the inference limiter lives in the ML service — so it degrades under
overload instead of shedding.

Use `local` for a single-container deployment, a low-concurrency edge box, or
local development. **For throughput, run both services and put them on the same
host**, which is what `docker-compose.yml` does: two processes, localhost
transport, one unit to scale.
