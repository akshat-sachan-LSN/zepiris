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

## Not embedding the reference twice

Recognition is 88% of a match: on a 1200x1600 capture, `balanced` on one core
spends ~19 ms detecting and ~164 ms embedding, per side. Halving the *work* is
therefore worth far more than shaving the parts around it.

Half of it is avoidable. The probe is a fresh capture — new pixels, nothing to
reuse. The reference is the enrolled selfie, and it is the same bytes on every
verification of that person, so its embedding is recomputed to produce a vector
that cannot have changed. `ML_SERVICE_FACE_REFERENCE_CACHE_SIZE` keeps those
vectors, keyed by a digest of the image bytes:

| | latency | note |
|---|---|---|
| both sides embedded | ~350 ms | what every request used to pay |
| reference cached | ~183 ms | one embed instead of two |

Keying on content rather than on a user id or an S3 URL is what makes this safe
to leave on. Different bytes are a different key, identical bytes through the
same model are an identical embedding, and the cache dies with the process — so
there is no TTL to tune, no invalidation to get wrong, and no way to serve a
vector belonging to a re-enrolled photo. A reference with no detectable face is
cached too, so a caller retrying against an unusable enrolled image stops paying
full detection each time.

The hit rate is the number to watch, on `/metrics` under `reference_cache`. It is
a property of the traffic, not the service: a population verifying repeatedly
against stable enrolments approaches 1.0, while genuinely first-time
verifications never hit and the ~8 MB buys nothing.

### Embedding the two sides at once

On a cache miss there are still two embeds, and they do not have to be
sequential — ONNX Runtime releases the GIL, so two threads genuinely use two
cores (measured 349 ms -> 177 ms). The catch is that this is only free while
cores are idle: at high concurrency every core already has a request on it, and
fanning one request across two takes throughput from everyone to help one
caller. So `ML_SERVICE_FACE_PARALLEL_PAIR_EMBED` is decided per request against
the limiter's occupancy, and the condition is that doubling *all* in-flight work
would still fit (`active * 2 <= limit`), not merely that a slot is free. A looser
rule backfired measurably: at 5 concurrent requests on 8 cores it let all five
fan out to ten threads and p50 rose from 564 ms to 658 ms.

Measured back to back on one 8-core machine, 20 enrolled references, every probe
unique, both features off then on:

| Concurrent users | p50 off | p50 on | req/s off | req/s on | shed off | shed on |
|---|---|---|---|---|---|---|
| 5 | 572 ms | 364 ms | 8.6 | 10.4 | 0 | 0 |
| 10 | 843 ms | 629 ms | 9.8 | 15.0 | 0 | 0 |
| 20 | 2239 ms | 1544 ms | 8.4 | 12.6 | 0 | 0 |
| 100 | 19200 ms | 14502 ms | 4.8 | 6.3 | 2 | 0 |
| 200 | 19425 ms | 17628 ms | 5.5 | 10.5 | 105 | 0 |

Neither changes a score. Both were checked against the uncached, sequential path
on real images and agreed to the bit.

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

## Sizing at the launch target — 100 req/s

100 req/s against a 500 ms deadline means ~50 requests in flight. Unlike the
larger targets below, this one is comfortably servable **without giving up any
accuracy**:

| Tier | req/s per 8 vCPU | `c7i.2xlarge` for 100 req/s | Thresholds |
|---|---|---|---|
| `accurate` | ~9.8 | 11 | unchanged |
| **`balanced`** ← recommended | **~17.5** | **6** | **unchanged** |
| `fast` | ~52 | 2 | must be recalibrated |

Run `balanced`. It keeps the ResNet50 recognition weights — the model that
produces the embedding and decides every match — so existing thresholds carry
over and no recalibration campaign stands between you and launch. `fast` would
cut the fleet to two instances, but moving the embedding space to save four
`c7i.2xlarge` is a poor trade on a biometric decision system. That trade only
starts paying at the scales below, where the CPU alternative stops being
affordable at all.

Derived from a clean full-path measurement of `fast` (52 req/s on 8 cores) plus
per-tier model cost measured single-threaded. Conservative for uniform x86 cores
— calibrate with `scripts/loadtest.py` on the instance type you intend to buy.

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

Use a GPU. ResNet50 ArcFace at 112x112 is trivial work for even a small inference
GPU, and it keeps `accurate`-tier weights so no threshold moves. Sizing, card
selection, and the deployment steps are in [CPU_VS_GPU.md](CPU_VS_GPU.md) —
including why `g5.xlarge` is a trap (4 vCPU cannot decode fast enough to feed its
own card).

---

## Scaling to 1000 req/s

1000 req/s is **5x** the 100-concurrent target and roughly **40x** the busiest
minute in production traffic today (~24 req/s, per `DEPLOYMENT.md`). Worth
confirming it is a real requirement rather than a headroom figure before
committing to the cost, because the cost is where this stops being a tuning
exercise.

### On CPU: it works, and it is expensive

| | |
|---|---|
| Required | 1000 req/s, `fast` tier |
| Measured | ~6.6 req/s per vCPU |
| Fleet | **~150 vCPU ≈ 19 x `c7i.2xlarge`** |
| On-demand | ~$5,750/month |
| Mixed Spot (base on-demand) | ~$2,000–2,500/month |

Nothing about the architecture breaks at this size — it is a linear scale-out of
what §"Sizing" already describes, with `max-size` raised and the warm pool grown
to match. It is simply a lot of instances to run for a face comparison.

### On GPU: cheaper, and keeps the accurate model

> Full comparison across 100 / 500 / 1000 req/s, card selection, and deployment
> steps: [CPU_VS_GPU.md](CPU_VS_GPU.md).

An A10G (`g5.xlarge`, ~$1.00/hr) runs ResNet50 at 112x112 in the thousands of
images per second when work is **batched**. Two or three of them plausibly carry
1000 req/s on `accurate`-tier weights, at ~$2,200/month — cheaper than the CPU
fleet *and* without the threshold recalibration `fast` demands.

**Two pieces of work stand between here and that number, and neither is done:**

1. **`onnxruntime-gpu` with the CUDA execution provider.** The engine already
   takes `device` and selects the provider (`face_engine.py`), but that path has
   not been run on real GPU hardware.
2. **Dynamic batching — the part that matters.** A GPU serving one image at a
   time is mostly idle; the throughput above assumes requests are gathered into
   batches of 16–32 before hitting the model. That means a micro-batching queue
   (collect for ~5 ms, run together, scatter results) in front of the recognizer.
   It does **not** exist today, and it is the difference between a GPU that is
   ~5x a CPU core and one that is ~50x.

Note that batching is a *GPU* optimization specifically. Measured on CPU it was
**slower** — 3.5 vs 4.8 req/s for batch-2 versus two batch-1 runs — because with
one thread per inference a batch just serializes the work while holding the
session longer. That is why the current code deliberately does not batch.

I have not built the batching queue because it cannot be validated here: there is
no CUDA device in this environment, and shipping unvalidated inference code into
a biometric verification path is not a reasonable trade. It is a well-understood
piece of work — say the word and it can be built behind a flag, but it needs a
real GPU instance to prove out before it carries traffic.

### Free throughput before buying anything: shrink the images

Input size is the largest client-side lever, measured on the `fast` tier at
16-way concurrency:

| Probe resolution | Payload | Throughput |
|---|---|---|
| 640 x 854 | ~74 KB | **37.7 req/s** |
| 900 x 1200 | ~121 KB | 29.5 req/s |
| 1200 x 1600 | ~183 KB | 30.8 req/s |
| 1800 x 2400 | ~332 KB | 15.6 req/s |

**Capturing at 640x854 instead of 1800x2400 is ~2.4x throughput for free** — a
third of the fleet, no accuracy change. The detector letterboxes to 512x512
regardless and the recognizer works from a 112x112 crop, so a full-resolution
phone capture spends its extra pixels on JPEG decode and nothing else.

If the mobile client can be changed, do this before adding a single instance. It
is the cheapest 2x available and it also cuts S3 egress and upload latency.

`ML_SERVICE_FACE_MAX_INPUT_SIDE` caps the image *after* decode, so it protects
the detector but not the decode itself — the saving has to happen at capture.

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

## Storage: nothing should accumulate

The verification path persists no images and no embeddings. Left at defaults,
though, three things still grew on disk — and all three scale with request rate,
so they are smallest exactly when you are testing and largest in production.

| Source | At 1000 req/s | Now |
|---|---|---|
| API access log (one line per request) | ~13 GB/day | **off** (`ZEPIRIS_ACCESS_LOG=false`) |
| ML access log (duplicate of the same request) | ~13 GB/day | **off** (`ML_SERVICE_ACCESS_LOG=false`) |
| Threshold-calibration log (one JSON line per verification) | ~17 GB/day | **off** (`ZEPIRIS_LEARNING_ENABLED=false`) |
| Multipart spooling images over 1 MB to temp files | ~2000 file writes/s | **removed** (framed body) |
| Docker `json-file` driver | unbounded | capped at 30 MB/container |

### The temp-file one was invisible

Starlette's multipart parser spools any part over 1 MB to a real temporary
**file**. Since a normal phone photo exceeds that, every such request wrote both
images to disk and read them back — on a service whose entire design claim is
that it persists nothing.

`/v1/face/match` now takes both images in one framed binary body: a 4-byte
big-endian length, the probe, then the reference (`zepiris/framing.py`). That
keeps everything in memory and skips boundary scanning. Verified with a 5.91 MB
payload — well over the 1 MB threshold — creating zero temp files.

### Access logs

Off by default on both services. At 10 req/s a line per request is free; at 1000
it is 13 GB/day, and it duplicates what the load balancer already records with
retention controls the instance disk does not have. The ALB is the request log.

Startup lines still print (which tier loaded, the resolved concurrency limit,
whether warm-up ran) — one-time, and the first thing you want when an instance
misbehaves. `LOG_LEVEL=WARNING` silences even those.

### Threshold calibration

Off by default now. Beyond the volume, it is process-local: on an autoscaled
fleet the file dies with the instance, so samples are never joined with the
feedback that would calibrate anything. Turn it on only with a real destination
— a shared volume or a datastore — and a retention policy.

### Read-only containers

Both containers run `read_only: true` with a 64 MB tmpfs on `/tmp`. Nothing is
written at runtime, so anything that tries now fails at the point of the bug
rather than quietly filling the disk over weeks. Verified: 200 requests created
no files in the working tree, no learning directory, and left neither process
holding a temp file open.

> Not yet verified end to end in Docker — the container config above was written
> without a running Docker daemon to test against. Run `docker compose up` once
> and confirm both containers reach healthy before relying on it in production.

40 GB gp3 per instance is then ample: the AMI, the images, and the model cache,
with nothing growing underneath them.

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
