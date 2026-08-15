#!/usr/bin/env python
"""Drive N concurrent verifications and report the latency distribution.

Measures what a caller actually experiences: how long a verification takes while
N others are in flight. Averages hide the answer — a run can average 300 ms and
still miss a 500 ms deadline on a third of requests — so this reports
percentiles, and checks p95 against the deadline rather than the mean.

Two modes:

    --target api   full path: HTTP -> API -> ML service (default)
    --target ml    the ML service alone, isolating model time from API overhead

Usage:
    python scripts/loadtest.py --concurrency 100 --requests 500
    python scripts/loadtest.py --concurrency 100 --deadline-ms 500 --image face.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import itertools
import statistics
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; values must be sorted."""
    if not values:
        return float("nan")
    k = max(0, min(len(values) - 1, int(round(pct / 100.0 * len(values) + 0.5)) - 1))
    return values[k]


def _encode(bgr: np.ndarray, width: int, height: int) -> bytes:
    big = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise SystemExit("error: failed to encode the probe image")
    return buf.tobytes()


def _build_images(
    path: str | None, width: int, height: int, count: int
) -> list[tuple[bytes, bytes]]:
    """Build ``count`` distinct (probe, reference) pairs.

    Every pair must differ from every other. Replaying one image would let any
    content-keyed cache in the service answer from memory after the first
    request, which real traffic — a different person each time — never does. A
    benchmark that allows that measures the cache, not the pipeline.

    Sources come from ``gallery/`` (or ``--image``), and each pair gets a unique
    faint noise pattern so no two requests are byte-identical even when they
    reuse the same face.
    """
    sources: list[np.ndarray] = []
    if path:
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"error: could not decode {path}")
        sources.append(img)
    else:
        gallery = sorted(Path("gallery").glob("*.jpg")) if Path("gallery").is_dir() else []
        if not gallery:
            raise SystemExit(
                "error: no --image given and gallery/ has no .jpg files to work from"
            )
        sources = [cv2.imread(str(p)) for p in gallery]
        sources = [s for s in sources if s is not None]
    if not sources:
        raise SystemExit("error: no usable source images")

    rng = np.random.default_rng(1234)
    pairs: list[tuple[bytes, bytes]] = []
    for i in range(count):
        src = sources[i % len(sources)]
        # Same identity on both sides (so pairs genuinely match), each side
        # perturbed uniquely so the bytes are never repeated.
        probe = np.clip(
            src.astype(np.int16) + rng.integers(-3, 4, src.shape, dtype=np.int16), 0, 255
        ).astype(np.uint8)
        ref = np.clip(
            src.astype(np.int16) + rng.integers(-3, 4, src.shape, dtype=np.int16), 0, 255
        ).astype(np.uint8)
        pairs.append((_encode(probe, width, height), _encode(ref, width, height)))
    return pairs


async def _one_api(client: httpx.AsyncClient, url: str, payload: dict) -> tuple[float, int, str]:
    t = time.perf_counter()
    try:
        r = await client.post(url, json=payload)
        elapsed = (time.perf_counter() - t) * 1000
        if r.status_code != 200:
            return elapsed, r.status_code, r.text[:120]
        body = r.json()
        if body.get("verificationResult", {}).get("score") is None:
            return elapsed, 200, f"unscored: faceDetected={body.get('faceDetected')}"
        return elapsed, 200, ""
    except Exception as e:  # noqa: BLE001 — any client-side failure is a failed request
        return (time.perf_counter() - t) * 1000, 0, f"{type(e).__name__}: {e}"


async def _one_ml(
    client: httpx.AsyncClient, url: str, pair: tuple[bytes, bytes]
) -> tuple[float, int, str]:
    t = time.perf_counter()
    try:
        r = await client.post(
            url,
            files={
                "probe": ("p", pair[0], "application/octet-stream"),
                "reference": ("r", pair[1], "application/octet-stream"),
            },
        )
        elapsed = (time.perf_counter() - t) * 1000
        if r.status_code != 200:
            return elapsed, r.status_code, r.text[:120]
        if r.json().get("score") is None:
            return elapsed, 200, "unscored: no face detected"
        return elapsed, 200, ""
    except Exception as e:  # noqa: BLE001
        return (time.perf_counter() - t) * 1000, 0, f"{type(e).__name__}: {e}"


async def run(args: argparse.Namespace) -> int:
    # One distinct pair per request, so no cache anywhere can shortcut the work.
    pairs = _build_images(args.image, args.width, args.height, args.requests)
    avg_kb = sum(len(p) + len(r) for p, r in pairs) / len(pairs) / 2 / 1024
    print(f"{len(pairs)} distinct image pairs, ~{avg_kb:.0f} KB per image")

    counter = itertools.count()

    if args.target == "api":
        url = f"{args.base_url.rstrip('/')}/v1/faces/facematch/verify"
        payloads = [
            {
                "face_check_b64": base64.b64encode(p).decode(),
                "source_selfie_b64": base64.b64encode(r).decode(),
            }
            for p, r in pairs
        ]

        async def fire(client):
            return await _one_api(client, url, payloads[next(counter) % len(payloads)])
    else:
        url = f"{args.base_url.rstrip('/')}/v1/face/match"

        async def fire(client):
            return await _one_ml(client, url, pairs[next(counter) % len(pairs)])

    limits = httpx.Limits(
        max_connections=args.concurrency + 10,
        max_keepalive_connections=args.concurrency + 10,
    )
    timeout = httpx.Timeout(args.timeout, connect=10.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Warm the models and the connection pool so the first requests are not
        # measuring a cold ONNX session.
        print("warming up …")
        warm = await asyncio.gather(*(fire(client) for _ in range(min(4, args.concurrency))))
        for _lat, status, err in warm:
            if status != 200 or err:
                print(f"  warmup issue: status={status} {err}")

        print(f"running {args.requests} requests at concurrency {args.concurrency} …")
        sem = asyncio.Semaphore(args.concurrency)
        results: list[tuple[float, int, str]] = []

        async def bounded():
            async with sem:
                results.append(await fire(client))

        started = time.perf_counter()
        await asyncio.gather(*(bounded() for _ in range(args.requests)))
        wall = time.perf_counter() - started

    ok = [r for r in results if r[1] == 200 and not r[2]]
    unscored = [r for r in results if r[1] == 200 and r[2]]
    shed = [r for r in results if r[1] == 503]
    failed = [r for r in results if r[1] not in (200, 503)]

    lat = sorted(r[0] for r in ok)
    print()
    print(f"wall time        {wall:.2f} s")
    # Successful throughput, not attempted: a service that fails instantly would
    # otherwise post the best number in the report.
    print(f"throughput       {len(ok) / wall:.1f} req/s  (successful)")
    if len(ok) != len(results):
        print(f"  attempted      {len(results) / wall:.1f} req/s")
    print(f"succeeded        {len(ok)}/{len(results)}")
    if unscored:
        print(f"unscored         {len(unscored)}  (e.g. {unscored[0][2]})")
    if shed:
        print(f"shed (503)       {len(shed)}  — over capacity, rejected fast")
    if failed:
        print(f"failed           {len(failed)}  (e.g. status={failed[0][1]} {failed[0][2]})")

    if not lat:
        print("\nno successful requests to report latency for")
        return 1

    print()
    print(f"latency  min     {lat[0]:8.0f} ms")
    print(f"         p50     {_percentile(lat, 50):8.0f} ms")
    print(f"         p95     {_percentile(lat, 95):8.0f} ms")
    print(f"         p99     {_percentile(lat, 99):8.0f} ms")
    print(f"         max     {lat[-1]:8.0f} ms")
    print(f"         mean    {statistics.mean(lat):8.0f} ms")

    if args.deadline_ms:
        p95 = _percentile(lat, 95)
        within = sum(x <= args.deadline_ms for x in lat) / len(lat) * 100
        print()
        verdict = "PASS" if p95 <= args.deadline_ms and not failed else "FAIL"
        print(f"{verdict}: p95 {p95:.0f} ms vs {args.deadline_ms:.0f} ms deadline "
              f"({within:.1f}% of requests within deadline)")
        return 0 if verdict == "PASS" else 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--target", choices=("api", "ml"), default="api",
                    help="'api' for the full path, 'ml' to isolate model time")
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--image", help="Probe image; synthesized from gallery/ if omitted")
    ap.add_argument("--width", type=int, default=1200, help="Synthesized probe width")
    ap.add_argument("--height", type=int, default=1600, help="Synthesized probe height")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--deadline-ms", type=float, default=500.0,
                    help="p95 target; 0 to skip the pass/fail check")
    args = ap.parse_args()

    if args.requests < args.concurrency:
        args.requests = args.concurrency
        print(f"note: raising --requests to {args.requests} to fill the concurrency level")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
