#!/usr/bin/env python
"""Replay recorded production go-online events through the live API.

Synthetic benchmark images answer "how fast is the pipeline"; they cannot answer
"does it still give the same verdict on the traffic we actually get". This
replays real events — the S3 URL of the go-online capture as ``face_check_s3``
and the rider's enrolled selfie as ``source_selfie_s3`` — through
``POST /v1/faces/facematch/verify``, exactly as production calls it.

It reports two comparisons, and they answer different questions:

  * **run-to-run** — the same rows replayed twice against different service
    settings. Any score that moves is a regression: caching and parallel
    embedding must be invisible in the answer.
  * **against the record** — today's score versus the score stored in the
    recording. These *are* expected to differ when the recording predates a
    detector or tier change, since alignment keypoints move; the point is to see
    how much, not to demand zero.

Input is the JSONL a recording run produced: one object per event with
``go_online_url``, ``enrolled_url`` and (optionally) the ``score`` it saw.

Usage:
    python scripts/prod_replay.py --rows 250
    python scripts/prod_replay.py --jsonl /path/results.jsonl --rows 100 --concurrency 4
    python scripts/prod_replay.py --rows 100 --repeat 2   # same rider going online twice
    python scripts/prod_replay.py --rows 250 --save reports/prod-replay.json

Nothing is written back to the recording, and no image is stored locally: the API
fetches each URL itself, which is also what makes this exercise the real S3 path.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

DEFAULT_JSONL = "/Users/lsn-akshat/Desktop/LSN-Github/PROD test 2 distinct/results.jsonl"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


def _load(path: str, rows: int) -> list[dict]:
    """Read the recording, keeping only events that carry both image URLs."""
    events: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            probe, reference = rec.get("go_online_url"), rec.get("enrolled_url")
            if not probe or not reference:
                continue
            events.append(
                {
                    "event_id": rec.get("event_id"),
                    "user_id": rec.get("user_id"),
                    "probe": probe,
                    "reference": reference,
                    "recorded_score": rec.get("score"),
                    "recorded_face_detected": rec.get("face_detected"),
                }
            )
            if rows and len(events) >= rows:
                break
    return events


async def _one(client: httpx.AsyncClient, url: str, event: dict) -> dict:
    """One verification, reported the way the caller experiences it."""
    payload = {"face_check_s3": event["probe"], "source_selfie_s3": event["reference"]}
    event = {**event, "request_body": json.dumps(payload)}
    started = time.perf_counter()
    try:
        r = await client.post(url, json=payload)
        elapsed = (time.perf_counter() - started) * 1000
    except Exception as e:  # noqa: BLE001 — a client-side failure is a failed request
        return {
            **event,
            "status": 0,
            "ms": (time.perf_counter() - started) * 1000,
            "error": f"{type(e).__name__}: {e}",
        }

    out = {**event, "status": r.status_code, "ms": elapsed, "error": ""}
    if r.status_code != 200:
        out["error"] = r.text[:160]
        return out
    body = r.json()
    verdict = body.get("verificationResult") or {}
    out.update(
        score=verdict.get("score"),
        is_match=verdict.get("isMatch"),
        threshold=verdict.get("threshold"),
        face_detected=body.get("faceDetected"),
        decode_failed=bool(body.get("decodeFailed")),
    )
    return out


async def _replay(events: list[dict], args: argparse.Namespace) -> list[dict]:
    url = f"{args.base_url.rstrip('/')}/v1/faces/facematch/verify"
    pool = args.concurrency + 8
    limits = httpx.Limits(max_connections=pool, max_keepalive_connections=pool)
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async with httpx.AsyncClient(
        limits=limits, timeout=httpx.Timeout(args.timeout, connect=10.0)
    ) as client:

        async def bounded(event: dict) -> None:
            async with sem:
                results.append(await _one(client, url, event))

        # ``repeat`` replays the whole set again — the same rider going online a
        # second time, which is when a reference cache can hit at all.
        work = [e for _ in range(args.repeat) for e in events]
        started = time.perf_counter()
        await asyncio.gather(*(bounded(e) for e in work))
        args._wall = time.perf_counter() - started
    return results


def _report(results: list[dict], args: argparse.Namespace) -> dict:
    ok = [r for r in results if r["status"] == 200 and not r["error"]]
    # A burst separates into three very different outcomes, and collapsing them
    # into "failed" hides the one that matters: 503 means the service refused
    # fast and on purpose, which is the designed behaviour over capacity.
    shed = [r for r in results if r["status"] == 503]
    client = [r for r in results if r["status"] == 0]
    failed = [
        r
        for r in results
        if r["status"] not in (200, 503, 0) or (r["status"] == 200 and r["error"])
    ]
    scored = [r for r in ok if r.get("score") is not None]
    no_face = [r for r in ok if r.get("face_detected") is False]
    lat = [r["ms"] for r in ok]

    print()
    print(
        f"replayed         {len(results)} requests ({args.repeat}x {len(results) // max(1, args.repeat)} events)"
    )
    print(f"wall time        {args._wall:.1f} s   throughput {len(ok) / args._wall:.1f} req/s")
    print(f"succeeded        {len(ok)}/{len(results)}")
    if shed:
        print(
            f"shed (503)       {len(shed)}  ({len(shed) / len(results) * 100:.0f}%) "
            f"— over capacity, refused fast rather than queued past the caller's timeout"
        )
    if client:
        print(f"client-side      {len(client)}  e.g. {client[0]['error'][:80]}")
    if no_face:
        print(f"no face found    {len(no_face)}  (reported as a normal verdict, not an error)")
    if failed:
        print(
            f"failed           {len(failed)}  e.g. status={failed[0]['status']} {failed[0]['error'][:90]}"
        )
    if shed or client:
        # Latency below covers served requests only: a shed request's 50 ms is not
        # a fast verification, and averaging it in would flatter the numbers.
        print("                 (latency percentiles below count served requests only)")
    if lat:
        print(
            f"latency          p50 {_percentile(lat, 50):.0f} ms   "
            f"p95 {_percentile(lat, 95):.0f} ms   p99 {_percentile(lat, 99):.0f} ms   "
            f"max {max(lat):.0f} ms"
        )

    matched = sum(1 for r in scored if r.get("is_match"))
    if scored:
        print(
            f"verdicts         {matched} match / {len(scored) - matched} no-match "
            f"of {len(scored)} scored"
        )

    # Against the recording: informative, not pass/fail — a detector change moves
    # alignment, and therefore scores, without either run being wrong.
    drifted = [
        (r["recorded_score"], r["score"])
        for r in scored
        if isinstance(r.get("recorded_score"), (int, float))
    ]
    if drifted:
        deltas = [abs(a - b) for a, b in drifted]
        flips = sum(1 for a, b in drifted if (a >= args.threshold) != (b >= args.threshold))
        print()
        print(f"vs the recording ({len(drifted)} comparable events)")
        print(
            f"  |Δscore|       p50 {statistics.median(deltas):.4f}   p95 {_percentile(deltas, 95):.4f}   max {max(deltas):.4f}"
        )
        print(
            f"  verdict flips  {flips}  ({flips / len(drifted) * 100:.1f}% crossed the {args.threshold} threshold)"
        )

    return {
        "requests": len(results),
        "ok": len(ok),
        "shed_503": len(shed),
        "client_errors": len(client),
        "failed": len(failed),
        "no_face": len(no_face),
        "wall_s": round(args._wall, 2),
        "rps": round(len(ok) / args._wall, 2) if args._wall else 0.0,
        "p50_ms": round(_percentile(lat, 50)) if lat else None,
        "p95_ms": round(_percentile(lat, 95)) if lat else None,
        "p99_ms": round(_percentile(lat, 99)) if lat else None,
        "scored": len(scored),
        "matched": matched,
        # Keyed per event so a second run can be compared row by row.
        "scores": {f"{r['event_id']}": r.get("score") for r in ok if r.get("event_id") is not None},
    }


def _write_per_request(path: Path, results: list[dict]) -> None:
    """One row per request: what was sent, what came back, how long it took.

    Ordered by the time each request finished, which is the order the service
    actually served them — not the order they were submitted in, since a burst
    does not complete in the order it arrives.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seq",
        "event_id",
        "user_id",
        "status",
        "response_ms",
        "score",
        "is_match",
        "face_detected",
        "error",
        "request_body",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(results, 1):
            w.writerow(
                {
                    "seq": i,
                    "event_id": r.get("event_id"),
                    "user_id": r.get("user_id"),
                    "status": r.get("status"),
                    "response_ms": round(r["ms"]),
                    "score": r.get("score"),
                    "is_match": r.get("is_match"),
                    "face_detected": r.get("face_detected"),
                    "error": (r.get("error") or "")[:200],
                    "request_body": r.get("request_body", ""),
                }
            )
    print(f"per-request -> {path}  ({len(results)} rows)")


def _print_slowest(results: list[dict], n: int) -> None:
    print()
    print(f"slowest {n} requests")
    print(f"  {'ms':>7}  {'status':>6}  {'score':>8}  event / body")
    for r in sorted(results, key=lambda x: -x["ms"])[:n]:
        score = "—" if r.get("score") is None else f"{r['score']:.4f}"
        print(
            f"  {r['ms']:7.0f}  {r['status']:>6}  {score:>8}  "
            f"event {r.get('event_id')}  {r.get('request_body', '')[:150]}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", default=DEFAULT_JSONL, help="Recording to replay")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--rows", type=int, default=250, help="Events to replay (0 = all)")
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Replay the whole set N times — a second pass is where a reference cache can hit",
    )
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to count verdict flips against the recording",
    )
    ap.add_argument("--save", help="Write the summary here as JSON")
    ap.add_argument("--per-request", help="Write one CSV row per request: body, status, ms, score")
    ap.add_argument(
        "--show-slowest", type=int, default=0, help="Print the N slowest requests after the summary"
    )
    args = ap.parse_args()

    if not Path(args.jsonl).exists():
        raise SystemExit(f"error: no recording at {args.jsonl}")

    events = _load(args.jsonl, args.rows)
    if not events:
        raise SystemExit("error: the recording carried no usable event URLs")
    print(f"{len(events)} events from {Path(args.jsonl).name}")
    print(
        f"  {len({e['reference'] for e in events})} distinct enrolled selfies "
        f"(the reference cache can only hit on a repeat)"
    )

    results = asyncio.run(_replay(events, args))
    summary = _report(results, args)

    if args.per_request:
        _write_per_request(Path(args.per_request), results)
    if args.show_slowest:
        _print_slowest(results, args.show_slowest)

    if args.save:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nsaved -> {p}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
