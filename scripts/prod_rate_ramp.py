#!/usr/bin/env python
"""Send real prod events at a fixed *arrival rate*, stepping the rate up a ladder.

This is the open-loop counterpart to ``loadtest_ramp.py``. The difference matters
more than it sounds:

  * A **concurrency** test keeps N requests in flight — when the service slows
    down, the load generator slows down with it. It can never ask for more than
    the service can take, so it measures latency at a given occupancy.
  * A **rate** test launches requests on a clock and does not care whether the
    previous ones finished. Real traffic behaves this way: riders go online when
    they go online. If the arrival rate exceeds what the service can serve, the
    backlog grows for as long as the load lasts, and latency climbs without
    bound until requests are shed.

So a rate ladder is the test that finds the arrival rate a deployment can
actually absorb, rather than how it behaves once already saturated.

Each request is one real go-online event replayed through
``POST /v1/faces/facematch/verify`` with both images as S3 URLs — see
``prod_replay.py`` for the recording format.

Usage:
    python scripts/prod_rate_ramp.py
    python scripts/prod_rate_ramp.py --rates 10,25,50 --seconds 3
    python scripts/prod_rate_ramp.py --per-request reports/rate-requests.csv

Per-batch numbers land on stdout; ``--per-request`` writes every request with the
rate it was sent at, so "how long did each request take at 100/s" is a filter
away.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prod_replay import DEFAULT_JSONL, _load, _one, _percentile  # noqa: E402

DEFAULT_RATES = (10, 25, 50, 75, 100, 125, 150, 175, 200)


async def _fire_at_rate(
    client: httpx.AsyncClient,
    url: str,
    events: list[dict],
    rate: int,
    seconds: float,
    offset: int,
) -> tuple[list[dict], float]:
    """Launch ``rate * seconds`` requests spaced 1/rate apart, then await them all.

    Requests are launched on a wall clock and never wait for their predecessors —
    that is the whole point. ``launch_lag_ms`` records how late each launch was
    against its scheduled slot: if that grows, the generator itself fell behind
    and the rate reported is the rate actually offered, not the one requested.
    """
    total = max(1, int(round(rate * seconds)))
    interval = 1.0 / rate
    tasks: list[asyncio.Task] = []
    lags: list[float] = []

    started = time.perf_counter()
    for i in range(total):
        target = started + i * interval
        now = time.perf_counter()
        if target > now:
            await asyncio.sleep(target - now)
        lags.append(max(0.0, (time.perf_counter() - target) * 1000))
        event = events[(offset + i) % len(events)]
        tasks.append(asyncio.create_task(_one(client, url, event)))

    results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - started
    for r, lag in zip(results, lags, strict=True):
        r["launch_lag_ms"] = round(lag, 1)
    return results, wall


def _summarise(rate: int, results: list[dict], wall: float) -> dict:
    served = [r for r in results if r["status"] == 200 and not r["error"]]
    shed = [r for r in results if r["status"] == 503]
    client_err = [r for r in results if r["status"] == 0]
    other = [r for r in results if r["status"] not in (200, 503, 0)]
    lat = [r["ms"] for r in served]
    lags = [r["launch_lag_ms"] for r in results]

    return {
        "target_rps": rate,
        "attempted": len(results),
        "served": len(served),
        "shed_503": len(shed),
        "client_errors": len(client_err),
        "other_errors": len(other),
        "wall_s": round(wall, 1),
        # What the service actually got through, against what was asked of it.
        "achieved_rps": round(len(served) / wall, 1) if wall else 0.0,
        "offered_rps": round(len(results) / wall, 1) if wall else 0.0,
        "p50_ms": round(_percentile(lat, 50)) if lat else None,
        "p95_ms": round(_percentile(lat, 95)) if lat else None,
        "p99_ms": round(_percentile(lat, 99)) if lat else None,
        "max_ms": round(max(lat)) if lat else None,
        "mean_ms": round(statistics.mean(lat)) if lat else None,
        "max_launch_lag_ms": round(max(lags)) if lags else 0,
    }


def _print_table(rows: list[dict]) -> None:
    head = (
        f"{'target':>7} {'sent':>6} {'served':>7} {'shed':>6} {'err':>5} "
        f"{'got rps':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}"
    )
    print()
    print(head)
    print("-" * len(head))
    for r in rows:
        f = lambda v: "—" if v is None else str(v)  # noqa: E731
        print(
            f"{r['target_rps']:>6}/s {r['attempted']:>6} {r['served']:>7} "
            f"{r['shed_503']:>6} {r['client_errors'] + r['other_errors']:>5} "
            f"{r['achieved_rps']:>8} {f(r['p50_ms']):>8} {f(r['p95_ms']):>8} "
            f"{f(r['p99_ms']):>8} {f(r['max_ms']):>8}"
        )


def _absorbable(rows: list[dict]) -> str:
    """Highest rate that came back clean and under ~1s at the median."""
    good = [
        r
        for r in rows
        if not r["shed_503"]
        and not r["client_errors"]
        and not r["other_errors"]
        and r["p50_ms"] is not None
        and r["p50_ms"] <= 1000
    ]
    if not good:
        return "none of the tested rates stayed clean under 1s at the median"
    best = max(good, key=lambda r: r["target_rps"])
    return f"{best['target_rps']}/s (p50 {best['p50_ms']} ms, nothing shed)"


def _write_per_request(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target_rps",
        "seq_in_batch",
        "event_id",
        "user_id",
        "status",
        "response_ms",
        "launch_lag_ms",
        "score",
        "is_match",
        "face_detected",
        "error",
        "request_body",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "target_rps": r["target_rps"],
                    "seq_in_batch": r["seq_in_batch"],
                    "event_id": r.get("event_id"),
                    "user_id": r.get("user_id"),
                    "status": r.get("status"),
                    "response_ms": round(r["ms"]),
                    "launch_lag_ms": r.get("launch_lag_ms"),
                    "score": r.get("score"),
                    "is_match": r.get("is_match"),
                    "face_detected": r.get("face_detected"),
                    "error": (r.get("error") or "")[:200],
                    "request_body": r.get("request_body", ""),
                }
            )
    print(f"per-request -> {path}  ({len(rows)} rows)")


async def run(args: argparse.Namespace) -> int:
    events = _load(args.jsonl, args.rows)
    if not events:
        raise SystemExit("error: the recording carried no usable event URLs")
    rates = [int(x) for x in args.rates.split(",") if x.strip()]
    print(f"{len(events)} real events, replayed at {rates} req/s for {args.seconds}s each")
    print(f"  {len({e['reference'] for e in events})} distinct enrolled selfies")

    url = f"{args.base_url.rstrip('/')}/v1/faces/facematch/verify"
    # The pool has to be able to hold a full batch in flight at once, or the
    # generator would throttle itself and measure its own queue instead.
    pool = max(rates) * int(args.seconds) + 64
    limits = httpx.Limits(max_connections=pool, max_keepalive_connections=pool)

    rows: list[dict] = []
    every: list[dict] = []
    offset = 0
    async with httpx.AsyncClient(
        limits=limits, timeout=httpx.Timeout(args.timeout, connect=10.0)
    ) as client:
        for rate in rates:
            print(f"\n== {rate} req/s for {args.seconds}s ==")
            results, wall = await _fire_at_rate(client, url, events, rate, args.seconds, offset)
            offset += len(results)
            row = _summarise(rate, results, wall)
            rows.append(row)
            for i, r in enumerate(results, 1):
                every.append({**r, "target_rps": rate, "seq_in_batch": i})
            print(
                f"   served {row['served']}/{row['attempted']}  shed {row['shed_503']}  "
                f"p50 {row['p50_ms']} ms  p95 {row['p95_ms']} ms  max {row['max_ms']} ms  "
                f"got {row['achieved_rps']}/s"
            )
            if row["max_launch_lag_ms"] > 250:
                print(
                    f"   note: generator fell {row['max_launch_lag_ms']} ms behind schedule "
                    f"— offered rate was {row['offered_rps']}/s, not {rate}/s"
                )
            if rate != rates[-1] and args.settle:
                print(f"   draining {args.settle}s before the next rate …")
                await asyncio.sleep(args.settle)

    _print_table(rows)
    print(f"\nhighest rate absorbed cleanly: {_absorbable(rows)}")

    if args.per_request:
        _write_per_request(Path(args.per_request), every)
    if args.save:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"batches": rows}, indent=2), encoding="utf-8")
        print(f"summary -> {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", default=DEFAULT_JSONL)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--rates", default=",".join(str(r) for r in DEFAULT_RATES))
    ap.add_argument("--seconds", type=float, default=2.0, help="Duration of each rate step")
    ap.add_argument("--rows", type=int, default=400, help="Events to draw from (cycled)")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--settle", type=float, default=6.0, help="Drain gap between rates")
    ap.add_argument("--per-request", help="CSV of every request, tagged with its rate")
    ap.add_argument("--save", help="Write the per-batch summary here as JSON")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
