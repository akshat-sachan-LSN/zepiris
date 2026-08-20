#!/usr/bin/env python
"""Step concurrency up a ladder and report what breaks, and where.

A single-concurrency run answers "can it hold 100 users?". This answers the
question you actually ask before a launch: "how far does it hold, and what does
the user feel at each step on the way?" It walks a ladder of concurrency levels
— 10, 20, 50, 70, 100, 150, 200 by default — running a full burst at each and
recording the latency distribution, so the knee in the curve is visible instead
of inferred.

Each level is a fresh burst against a warmed service. Levels run in order and
never overlap, so a slow level cannot poison the next one's numbers.

Usage:
    python scripts/loadtest_ramp.py
    python scripts/loadtest_ramp.py --levels 10,50,100 --requests-per-user 3
    python scripts/loadtest_ramp.py --target ml --html out/ramp.html
    python scripts/loadtest_ramp.py --base-url http://65.0.4.214:8000

Outputs a table on stdout, plus optional --json and --html reports.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import html
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.loadtest import (  # noqa: E402
    _build_images,
    _one_api,
    _one_ml,
    _percentile,
)

DEFAULT_LEVELS = (10, 20, 50, 70, 100, 150, 200)


def _summarise(
    level: int,
    results: list[tuple[float, int, str]],
    wall: float,
    deadline_ms: float,
) -> dict:
    """Reduce one level's raw results to the numbers worth reporting."""
    ok = [r for r in results if r[1] == 200 and not r[2]]
    unscored = [r for r in results if r[1] == 200 and r[2]]
    shed = [r for r in results if r[1] == 503]
    failed = [r for r in results if r[1] not in (200, 503)]
    lat = sorted(r[0] for r in ok)

    row = {
        "users": level,
        "requests": len(results),
        "wall_s": round(wall, 2),
        # Successful throughput only — a service failing instantly must not post
        # the best number in the report.
        "rps": round(len(ok) / wall, 1) if wall > 0 else 0.0,
        "ok": len(ok),
        "unscored": len(unscored),
        "shed_503": len(shed),
        "failed": len(failed),
        "first_error": (unscored or shed or failed or [(0, 0, "")])[0][2][:120],
    }
    if lat:
        row.update(
            min_ms=round(lat[0]),
            p50_ms=round(_percentile(lat, 50)),
            p95_ms=round(_percentile(lat, 95)),
            p99_ms=round(_percentile(lat, 99)),
            max_ms=round(lat[-1]),
            mean_ms=round(statistics.mean(lat)),
            within_deadline_pct=(
                round(sum(x <= deadline_ms for x in lat) / len(lat) * 100, 1)
                if deadline_ms
                else None
            ),
        )
    else:
        row.update(
            min_ms=None,
            p50_ms=None,
            p95_ms=None,
            p99_ms=None,
            max_ms=None,
            mean_ms=None,
            within_deadline_pct=None,
        )
    return row


async def _run_level(
    client: httpx.AsyncClient, fire, level: int, requests: int
) -> tuple[list[tuple[float, int, str]], float]:
    """Fire ``requests`` calls with at most ``level`` in flight at once."""
    sem = asyncio.Semaphore(level)
    results: list[tuple[float, int, str]] = []

    async def bounded():
        async with sem:
            results.append(await fire(client))

    started = time.perf_counter()
    await asyncio.gather(*(bounded() for _ in range(requests)))
    return results, time.perf_counter() - started


def _print_table(rows: list[dict], deadline_ms: float) -> None:
    head = (
        f"{'users':>6} {'reqs':>6} {'ok':>6} {'503':>5} {'fail':>5} "
        f"{'rps':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}"
    )
    if deadline_ms:
        head += f" {'≤' + str(int(deadline_ms)) + 'ms':>9}"
    print()
    print(head)
    print("-" * len(head))
    for r in rows:
        line = (
            f"{r['users']:>6} {r['requests']:>6} {r['ok']:>6} {r['shed_503']:>5} "
            f"{r['failed']:>5} {r['rps']:>7.1f} "
            f"{_fmt(r['p50_ms']):>7} {_fmt(r['p95_ms']):>7} "
            f"{_fmt(r['p99_ms']):>7} {_fmt(r['max_ms']):>7}"
        )
        if deadline_ms:
            line += f" {_pct(r['within_deadline_pct']):>9}"
        print(line)


def _fmt(v: int | None) -> str:
    return "—" if v is None else str(v)


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f}%"


def _knee(rows: list[dict], deadline_ms: float) -> str:
    """The first level that misses the deadline or drops requests."""
    for r in rows:
        if r["failed"] or r["shed_503"]:
            return (
                f"{r['users']} users — service started rejecting/erroring "
                f"({r['shed_503']} shed, {r['failed']} failed)"
            )
        if deadline_ms and r["p95_ms"] is not None and r["p95_ms"] > deadline_ms:
            return f"{r['users']} users — p95 {r['p95_ms']} ms crossed the {int(deadline_ms)} ms deadline"
    return f"none — held all the way to {rows[-1]['users']} users"


def _write_html(path: Path, rows: list[dict], meta: dict) -> None:
    """Write a self-contained report: the table plus a latency/throughput chart."""
    max_p99 = max((r["p99_ms"] or 0) for r in rows) or 1
    max_rps = max((r["rps"] or 0) for r in rows) or 1

    bars = []
    for r in rows:
        p95 = r["p95_ms"] or 0
        p99 = r["p99_ms"] or 0
        bars.append(
            f'<div class="bar"><div class="bl">{r["users"]}</div>'
            f'<div class="track"><div class="p99" style="width:{p99 / max_p99 * 100:.1f}%"></div>'
            f'<div class="p95" style="width:{p95 / max_p99 * 100:.1f}%"></div></div>'
            f'<div class="bv">{_fmt(r["p95_ms"])} / {_fmt(r["p99_ms"])} ms</div>'
            f'<div class="track rps"><div class="rf" style="width:{(r["rps"] or 0) / max_rps * 100:.1f}%"></div></div>'
            f'<div class="bv">{r["rps"]:.1f} rps</div></div>'
        )

    trs = []
    for r in rows:
        bad = r["failed"] or r["shed_503"]
        over = meta["deadline_ms"] and r["p95_ms"] and r["p95_ms"] > meta["deadline_ms"]
        cls = "row bad" if bad else ("row over" if over else "row")
        trs.append(
            f'<tr class="{cls}"><td>{r["users"]}</td><td>{r["requests"]}</td>'
            f"<td>{r['ok']}</td><td>{r['shed_503']}</td><td>{r['failed']}</td>"
            f"<td>{r['rps']:.1f}</td><td>{_fmt(r['p50_ms'])}</td><td>{_fmt(r['p95_ms'])}</td>"
            f"<td>{_fmt(r['p99_ms'])}</td><td>{_fmt(r['max_ms'])}</td>"
            f"<td>{_pct(r['within_deadline_pct'])}</td></tr>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ZepIris · Face-match load ramp</title>
<style>
 :root{{--bg:#0b0f17;--panel:#121826;--line:#1f2937;--muted:#8b97a7;--text:#e8edf4;
   --accent:#5b8cff;--ok:#1fbf75;--warn:#f5a524;--no:#ff5d6c;}}
 *{{box-sizing:border-box}}
 body{{margin:0;font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
   background:radial-gradient(1200px 600px at 50% -10%,#16203a,var(--bg));color:var(--text);min-height:100vh}}
 .wrap{{max-width:1000px;margin:0 auto;padding:32px 20px 64px}}
 h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 12px;color:var(--muted);
   text-transform:uppercase;letter-spacing:.09em}}
 .sub{{color:var(--muted);margin:0 0 24px;font-size:13px}}
 .card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;margin-bottom:18px}}
 .meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;font-size:13px}}
 .meta div span{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}
 table{{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
 th,td{{padding:9px 8px;text-align:right;border-bottom:1px solid var(--line)}}
 th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.07em;font-weight:600}}
 th:first-child,td:first-child{{text-align:left;font-weight:600}}
 tr.over td{{color:var(--warn)}} tr.bad td{{color:var(--no)}}
 .bar{{display:grid;grid-template-columns:44px 1fr 120px 1fr 84px;align-items:center;gap:10px;margin-bottom:9px;font-size:12px}}
 .bl{{color:var(--muted)}} .bv{{color:var(--muted);font-variant-numeric:tabular-nums}}
 .track{{position:relative;height:16px;background:#0e1422;border:1px solid var(--line);border-radius:8px;overflow:hidden}}
 .p99{{position:absolute;inset:0 auto 0 0;background:#2c3e6b}}
 .p95{{position:absolute;inset:0 auto 0 0;background:var(--accent)}}
 .rf{{position:absolute;inset:0 auto 0 0;background:var(--ok)}}
 .legend{{color:var(--muted);font-size:12px;margin-top:14px}}
 .k{{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 5px 0 12px;vertical-align:middle}}
 .verdict{{font-size:15px;font-weight:600}}
 code{{background:#0e1422;border:1px solid var(--line);border-radius:6px;padding:2px 6px;font-size:12px}}
</style></head><body><div class="wrap">
<h1>Face-match load ramp</h1>
<p class="sub">Concurrency stepped {rows[0]["users"]} &rarr; {rows[-1]["users"]} simultaneous users against
<code>{html.escape(meta["url"])}</code></p>

<div class="card"><div class="meta">
 <div><span>target</span>{html.escape(meta["target"])}</div>
 <div><span>image size</span>{meta["width"]}&times;{meta["height"]}, ~{meta["avg_kb"]:.0f} KB</div>
 <div><span>requests / user</span>{meta["requests_per_user"]}</div>
 <div><span>p95 deadline</span>{int(meta["deadline_ms"]) if meta["deadline_ms"] else "—"} ms</div>
 <div><span>run</span>{html.escape(meta["when"])}</div>
</div></div>

<h2>Latency &amp; throughput by level</h2>
<div class="card">
{"".join(bars)}
<div class="legend"><span class="k" style="background:var(--accent)"></span>p95
 <span class="k" style="background:#2c3e6b"></span>p99
 <span class="k" style="background:var(--ok)"></span>successful req/s</div>
</div>

<h2>Numbers</h2>
<div class="card"><table>
<tr><th>users</th><th>reqs</th><th>ok</th><th>503</th><th>fail</th><th>rps</th>
<th>p50</th><th>p95</th><th>p99</th><th>max</th><th>&le;deadline</th></tr>
{"".join(trs)}
</table></div>

<h2>Where it breaks</h2>
<div class="card"><p class="verdict">{html.escape(meta["knee"])}</p></div>
</div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if not levels:
        raise SystemExit("error: --levels parsed to nothing")
    peak_requests = max(levels) * args.requests_per_user

    # One distinct pair per request at the largest level, reused across levels —
    # no cache anywhere can shortcut the work.
    pairs = _build_images(args.image, args.width, args.height, peak_requests)

    if args.reference_pool > 0:
        # Real traffic is not one fresh enrolled selfie per verification: a
        # population of N users verifies repeatedly against the *same* N enrolled
        # references, while every probe is a new capture. Collapsing the reference
        # side onto a pool of that size is what lets the service's
        # reference-embedding cache behave as it does in production; leaving every
        # reference unique measures a permanent-cache-miss world that exists only
        # for the first request per user.
        refs = [r for _, r in pairs[: args.reference_pool]]
        pairs = [(probe, refs[i % len(refs)]) for i, (probe, _) in enumerate(pairs)]
        print(f"reference pool: {len(refs)} enrolled selfies, every probe unique")

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

    pool = max(levels) + 20
    limits = httpx.Limits(max_connections=pool, max_keepalive_connections=pool)
    timeout = httpx.Timeout(args.timeout, connect=10.0)

    rows: list[dict] = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Warm the ONNX sessions and the pool, so level 1 is not measuring a cold start.
        print("warming up …")
        for _lat, status, err in await asyncio.gather(*(fire(client) for _ in range(4))):
            if status != 200 or err:
                print(f"  warmup issue: status={status} {err}")

        for level in levels:
            requests = level * args.requests_per_user
            print(f"\n== {level} concurrent users · {requests} requests ==")
            results, wall = await _run_level(client, fire, level, requests)
            row = _summarise(level, results, wall, args.deadline_ms)
            rows.append(row)
            print(
                f"   ok {row['ok']}/{row['requests']}  "
                f"p50 {_fmt(row['p50_ms'])} ms  p95 {_fmt(row['p95_ms'])} ms  "
                f"p99 {_fmt(row['p99_ms'])} ms  {row['rps']:.1f} rps"
            )
            if row["first_error"]:
                print(f"   note: {row['first_error']}")
            if args.settle > 0 and level != levels[-1]:
                await asyncio.sleep(args.settle)

    _print_table(rows, args.deadline_ms)
    knee = _knee(rows, args.deadline_ms)
    print(f"\nknee: {knee}")

    meta = {
        "url": url,
        "target": args.target,
        "width": args.width,
        "height": args.height,
        "avg_kb": avg_kb,
        "requests_per_user": args.requests_per_user,
        "deadline_ms": args.deadline_ms,
        "when": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "knee": knee,
    }
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"meta": meta, "levels": rows}, indent=2), encoding="utf-8")
        print(f"json  -> {p}")
    if args.html:
        _write_html(Path(args.html), rows, meta)
        print(f"html  -> {Path(args.html)}")

    # Non-zero only on hard failures: crossing a latency deadline is information,
    # not a broken service, and the table already says where it happened.
    return 1 if any(r["failed"] for r in rows) else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--target",
        choices=("api", "ml"),
        default="api",
        help="'api' for the full path, 'ml' to isolate model time",
    )
    ap.add_argument(
        "--levels",
        default=",".join(str(x) for x in DEFAULT_LEVELS),
        help="Comma-separated concurrency levels to step through",
    )
    ap.add_argument(
        "--requests-per-user",
        type=int,
        default=2,
        help="Requests each simulated user sends per level",
    )
    ap.add_argument(
        "--reference-pool",
        type=int,
        default=0,
        help="Distinct enrolled selfies to verify against, as real traffic would "
        "(0 = every reference unique, i.e. never a cache hit)",
    )
    ap.add_argument("--image", help="Probe image; synthesized from gallery/ if omitted")
    ap.add_argument("--width", type=int, default=1200, help="Synthesized probe width")
    ap.add_argument("--height", type=int, default=1600, help="Synthesized probe height")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument(
        "--settle", type=float, default=3.0, help="Seconds to idle between levels so queues drain"
    )
    ap.add_argument(
        "--deadline-ms",
        type=float,
        default=500.0,
        help="p95 target used for the ≤deadline column; 0 to skip",
    )
    ap.add_argument("--json", help="Write the raw per-level summary here")
    ap.add_argument("--html", help="Write a self-contained HTML report here")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
