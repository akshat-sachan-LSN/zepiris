#!/usr/bin/env node
/**
 * Fire N verification requests at the API simultaneously and record every one.
 *
 *   node scripts/hit.js 20            -> 20 requests at once, 20 users hitting together
 *   node scripts/hit.js 50 --url https://api.example.com/v1/faces/facematch/verify
 *   node scripts/hit.js 20 --cases my-pairs.jsonl
 *   node scripts/hit.js 10 20 50      -> three waves, one after the other
 *
 * All N requests are launched in the same tick — no ramp, no pacing. That is the
 * point: it answers "what happens when 20 people go online at the same second",
 * which is a different question from "what happens at a sustained 20/s".
 *
 * Every hit is written out with its request body and its response body, so a
 * failure can be traced back to the exact payload that produced it. Output goes
 * to a JSON file (machine-readable) and an HTML file (readable), both named after
 * the run.
 *
 * No dependencies — Node 18+ only (global fetch).
 *
 * Options:
 *   --url <url>       endpoint to hit (default http://localhost:8000/v1/faces/facematch/verify)
 *   --cases <file>    request bodies to cycle through. Accepts either a JSON array
 *                     of bodies, or JSONL carrying go_online_url + enrolled_url
 *                     (the recording format from the prod test runs).
 *   --out <prefix>    output path prefix (default reports/hit-<n>)
 *   --timeout <sec>   per-request timeout (default 60)
 *   --label <text>    note stored with the run, e.g. "staging, cache off"
 *   --quiet           summary only, no per-hit console lines
 */

'use strict';

const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://localhost:8000/v1/faces/facematch/verify';

// Stand-in case used when no --cases file is given: same image on both sides, so
// the score should come back at 1.0 and anything else is a service problem
// rather than a bad match.
const DEFAULT_CASES = [
  {
    face_check_s3:
      'https://earn-de-docs.s3.ap-south-1.amazonaws.com/images/USER_2190283_1770149064803.jpg',
    source_selfie_s3:
      'https://earn-de-docs.s3.ap-south-1.amazonaws.com/images/USER_2190283_1770149064803.jpg',
  },
];

function parseArgs(argv) {
  const counts = [];
  const opts = {
    url: DEFAULT_URL,
    cases: null,
    out: null,
    timeout: 60,
    label: '',
    quiet: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--quiet') opts.quiet = true;
    else if (a === '--url') opts.url = argv[++i];
    else if (a === '--cases') opts.cases = argv[++i];
    else if (a === '--out') opts.out = argv[++i];
    else if (a === '--label') opts.label = argv[++i];
    else if (a === '--timeout') opts.timeout = Number(argv[++i]);
    else if (a === 'run') continue; // tolerate `hit.js run 20`
    else if (/^\d+$/.test(a)) counts.push(Number(a));
    else if (a === '-h' || a === '--help') {
      console.log(fs.readFileSync(__filename, 'utf8').split('*/')[0]);
      process.exit(0);
    } else {
      console.error(`unknown argument: ${a}`);
      process.exit(2);
    }
  }
  if (!counts.length) counts.push(20);
  return { counts, opts };
}

/** Load request bodies: a JSON array of bodies, or JSONL with the URL pair fields. */
function loadCases(file) {
  if (!file) return DEFAULT_CASES;
  const raw = fs.readFileSync(file, 'utf8').trim();
  if (raw.startsWith('[')) {
    const arr = JSON.parse(raw);
    if (!arr.length) throw new Error(`${file} held an empty array`);
    return arr;
  }
  const bodies = [];
  for (const line of raw.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    let rec;
    try {
      rec = JSON.parse(t);
    } catch {
      continue;
    }
    // Recording format from the prod runs, or a bare request body.
    const probe = rec.go_online_url || rec.face_check_s3;
    const reference = rec.enrolled_url || rec.source_selfie_s3;
    if (probe && reference) {
      bodies.push({
        face_check_s3: probe,
        source_selfie_s3: reference,
        _event_id: rec.event_id,
        _user_id: rec.user_id,
      });
    }
  }
  if (!bodies.length) throw new Error(`${file} held no usable cases`);
  return bodies;
}

async function oneHit(url, body, index, timeoutSec, t0) {
  // Strip the bookkeeping fields before sending; keep them for the report.
  const payload = { ...body };
  const meta = { event_id: payload._event_id, user_id: payload._user_id };
  delete payload._event_id;
  delete payload._user_id;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutSec * 1000);
  const started = performance.now();
  const record = {
    hit: index,
    event_id: meta.event_id ?? null,
    user_id: meta.user_id ?? null,
    launched_at_ms: Math.round(started - t0),
    request: payload,
  };

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const text = await res.text();
    record.response_ms = Math.round(performance.now() - started);
    record.status = res.status;
    try {
      record.response = JSON.parse(text);
    } catch {
      record.response = text.slice(0, 2000);
    }
    const v = record.response && record.response.verificationResult;
    if (v) {
      record.score = v.score ?? null;
      record.is_match = v.isMatch ?? null;
      record.threshold = v.threshold ?? null;
    }
    if (record.response && record.response.faceDetected !== undefined) {
      record.face_detected = record.response.faceDetected;
    }
    record.ok = res.status === 200;
    return record;
  } catch (e) {
    record.response_ms = Math.round(performance.now() - started);
    record.status = 0;
    record.ok = false;
    // An abort is a timeout, and saying so beats reporting "AbortError".
    record.error =
      e.name === 'AbortError' ? `timeout after ${timeoutSec}s` : `${e.name}: ${e.message}`;
    record.response = null;
    return record;
  } finally {
    clearTimeout(timer);
  }
}

function percentile(sorted, pct) {
  if (!sorted.length) return null;
  const k = Math.min(sorted.length - 1, Math.max(0, Math.round((pct / 100) * sorted.length + 0.5) - 1));
  return sorted[k];
}

function summarise(count, hits, wallMs) {
  const served = hits.filter((h) => h.ok);
  const shed = hits.filter((h) => h.status === 503);
  const timedOut = hits.filter((h) => h.status === 0);
  const other = hits.filter((h) => !h.ok && h.status !== 503 && h.status !== 0);
  const lat = served.map((h) => h.response_ms).sort((a, b) => a - b);
  return {
    concurrent_hits: count,
    wall_ms: Math.round(wallMs),
    served: served.length,
    shed_503: shed.length,
    timed_out: timedOut.length,
    other_errors: other.length,
    // Throughput over the wall clock of the wave: N requests all at once means
    // the wave is only as done as its slowest member.
    effective_rps: wallMs > 0 ? Number(((served.length / wallMs) * 1000).toFixed(1)) : 0,
    min_ms: lat.length ? lat[0] : null,
    p50_ms: percentile(lat, 50),
    p95_ms: percentile(lat, 95),
    p99_ms: percentile(lat, 99),
    max_ms: lat.length ? lat[lat.length - 1] : null,
    matched: served.filter((h) => h.is_match === true).length,
    not_matched: served.filter((h) => h.is_match === false).length,
    no_face: served.filter((h) => h.face_detected === false).length,
  };
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
  );
}

function renderHtml(runs, meta) {
  const waveBlocks = runs
    .map((run) => {
      const s = run.summary;
      const rows = run.hits
        .map((h) => {
          const cls = h.ok ? 'ok' : h.status === 503 ? 'shed' : 'bad';
          const verdict = h.ok
            ? h.is_match === true
              ? 'MATCH'
              : h.is_match === false
                ? 'no match'
                : h.face_detected === false
                  ? 'no face'
                  : '—'
            : h.error
              ? 'error'
              : `HTTP ${h.status}`;
          return `<tr class="${cls}">
  <td>${h.hit}</td>
  <td>${h.status || '—'}</td>
  <td class="num">${h.response_ms}</td>
  <td class="num">${h.launched_at_ms}</td>
  <td>${h.score === null || h.score === undefined ? '—' : Number(h.score).toFixed(4)}</td>
  <td>${verdict}</td>
  <td><details><summary>view</summary><pre>REQUEST
${esc(JSON.stringify(h.request, null, 2))}

RESPONSE${h.error ? ' (failed)' : ''}
${esc(h.error || JSON.stringify(h.response, null, 2))}</pre></details></td>
</tr>`;
        })
        .join('\n');

      return `<section class="card">
  <h2>${s.concurrent_hits} hits at once</h2>
  <div class="tiles">
    <div><span>served</span>${s.served}/${s.concurrent_hits}</div>
    <div><span>shed 503</span>${s.shed_503}</div>
    <div><span>timed out</span>${s.timed_out}</div>
    <div><span>other errors</span>${s.other_errors}</div>
    <div><span>wall</span>${(s.wall_ms / 1000).toFixed(1)}s</div>
    <div><span>effective rps</span>${s.effective_rps}</div>
    <div><span>fastest</span>${s.min_ms ?? '—'} ms</div>
    <div><span>p50</span>${s.p50_ms ?? '—'} ms</div>
    <div><span>p95</span>${s.p95_ms ?? '—'} ms</div>
    <div><span>slowest</span>${s.max_ms ?? '—'} ms</div>
    <div><span>match / no</span>${s.matched} / ${s.not_matched}</div>
    <div><span>no face</span>${s.no_face}</div>
  </div>
  <table>
    <tr><th>#</th><th>status</th><th>ms</th><th>launched +ms</th><th>score</th><th>verdict</th><th>req / resp</th></tr>
    ${rows}
  </table>
</section>`;
    })
    .join('\n');

  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>API hit test</title>
<style>
 :root{--bg:#0b0f17;--panel:#121826;--line:#1f2937;--muted:#8b97a7;--text:#e8edf4;
   --accent:#5b8cff;--ok:#1fbf75;--warn:#f5a524;--no:#ff5d6c;}
 *{box-sizing:border-box}
 body{margin:0;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
   background:radial-gradient(1200px 600px at 50% -10%,#16203a,var(--bg));color:var(--text);min-height:100vh}
 .wrap{max-width:1100px;margin:0 auto;padding:30px 18px 60px}
 h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:0 0 14px}
 .sub{color:var(--muted);font-size:13px;margin:0 0 22px}
 code{background:#0e1422;border:1px solid var(--line);border-radius:6px;padding:2px 6px;font-size:12px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:18px}
 .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:12px;margin-bottom:18px}
 .tiles div{background:#0e1422;border:1px solid var(--line);border-radius:10px;padding:10px 12px;
   font-variant-numeric:tabular-nums;font-weight:600}
 .tiles span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;
   letter-spacing:.07em;font-weight:500;margin-bottom:3px}
 table{width:100%;border-collapse:collapse;font-size:12.5px}
 th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
 th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
 td.num,td:nth-child(3),td:nth-child(4){text-align:right;font-variant-numeric:tabular-nums}
 tr.shed td{color:var(--warn)} tr.bad td{color:var(--no)}
 details summary{cursor:pointer;color:var(--accent)}
 pre{white-space:pre-wrap;word-break:break-word;background:#0e1422;border:1px solid var(--line);
   border-radius:8px;padding:10px;margin:8px 0 0;font-size:11.5px;max-height:340px;overflow:auto}
</style></head><body><div class="wrap">
<h1>API hit test</h1>
<p class="sub">${esc(meta.url)} &middot; ${esc(meta.when)}${meta.label ? ' &middot; ' + esc(meta.label) : ''}
 &middot; cases: ${esc(meta.cases)}</p>
${waveBlocks}
</div></body></html>`;
}

async function main() {
  const { counts, opts } = parseArgs(process.argv.slice(2));
  const cases = loadCases(opts.cases);
  const when = new Date().toISOString().replace('T', ' ').slice(0, 19);

  console.log(`endpoint  ${opts.url}`);
  console.log(`cases     ${opts.cases ? `${cases.length} from ${path.basename(opts.cases)}` : '1 built-in default pair'}`);
  console.log(`waves     ${counts.join(', ')} simultaneous hits`);

  const runs = [];
  for (const count of counts) {
    console.log(`\n=== firing ${count} requests at once ===`);
    const t0 = performance.now();
    // Launched in one tick: every request starts before any of them finishes.
    const hits = await Promise.all(
      Array.from({ length: count }, (_, i) =>
        oneHit(opts.url, cases[i % cases.length], i + 1, opts.timeout, t0)
      )
    );
    const wallMs = performance.now() - t0;
    hits.sort((a, b) => a.hit - b.hit);
    const summary = summarise(count, hits, wallMs);
    runs.push({ summary, hits });

    if (!opts.quiet) {
      for (const h of hits) {
        const verdict = h.ok
          ? `score ${h.score === null || h.score === undefined ? '—' : Number(h.score).toFixed(4)}${h.is_match ? ' MATCH' : ''}`
          : h.error || `HTTP ${h.status}`;
        console.log(`  #${String(h.hit).padStart(3)}  ${String(h.status).padStart(3)}  ${String(h.response_ms).padStart(6)} ms  ${verdict}`);
      }
    }
    console.log(
      `  served ${summary.served}/${count}  shed ${summary.shed_503}  timeouts ${summary.timed_out}  ` +
        `errors ${summary.other_errors}\n  fastest ${summary.min_ms} ms  p50 ${summary.p50_ms} ms  ` +
        `p95 ${summary.p95_ms} ms  slowest ${summary.max_ms} ms  wall ${(wallMs / 1000).toFixed(1)}s`
    );
    if (counts.length > 1 && count !== counts[counts.length - 1]) {
      await new Promise((r) => setTimeout(r, 4000)); // let the queue drain between waves
    }
  }

  const prefix = opts.out || path.join('reports', `hit-${counts.join('-')}`);
  fs.mkdirSync(path.dirname(prefix), { recursive: true });
  const meta = {
    url: opts.url,
    when,
    label: opts.label,
    cases: opts.cases ? `${cases.length} from ${path.basename(opts.cases)}` : 'built-in default pair',
  };
  fs.writeFileSync(`${prefix}.json`, JSON.stringify({ meta, runs }, null, 2));
  fs.writeFileSync(`${prefix}.html`, renderHtml(runs, meta));
  console.log(`\njson  -> ${prefix}.json`);
  console.log(`html  -> ${prefix}.html`);

  const anyBad = runs.some((r) => r.summary.served !== r.summary.concurrent_hits);
  process.exit(anyBad ? 1 : 0);
}

main().catch((e) => {
  console.error(`error: ${e.message}`);
  process.exit(2);
});
