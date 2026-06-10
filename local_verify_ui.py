#!/usr/bin/env python3
"""Local face-verify UI that talks to a REMOTE ZepIris API (e.g. the EC2 box).

Why this exists: a browser page opened locally cannot call the EC2 API directly
(CORS blocks cross-origin fetch, and the API sends no CORS headers). This script
serves the UI from localhost AND proxies the verify request to the remote API, so
the browser only ever talks to localhost — no CORS, and the camera works too
(localhost is a "secure origin").

Usage:
    python3 local_verify_ui.py
    # then open http://localhost:8090/

Point it at a different backend / port:
    ZEPIRIS_API=http://65.0.4.214:8000 PORT=8090 python3 local_verify_ui.py

No dependencies — Python 3 standard library only.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND = os.environ.get("ZEPIRIS_API", "http://65.0.4.214:8000").rstrip("/")
PORT = int(os.environ.get("PORT", "8090"))

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ZepIris · Verify (local)</title>
<style>
  :root{--bg:#0b0f17;--panel:#121826;--line:#1f2937;--muted:#8b97a7;
    --text:#e8edf4;--accent:#5b8cff;--ok:#1fbf75;--no:#ff5d6c;}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:radial-gradient(1200px 600px at 50% -10%,#16203a,var(--bg));color:var(--text);min-height:100vh}
  .wrap{max-width:820px;margin:0 auto;padding:28px 20px 64px}
  h1{font-size:22px;margin:0 0 2px}
  .sub{color:var(--muted);margin:0 0 20px;font-size:13px}
  .be{color:var(--accent)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:680px){.grid{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:18px}
  label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:8px}
  input[type=text]{width:100%;padding:11px 12px;border-radius:10px;border:1px solid var(--line);
    background:#0e1422;color:var(--text);font-size:14px}
  .drop{border:1.5px dashed var(--line);border-radius:12px;background:#0e1422;padding:18px;text-align:center;
    cursor:pointer;transition:border-color .15s}
  .drop:hover{border-color:var(--accent)}
  .drop input{display:none}
  .thumb{width:100%;aspect-ratio:1;object-fit:cover;border-radius:12px;border:1px solid var(--line);
    background:#000;margin-top:12px;display:none}
  .row{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap}
  button{flex:1;min-width:150px;padding:13px 16px;border-radius:11px;border:0;cursor:pointer;
    font-size:15px;font-weight:600;color:#fff;background:var(--accent)}
  button.secondary{background:#222c40}
  button:disabled{opacity:.45;cursor:not-allowed}
  .note{color:var(--muted);font-size:13px;margin:12px 0 0}
  .err{color:var(--no)}
  .verdict{font-size:30px;font-weight:800;margin:4px 0 2px;text-align:center}
  .verdict.match{color:var(--ok)} .verdict.nomatch{color:var(--no)}
  .bar{height:10px;border-radius:6px;background:#0e1422;overflow:hidden;margin:14px 0 4px}
  .bar>span{display:block;height:100%;width:0;background:linear-gradient(90deg,#5b8cff,#1fbf75);transition:width .4s}
  .pair{display:flex;gap:16px;justify-content:center;margin-top:18px;flex-wrap:wrap}
  .pair figure{margin:0;width:200px}
  .pair img{width:200px;height:200px;object-fit:cover;border-radius:12px;border:1px solid var(--line);background:#000}
  figcaption{color:var(--muted);font-size:12px;margin-top:6px;text-align:center}
  .center{text-align:center}
  details{margin-top:14px} summary{cursor:pointer;color:var(--muted);font-size:13px}
  pre{background:#0e1422;border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <h1>ZepIris · Verify <small style="font-size:12px;color:var(--muted)">(local)</small></h1>
  <p class="sub">Backend: <span class="be">__BACKEND__</span></p>

  <div class="card">
    <label>Mode</label>
    <div class="row" style="margin:0">
      <button class="secondary" id="modeFace" style="flex:1">Face match</button>
      <button class="secondary" id="modeDoc" style="flex:1">Doc match (Aadhaar/PAN)</button>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <label>Selfie</label>
      <div class="row" style="margin:0 0 12px">
        <button class="secondary" id="selUpload" style="flex:1">Upload / camera</button>
        <button class="secondary" id="selUrl" style="flex:1">S3 URL</button>
      </div>
      <div id="selLocalWrap">
        <div class="drop" id="drop">
          <span id="dropText">Click to choose a photo<br/><small style="color:var(--muted)">(or drag &amp; drop)</small></span>
          <input type="file" id="fileInput" accept="image/*"/>
        </div>
        <button class="secondary" id="camBtn" style="margin-top:12px;width:100%">Use camera instead</button>
        <div id="camWrap" style="display:none;margin-top:12px">
          <video id="vid" autoplay playsinline muted style="width:100%;border-radius:12px;background:#000"></video>
          <button id="snap" style="margin-top:10px;width:100%">Capture from camera</button>
        </div>
      </div>
      <input type="text" id="selfieUrl" placeholder="https://bucket.s3.region.amazonaws.com/selfie.jpg" autocomplete="off" style="display:none"/>
      <img id="livePrev" class="thumb" alt="selfie preview"/>
    </div>
    <div class="card">
      <label id="refLabel">Reference face (photo)</label>
      <div class="row" style="margin:0 0 12px">
        <button class="secondary" id="modeUrl" style="flex:1">S3 URL</button>
        <button class="secondary" id="modeFile" style="flex:1">Upload doc</button>
      </div>
      <input type="text" id="s3url" placeholder="https://bucket.s3.region.amazonaws.com/ref.jpg" autocomplete="off"/>
      <div class="drop" id="refDrop" style="display:none">
        <span id="refDropText">Click to choose a document/photo<br/><small style="color:var(--muted)">(Aadhaar / PAN / any face image)</small></span>
        <input type="file" id="refFileInput" accept="image/*"/>
      </div>
      <img id="refPrev" class="thumb" alt="reference preview"/>
    </div>
  </div>

  <div class="card">
    <div class="row"><button id="verifyBtn" disabled>Verify</button></div>
    <p id="status" class="note"></p>
  </div>

  <div class="card center" id="resultCard" style="display:none">
    <div class="verdict" id="verdict"></div>
    <div class="bar"><span id="barFill"></span></div>
    <div class="note" id="scoreText"></div>
    <div class="pair">
      <figure><img id="liveImg" alt="live"/><figcaption>Live photo</figcaption></figure>
      <figure><img id="refImg" alt="reference"/><figcaption>Reference (S3)</figcaption></figure>
    </div>
    <details><summary>Raw response</summary><pre id="raw"></pre></details>
  </div>
</div>

<canvas id="scratch" style="display:none"></canvas>
<script>
const $ = (id) => document.getElementById(id);
let liveBlob = null, liveURL = null, stream = null;
let refMode = 'url', refBlob = null, refURL = null;
let matchMode = 'face';  // 'face' -> facematch/verify, 'doc' -> docmatch/verify

function setMatchMode(m){
  matchMode = m;
  $('modeFace').style.background = m === 'face' ? 'var(--accent)' : '#222c40';
  $('modeDoc').style.background = m === 'doc' ? 'var(--accent)' : '#222c40';
  $('refLabel').textContent = m === 'doc' ? 'Reference document (Aadhaar/PAN)' : 'Reference face (photo)';
  $('refDropText').innerHTML = m === 'doc'
    ? 'Click to choose the ID document<br/><small style="color:var(--muted)">(Aadhaar / PAN scan)</small>'
    : 'Click to choose a reference photo<br/><small style="color:var(--muted)">(any face image)</small>';
}
$('modeFace').addEventListener('click', () => setMatchMode('face'));
$('modeDoc').addEventListener('click', () => setMatchMode('doc'));

let selfieMode = 'local';  // 'local' (upload/camera) or 'url' (S3 link)
function setSelfieMode(m){
  selfieMode = m;
  $('selLocalWrap').style.display = m === 'local' ? 'block' : 'none';
  $('selfieUrl').style.display = m === 'url' ? 'block' : 'none';
  $('selUpload').style.background = m === 'local' ? 'var(--accent)' : '#222c40';
  $('selUrl').style.background = m === 'url' ? 'var(--accent)' : '#222c40';
  $('livePrev').style.display = 'none';
  if(m === 'url' && $('selfieUrl').value.trim()){ $('livePrev').src = $('selfieUrl').value.trim(); $('livePrev').style.display='block'; }
  if(m === 'local' && liveURL){ $('livePrev').src = liveURL; $('livePrev').style.display='block'; }
  refreshBtn();
}
$('selUpload').addEventListener('click', () => setSelfieMode('local'));
$('selUrl').addEventListener('click', () => setSelfieMode('url'));
$('selfieUrl').addEventListener('input', () => {
  const u = $('selfieUrl').value.trim();
  if(u){ $('livePrev').src = u; $('livePrev').style.display='block'; } else { $('livePrev').style.display='none'; }
  refreshBtn();
});

function setLive(blob){
  liveBlob = blob;
  if(liveURL) URL.revokeObjectURL(liveURL);
  liveURL = URL.createObjectURL(blob);
  const p = $('livePrev'); p.src = liveURL; p.style.display = 'block';
  $('dropText').innerHTML = 'Photo selected — click to change';
  refreshBtn();
}
function hasSelfie(){ return selfieMode === 'url' ? !!$('selfieUrl').value.trim() : !!liveBlob; }
function hasRef(){ return refMode === 'url' ? !!$('s3url').value.trim() : !!refBlob; }
function refreshBtn(){ $('verifyBtn').disabled = !(hasSelfie() && hasRef()); }

function setRefMode(m){
  refMode = m;
  $('s3url').style.display = m === 'url' ? 'block' : 'none';
  $('refDrop').style.display = m === 'file' ? 'block' : 'none';
  $('modeUrl').style.background = m === 'url' ? 'var(--accent)' : '#222c40';
  $('modeFile').style.background = m === 'file' ? 'var(--accent)' : '#222c40';
  $('refPrev').style.display = 'none';
  if(m === 'url' && $('s3url').value.trim()){ $('refPrev').src = $('s3url').value.trim(); $('refPrev').style.display='block'; }
  if(m === 'file' && refURL){ $('refPrev').src = refURL; $('refPrev').style.display='block'; }
  refreshBtn();
}
function setRefFile(blob){
  refBlob = blob;
  if(refURL) URL.revokeObjectURL(refURL);
  refURL = URL.createObjectURL(blob);
  $('refPrev').src = refURL; $('refPrev').style.display = 'block';
  $('refDropText').innerHTML = 'Document selected — click to change';
  refreshBtn();
}
$('modeUrl').addEventListener('click', () => setRefMode('url'));
$('modeFile').addEventListener('click', () => setRefMode('file'));
$('refDrop').addEventListener('click', () => $('refFileInput').click());
$('refFileInput').addEventListener('change', e => { if(e.target.files[0]) setRefFile(e.target.files[0]); });
setRefMode('url');
setMatchMode('face');
setSelfieMode('local');

$('drop').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', e => { if(e.target.files[0]) setLive(e.target.files[0]); });
$('drop').addEventListener('dragover', e => { e.preventDefault(); $('drop').style.borderColor='var(--accent)'; });
$('drop').addEventListener('dragleave', () => { $('drop').style.borderColor=''; });
$('drop').addEventListener('drop', e => { e.preventDefault(); $('drop').style.borderColor='';
  if(e.dataTransfer.files[0]) setLive(e.dataTransfer.files[0]); });

$('s3url').addEventListener('input', () => {
  const u = $('s3url').value.trim(); const p = $('refPrev');
  if(u){ p.src = u; p.style.display='block'; } else { p.style.display='none'; }
  refreshBtn();
});

$('camBtn').addEventListener('click', async () => {
  if(stream) return;
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}, audio:false});
    $('vid').srcObject = stream; $('camWrap').style.display='block'; $('camBtn').style.display='none';
  }catch(e){ $('status').innerHTML = '<span class="err">Camera unavailable: '+e.message+'. Use file upload.</span>'; }
});
$('snap').addEventListener('click', () => {
  const v=$('vid'), c=$('scratch'); c.width=v.videoWidth; c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0); c.toBlob(b=>setLive(b),'image/jpeg',0.92);
});

$('verifyBtn').addEventListener('click', async () => {
  if(!hasSelfie() || !hasRef()) return;
  $('verifyBtn').disabled = true; $('status').textContent='Verifying…'; $('resultCard').style.display='none';
  const fd = new FormData();
  // selfie: file upload OR S3 URL (S3 selfie => server skips liveness, pure image match)
  let selfieDisplay;
  if(selfieMode === 'url'){ selfieDisplay = $('selfieUrl').value.trim(); fd.append('selfie_s3_url', selfieDisplay); }
  else { selfieDisplay = liveURL; fd.append('file', liveBlob, 'live.jpg'); }
  // reference / document: file upload OR S3 URL
  const fileField = matchMode === 'doc' ? 'document_file' : 'reference_file';
  const endpoint = matchMode === 'doc' ? '/v1/faces/docmatch/verify' : '/v1/faces/facematch/verify';
  let refDisplay;
  if(refMode === 'url'){ refDisplay = $('s3url').value.trim(); fd.append('s3_url', refDisplay); }
  else { refDisplay = refURL; fd.append(fileField, refBlob, 'reference.jpg'); }
  try{
    const r = await fetch(endpoint, {method:'POST', body:fd});  // proxied to backend
    const d = await r.json();
    if(!r.ok){
      $('status').innerHTML = '<span class="err">'+(d.detail?.message || d.detail || ('HTTP '+r.status))+'</span>';
      $('raw').textContent = JSON.stringify(d,null,2); $('resultCard').style.display='block';
      $('verdict').textContent='✕ Error'; $('verdict').className='verdict nomatch';
    } else { showResult(selfieDisplay, refDisplay, d); }
  }catch(e){ $('status').innerHTML = '<span class="err">Request failed: '+e.message+'</span>'; }
  $('verifyBtn').disabled = false;
});

function showResult(selfieUrl, url, d){
  $('status').textContent=''; $('resultCard').style.display='block';
  $('liveImg').src = selfieUrl; $('refImg').src = url; $('raw').textContent = JSON.stringify(d,null,2);
  const v = $('verdict');
  // iqaPassed===false alone is NOT a failure (it's expected for S3->S3 no-liveness matches).
  if(d.decodeFailed || d.livenessFailed===true || d.faceDetected===false){
    const reason = d.livenessFailed===true ? 'Liveness failed — looks like a photo/screen'
      : d.faceDetected===false ? 'No face detected in the selfie'
      : 'Could not read the image';
    v.textContent='✕ '+reason; v.className='verdict nomatch';
    $('barFill').style.width='0%'; $('scoreText').textContent=''; return;
  }
  const vr = d.verificationResult || {};
  const pct = Math.max(0, Math.min(100, Math.round((vr.score ?? 0)*100)));
  v.textContent = vr.isMatch ? '✓ Match' : '✕ No match';
  v.className = 'verdict '+(vr.isMatch?'match':'nomatch');
  $('barFill').style.width = pct+'%';
  $('scoreText').textContent = 'Similarity '+pct+'%  (threshold '+Math.round((vr.threshold ?? 0)*100)+'%)';
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        print("  " + (fmt % args))

    def do_GET(self):
        if self.path in ("/", "/ui", "/index.html"):
            body = PAGE.replace("__BACKEND__", BACKEND).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/v1/faces/"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        # Forward the multipart body verbatim to the remote backend (same path).
        req = urllib.request.Request(
            f"{BACKEND}{self.path}",
            data=body,
            method="POST",
            headers={"Content-Type": self.headers.get("Content-Type", "")},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                status = resp.status
                ctype = resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            data = e.read()
            status = e.code
            ctype = e.headers.get("Content-Type", "application/json")
        except Exception as e:  # connection error to backend
            data = f'{{"detail":"proxy_error: {e}"}}'.encode("utf-8")
            status = 502
            ctype = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"ZepIris local verify UI  ->  proxying to {BACKEND}")
    print(f"Open:  http://localhost:{PORT}/")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
