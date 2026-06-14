"""Single-mode face verification UI and the face-detection poll endpoint.

GET  /ui                 the verify page (upload/camera selfie sent as base64 + S3 reference URL)
POST /v1/faces/detect    face-present poll for the on-screen readiness ring
(Verification itself is POST /v1/faces/verify in face.py.)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse

from zepiris.deps import EmbeddingDep

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/faces/detect")
async def detect_face(
    embedding_svc: EmbeddingDep,
    file: UploadFile = File(...),
) -> dict:
    """Fast poll for the UI ring: is a face visible, and where?"""
    raw = await file.read()
    if not raw:
        return {"faceDetected": False, "bbox": [0, 0, 0, 0]}
    arr = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"faceDetected": False, "bbox": [0, 0, 0, 0]}
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        result = embedding_svc.detect_box(image_rgb)
    except Exception:
        return {"faceDetected": False, "bbox": [0, 0, 0, 0]}
    return {"faceDetected": bool(result.face_detected), "bbox": result.bbox}


@router.get("/ui", response_class=HTMLResponse)
async def ui_page() -> HTMLResponse:
    return HTMLResponse(_PAGE)


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ZepIris · Verify</title>
<style>
  :root{--bg:#0b0f17;--panel:#121826;--line:#1f2937;--muted:#8b97a7;
    --text:#e8edf4;--accent:#5b8cff;--ok:#1fbf75;--no:#ff5d6c;--warn:#e3b341;}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:radial-gradient(1200px 600px at 50% -10%,#16203a,var(--bg));color:var(--text);min-height:100vh}
  .wrap{max-width:820px;margin:0 auto;padding:28px 20px 64px}
  h1{font-size:22px;margin:0 0 2px}
  .sub{color:var(--muted);margin:0 0 20px;font-size:13px}
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
  .err{color:var(--no)} .good{color:var(--ok)}
  .verdict{font-size:30px;font-weight:800;margin:4px 0 2px;text-align:center}
  .verdict.match{color:var(--ok)} .verdict.nomatch{color:var(--no)}
  .bar{height:10px;border-radius:6px;background:#0e1422;overflow:hidden;margin:14px 0 4px}
  .bar>span{display:block;height:100%;width:0;background:linear-gradient(90deg,#5b8cff,#1fbf75);transition:width .4s}
  .pair{display:flex;gap:16px;justify-content:center;margin-top:18px;flex-wrap:wrap}
  .pair figure{margin:0;width:200px}
  .pair img{width:200px;height:200px;object-fit:cover;border-radius:12px;border:1px solid var(--line);background:#000}
  figcaption{color:var(--muted);font-size:12px;margin-top:6px;text-align:center}
  .center{text-align:center}
  details{margin-top:14px}
  summary{cursor:pointer;color:var(--muted);font-size:13px}
  pre{background:#0e1422;border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <h1>ZepIris · Verify</h1>
  <p class="sub">Capture a live selfie (sent as base64) and give a reference S3 URL (photo or ID document). Nothing is stored.</p>

  <div class="card">
    <label>Mode</label>
    <div class="row" style="margin:0">
      <button class="secondary" id="modeFace" style="flex:1">Face match</button>
      <button class="secondary" id="modeDoc" style="flex:1">Doc match (Aadhaar/PAN)</button>
    </div>
  </div>

  <div class="grid">
    <!-- SELFIE -->
    <div class="card">
      <label>Selfie (live capture, sent as base64)</label>
      <div class="drop" id="drop">
        <span id="dropText">Click to choose a photo<br/><small style="color:var(--muted)">(or drag &amp; drop)</small></span>
        <input type="file" id="fileInput" accept="image/*"/>
      </div>
      <button class="secondary" id="camBtn" style="margin-top:12px;width:100%">Use camera instead</button>
      <div id="camWrap" style="display:none;margin-top:12px">
        <video id="vid" autoplay playsinline muted style="width:100%;border-radius:12px;background:#000"></video>
        <button id="snap" style="margin-top:10px;width:100%">Capture from camera</button>
      </div>
      <img id="livePrev" class="thumb" alt="selfie preview"/>
    </div>

    <!-- REFERENCE -->
    <div class="card">
      <label id="refLabel">Reference face (S3 URL)</label>
      <input type="text" id="s3url" placeholder="https://bucket.s3.region.amazonaws.com/ref.jpg" autocomplete="off"/>
      <img id="refPrev" class="thumb" alt="reference preview"/>
    </div>
  </div>

  <div class="card">
    <div class="row">
      <button id="verifyBtn" disabled>Verify</button>
    </div>
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
let matchMode = 'face';  // 'face' -> facematch/verify, 'doc' -> docmatch/verify

function setMatchMode(m){
  matchMode = m;
  $('modeFace').style.background = m === 'face' ? 'var(--accent)' : '#222c40';
  $('modeDoc').style.background = m === 'doc' ? 'var(--accent)' : '#222c40';
  $('refLabel').textContent = m === 'doc' ? 'Reference document — Aadhaar/PAN (S3 URL)' : 'Reference face (S3 URL)';
  $('s3url').placeholder = m === 'doc'
    ? 'https://bucket.s3.region.amazonaws.com/aadhaar.jpg'
    : 'https://bucket.s3.region.amazonaws.com/ref.jpg';
}
$('modeFace').addEventListener('click', () => setMatchMode('face'));
$('modeDoc').addEventListener('click', () => setMatchMode('doc'));

function setLive(blob){
  liveBlob = blob;
  if(liveURL) URL.revokeObjectURL(liveURL);
  liveURL = URL.createObjectURL(blob);
  const p = $('livePrev'); p.src = liveURL; p.style.display = 'block';
  $('dropText').innerHTML = 'Photo selected — click to change';
  refreshBtn();
}
function hasSelfie(){ return !!liveBlob; }
function hasRef(){ return !!$('s3url').value.trim(); }
function refreshBtn(){ $('verifyBtn').disabled = !(hasSelfie() && hasRef()); }

// Read a Blob as a base64 data: URI (the API strips the "data:...;base64," prefix).
function blobToDataURL(blob){
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(blob);
  });
}

// --- file upload ---
$('drop').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', e => { if(e.target.files[0]) setLive(e.target.files[0]); });
$('drop').addEventListener('dragover', e => { e.preventDefault(); $('drop').style.borderColor = 'var(--accent)'; });
$('drop').addEventListener('dragleave', () => { $('drop').style.borderColor = ''; });
$('drop').addEventListener('drop', e => {
  e.preventDefault(); $('drop').style.borderColor = '';
  if(e.dataTransfer.files[0]) setLive(e.dataTransfer.files[0]);
});

// --- reference URL preview ---
$('s3url').addEventListener('input', () => {
  const u = $('s3url').value.trim();
  const p = $('refPrev');
  if(u){ p.src = u; p.style.display = 'block'; } else { p.style.display = 'none'; }
  refreshBtn();
});

// --- optional camera (works on https or localhost) ---
$('camBtn').addEventListener('click', async () => {
  if(stream){ return; }
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}, audio:false});
    $('vid').srcObject = stream;
    $('camWrap').style.display = 'block';
    $('camBtn').style.display = 'none';
  }catch(e){
    $('status').innerHTML = '<span class="err">Camera unavailable (needs HTTPS or localhost): '+e.message+'. Use file upload instead.</span>';
  }
});
$('snap').addEventListener('click', () => {
  const v = $('vid'), c = $('scratch');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(b => setLive(b), 'image/jpeg', 0.92);
});

// --- verify ---
$('verifyBtn').addEventListener('click', async () => {
  if(!hasSelfie() || !hasRef()){ return; }
  $('verifyBtn').disabled = true; $('status').textContent = 'Verifying…'; $('resultCard').style.display = 'none';
  const selfieDisplay = liveURL;
  const liveB64 = await blobToDataURL(liveBlob);
  const url = $('s3url').value.trim();
  const payload = {};
  if (matchMode === 'doc') {
    // doc mode: the S3 document is the probe (face extracted, no liveness gate);
    // the live capture stands in as the enrolled source-of-truth selfie.
    payload.doc_check_s3 = url;
    payload.source_selfie_b64 = liveB64;
  } else {
    // face mode: the live capture is the probe being verified (liveness runs on it);
    // the S3 image is the enrolled source-of-truth selfie.
    payload.face_check_b64 = liveB64;
    payload.source_selfie_s3 = url;
  }
  const endpoint = matchMode === 'doc' ? '/v1/faces/docmatch/verify' : '/v1/faces/facematch/verify';
  try{
    const r = await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const d = await r.json();
    if(!r.ok){
      $('status').innerHTML = '<span class="err">'+(d.detail?.message || d.detail || ('HTTP '+r.status))+'</span>';
      $('raw').textContent = JSON.stringify(d, null, 2);
      $('resultCard').style.display = 'block';
      $('verdict').textContent = '✕ Error'; $('verdict').className = 'verdict nomatch';
    } else {
      showResult(selfieDisplay, url, d);
    }
  }catch(e){ $('status').innerHTML = '<span class="err">Request failed: '+e.message+'</span>'; }
  $('verifyBtn').disabled = false;
});

function showResult(selfieUrl, url, d){
  $('status').textContent = ''; $('resultCard').style.display = 'block';
  $('liveImg').src = selfieUrl; $('refImg').src = url;
  $('raw').textContent = JSON.stringify(d, null, 2);
  const v = $('verdict');
  if(d.decodeFailed || d.livenessFailed === true || d.faceDetected === false){
    const reason = d.livenessFailed === true ? 'Liveness failed — looks like a photo/screen, not a live face'
      : d.faceDetected === false ? 'No face detected in the selfie'
      : 'Could not read the image';
    v.textContent = '✕ ' + reason; v.className = 'verdict nomatch';
    $('barFill').style.width = '0%'; $('scoreText').textContent = '';
    return;
  }
  const vr = d.verificationResult || {};
  const pct = Math.max(0, Math.min(100, Math.round((vr.score ?? 0) * 100)));
  v.textContent = vr.isMatch ? '✓ Match' : '✕ No match';
  v.className = 'verdict ' + (vr.isMatch ? 'match' : 'nomatch');
  $('barFill').style.width = pct + '%';
  let line = 'Similarity ' + pct + '%  (threshold ' + Math.round((vr.threshold ?? 0) * 100) + '%)';
  // Document-quality hint: a blurry card photo embeds poorly -> low/unreliable score.
  const doc = d.documentFace;
  if (doc && doc.sharpness != null) {
    const blurry = doc.sharpness < 25;   // crisp ID photo > 100; blurry captures 5-20
    line += '  •  doc sharpness ' + doc.sharpness.toFixed(0)
      + (blurry ? ' ⚠ low — retake the document photo (sharper, in focus, no glare)' : '');
    if (!doc.usedCrop) line += '  •  ⚠ could not isolate the face on the card';
  }
  $('scoreText').textContent = line;
}

setMatchMode('face');
</script>
</body>
</html>"""
