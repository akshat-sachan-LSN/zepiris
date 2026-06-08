"""Single-mode face verification UI and the face-detection poll endpoint.

GET  /ui                 the verify page (S3 reference URL + live camera)
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
  .wrap{max-width:760px;margin:0 auto;padding:28px 20px 64px}
  h1{font-size:22px;margin:0 0 2px}
  .sub{color:var(--muted);margin:0 0 20px;font-size:13px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:18px}
  label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:8px}
  input[type=text]{width:100%;padding:11px 12px;border-radius:10px;border:1px solid var(--line);
    background:#0e1422;color:var(--text);font-size:15px}
  .videoBox{position:relative;border-radius:12px;overflow:hidden;background:#000;aspect-ratio:4/3;margin-top:6px}
  video,canvas{width:100%;display:block}
  .ring{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none}
  .ring::before{content:"";width:60%;aspect-ratio:1;border-radius:50%;
    border:5px solid var(--warn);box-shadow:0 0 18px rgba(0,0,0,.45) inset;transition:border-color .2s, box-shadow .2s}
  .ring.ok::before{border-color:var(--ok);box-shadow:0 0 22px rgba(31,191,117,.55)}
  .ringHint{position:absolute;left:0;right:0;bottom:10px;text-align:center;font-size:13px;font-weight:600;
    color:var(--warn);text-shadow:0 1px 3px rgba(0,0,0,.8)}
  .ringHint.ok{color:var(--ok)}
  .row{display:flex;gap:12px;margin-top:14px;flex-wrap:wrap}
  button{flex:1;min-width:130px;padding:13px 16px;border-radius:11px;border:0;cursor:pointer;
    font-size:15px;font-weight:600;color:#fff;background:var(--accent)}
  button:disabled{opacity:.45;cursor:not-allowed}
  .hide{display:none!important}
  .note{color:var(--muted);font-size:13px;margin:12px 0 0}
  .err{color:var(--no)} .good{color:var(--ok)}
  .verdict{font-size:28px;font-weight:800;margin:4px 0 2px;text-align:center}
  .verdict.match{color:var(--ok)} .verdict.nomatch{color:var(--no)}
  .bar{height:10px;border-radius:6px;background:#0e1422;overflow:hidden;margin:12px 0 4px}
  .bar>span{display:block;height:100%;background:linear-gradient(90deg,#5b8cff,#1fbf75)}
  .pair{display:flex;gap:16px;justify-content:center;margin-top:16px;flex-wrap:wrap}
  .pair figure{margin:0;width:190px}
  .pair img{width:190px;height:190px;object-fit:cover;border-radius:12px;border:1px solid var(--line);background:#000}
  figcaption{color:var(--muted);font-size:12px;margin-top:6px;text-align:center}
  .refPrev{width:120px;height:120px;object-fit:cover;border-radius:12px;border:1px solid var(--line);background:#000;margin-top:10px;display:none}
  .center{text-align:center}
</style>
</head>
<body>
<div class="wrap">
  <h1>ZepIris · Verify</h1>
  <p class="sub">Compare a live photo against a reference image from an S3 link. Nothing is stored.</p>

  <div class="card">
    <label for="s3url">Reference image — S3 URL</label>
    <input type="text" id="s3url" placeholder="https://your-bucket.s3.amazonaws.com/ref.jpg?..." autocomplete="off"/>
    <img id="refPrev" class="refPrev" alt="reference preview"/>
  </div>

  <div class="card">
    <label>Live photo</label>
    <div class="videoBox">
      <video id="vMatch" autoplay playsinline muted></video>
      <div class="ring" id="ringMatch"></div>
      <div class="ringHint" id="hintMatch">Position your face in the circle…</div>
    </div>
    <div class="row">
      <button id="matchCapture" disabled>Capture &amp; Verify</button>
    </div>
    <p id="matchStatus" class="note"></p>
  </div>

  <div class="card center hide" id="resultCard">
    <div class="verdict" id="verdict"></div>
    <div class="bar"><span id="barFill" style="width:0%"></span></div>
    <div class="note" id="scoreText"></div>
    <div class="pair">
      <figure><img id="selfieImg" alt="live"/><figcaption>Live photo</figcaption></figure>
      <figure><img id="refImg" alt="reference"/><figcaption>Reference (S3)</figcaption></figure>
    </div>
  </div>
</div>

<canvas id="scratch" class="hide"></canvas>
<script>
const $ = (id) => document.getElementById(id);
let stream = null, selfieURL = null, faceOk = false, detectInFlight = false;
const dcanvas = document.createElement('canvas');

$('s3url').addEventListener('input', () => {
  const u = $('s3url').value.trim();
  const p = $('refPrev');
  if(u){ p.src = u; p.style.display='block'; } else { p.style.display='none'; }
});

async function initCamera(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}, audio:false});
    $('vMatch').srcObject = stream;
  }catch(e){
    $('matchStatus').innerHTML = '<span class="err">Camera unavailable: '+e.message+
      '. Open http://localhost:8000/ui (camera needs localhost or HTTPS).</span>';
    $('matchCapture').disabled = true;
  }
}
function grab(video){
  const c = $('scratch'), w = video.videoWidth, h = video.videoHeight;
  c.width = w; c.height = h; c.getContext('2d').drawImage(video,0,0,w,h);
  return new Promise(res => c.toBlob(b=>res(b),'image/jpeg',0.92));
}
function grabSmall(video, targetW){
  const w = video.videoWidth, h = video.videoHeight;
  if(!w){ return null; }
  dcanvas.width = targetW; dcanvas.height = Math.round(targetW * h / w);
  dcanvas.getContext('2d').drawImage(video, 0, 0, dcanvas.width, dcanvas.height);
  return new Promise(res => dcanvas.toBlob(b=>res(b),'image/jpeg',0.8));
}
function faceInCircle(b){
  if(!b || b.length < 4) return false;
  const cx=(b[0]+b[2])/2, cy=(b[1]+b[3])/2, fw=b[2]-b[0], fh=b[3]-b[1];
  if(fw<=0||fh<=0) return false;
  const ar=3/4, R=0.30;
  const centered = Math.pow(cx-0.5,2)+Math.pow((cy-0.5)*ar,2) <= R*R;
  const sized = Math.max(fw,fh)>=0.18 && fw<=0.95;
  return centered && sized;
}
function ringUI(ok, msg){
  $('ringMatch').classList.toggle('ok', ok);
  const h=$('hintMatch'); h.classList.toggle('ok', ok); h.textContent=msg;
}
async function pollDetect(){
  if(detectInFlight || !stream) return;
  const blob = await grabSmall($('vMatch'), 320);
  if(!blob) return;
  detectInFlight = true;
  try{
    const fd = new FormData(); fd.append('file', blob, 'f.jpg');
    const d = await (await fetch('/v1/faces/detect',{method:'POST',body:fd})).json();
    const detected = !!d.faceDetected;
    const inCircle = detected && faceInCircle(d.bbox);
    faceOk = inCircle;
    ringUI(inCircle, inCircle ? '✓ Face in position — ready'
      : detected ? 'Move your face into the circle' : 'Position your face in the circle…');
    $('matchCapture').disabled = !faceOk;
  }catch(e){}
  finally{ detectInFlight = false; }
}

$('matchCapture').addEventListener('click', async () => {
  const url = $('s3url').value.trim();
  if(!url){ $('matchStatus').innerHTML='<span class="err">Enter the reference S3 URL first.</span>'; return; }
  $('matchCapture').disabled = true; $('matchStatus').textContent='Verifying…'; $('resultCard').classList.add('hide');
  const blob = await grab($('vMatch'));
  if(selfieURL) URL.revokeObjectURL(selfieURL);
  selfieURL = URL.createObjectURL(blob);
  const fd = new FormData();
  fd.append('s3_url', url); fd.append('file', blob, 'live.jpg');
  try{
    const r = await fetch('/v1/faces/verify',{method:'POST',body:fd});
    const d = await r.json();
    if(!r.ok){ $('matchStatus').innerHTML='<span class="err">'+(d.detail?.message||d.detail||('Error '+r.status))+'</span>'; }
    else showResult(url, d);
  }catch(e){ $('matchStatus').innerHTML='<span class="err">Request failed: '+e.message+'</span>'; }
  $('matchCapture').disabled = false;
});

function showResult(url, d){
  $('matchStatus').textContent=''; $('resultCard').classList.remove('hide');
  $('selfieImg').src = selfieURL; $('refImg').src = url;
  const v = $('verdict');
  if(d.decodeFailed || d.iqaPassed===false || d.faceDetected===false){
    const reason = d.livenessFailed===true ? 'Liveness failed — show your real face, not a photo or screen'
      : d.faceDetected===false ? 'No face detected — try again'
      : d.iqaPassed===false ? 'Image quality check failed' : 'Could not read image';
    v.textContent='✕ '+reason; v.className='verdict nomatch';
    $('barFill').style.width='0%'; $('scoreText').textContent=''; return;
  }
  const vr = d.verificationResult || {};
  const pct = Math.max(0, Math.min(100, Math.round((vr.score ?? 0)*100)));
  v.textContent = vr.isMatch ? '✓ Match' : '✕ No match';
  v.className = 'verdict '+(vr.isMatch?'match':'nomatch');
  $('barFill').style.width = pct+'%';
  $('scoreText').textContent = 'Similarity '+pct+'%  (threshold '+Math.round((vr.threshold??0)*100)+'%)';
}

initCamera();
setInterval(pollDetect, 800);
</script>
</body>
</html>"""
