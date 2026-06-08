#!/usr/bin/env bash
# Start the ZepIris ML inference service (:8001) with correct LOCAL model paths
# and demo-friendly (permissive) IQA thresholds. No pyenv/poetry shell needed —
# uses the project venv directly.
#
# Override any threshold by exporting it before running, e.g.
#   ML_SERVICE_SPOOF_THRESHOLD=0.5 scripts/run_ml.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x "$ROOT/.venv/bin/zepiris-ml-inference-api" ]; then
  echo "ERROR: $ROOT/.venv not found. Run scripts/setup.sh first." >&2
  exit 1
fi

# Free port 8001 (only the listening process) so we don't hit "address in use".
PID="$(lsof -nP -iTCP:8001 -sTCP:LISTEN -t 2>/dev/null || true)"
if [ -n "$PID" ]; then
  echo "Stopping existing ML service on :8001 (pid $PID)…"
  kill "$PID" 2>/dev/null || true
  sleep 1
fi

# Local weights live in ./models (NOT the Docker path /app/models).
export ML_SERVICE_NSFW_LOCAL_MODEL_PATH="$ROOT/models/nsfw_model.pth"
export ML_SERVICE_SPOOF_LOCAL_MODEL_PATH="$ROOT/models/spoof_model.pth"
export ML_SERVICE_BLUR_LOCAL_MODEL_PATH="$ROOT/models/blur_model.pth"

# Bundled IQA models are poorly calibrated; permissive defaults keep the demo
# usable. Tighten these for production.
export ML_SERVICE_NSFW_THRESHOLD="${ML_SERVICE_NSFW_THRESHOLD:-1.0}"
export ML_SERVICE_SPOOF_THRESHOLD="${ML_SERVICE_SPOOF_THRESHOLD:-0.0}"
export ML_SERVICE_BLUR_THRESHOLD="${ML_SERVICE_BLUR_THRESHOLD:-1.0}"
export ML_SERVICE_ML_DEVICE="${ML_SERVICE_ML_DEVICE:-cpu}"

echo "Starting ML inference service on http://localhost:8001 …"
exec "$ROOT/.venv/bin/zepiris-ml-inference-api"
