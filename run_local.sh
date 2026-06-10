#!/usr/bin/env bash
#
# run_local.sh — start BOTH ZepIris services locally with Poetry (no Docker).
#
#   1. ml-inference  (PyTorch models)  ->  http://127.0.0.1:8001
#   2. api           (FastAPI + /ui)   ->  http://127.0.0.1:8000
#
# The ML service starts first; the API waits until it is healthy, then starts.
# Press Ctrl+C once to stop BOTH services cleanly.
#
# Usage:
#   ./run_local.sh
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

ML_PORT="${ML_PORT:-8001}"
API_PORT="${API_PORT:-8000}"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

# --- Point model paths at the local ./models dir (defaults are container paths) ---
export ML_SERVICE_HOST="0.0.0.0"
export ML_SERVICE_PORT="$ML_PORT"
export ML_SERVICE_ML_DEVICE="cpu"
export ML_SERVICE_NSFW_LOCAL_MODEL_PATH="$DIR/models/nsfw_model.pth"
export ML_SERVICE_SPOOF_LOCAL_MODEL_PATH="$DIR/models/spoof_model.pth"
export ML_SERVICE_SPOOF_ONNX_MODEL_PATH="$DIR/models/minifasnet_v2_yakhyo.onnx"
export ML_SERVICE_SPOOF_ONNX_MODEL_PATH_2="$DIR/models/minifasnet_v1se_yakhyo.onnx"
export ML_SERVICE_BLUR_LOCAL_MODEL_PATH="$DIR/models/blur_model.pth"

# --- API points at the local ML service ---
export ZEPIRIS_ML_INFERENCE_SERVICE_URL="http://127.0.0.1:${ML_PORT}"
export ZEPIRIS_VERIFY_THRESHOLD="${ZEPIRIS_VERIFY_THRESHOLD:-0.5}"

ML_PID=""
API_PID=""

cleanup() {
  echo ""
  echo "Stopping services..."
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
  [ -n "$ML_PID" ] && kill "$ML_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "Stopped."
}
trap cleanup EXIT INT TERM

echo "==> Starting ml-inference on :${ML_PORT} (logs: $LOG_DIR/ml.log)"
poetry run zepiris-ml-inference-api >"$LOG_DIR/ml.log" 2>&1 &
ML_PID=$!

echo "==> Waiting for ml-inference to become healthy..."
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${ML_PORT}/healthz" >/dev/null 2>&1; then
    echo "    ml-inference is up."
    break
  fi
  if ! kill -0 "$ML_PID" 2>/dev/null; then
    echo "ERROR: ml-inference exited during startup. Last log lines:"
    tail -n 30 "$LOG_DIR/ml.log"
    exit 1
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    echo "ERROR: ml-inference did not become healthy within ~120s. Last log lines:"
    tail -n 30 "$LOG_DIR/ml.log"
    exit 1
  fi
done

echo "==> Starting api on :${API_PORT} (logs: $LOG_DIR/api.log)"
poetry run python -m uvicorn zepiris.main:app --host 0.0.0.0 --port "$API_PORT" >"$LOG_DIR/api.log" 2>&1 &
API_PID=$!

echo "==> Waiting for api to become healthy..."
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${API_PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "ERROR: api exited during startup. Last log lines:"
    tail -n 30 "$LOG_DIR/api.log"
    exit 1
  fi
  sleep 1
done

echo ""
echo "================================================================"
echo "  ZepIris is running:"
echo "    UI:   http://127.0.0.1:${API_PORT}/ui"
echo "    API:  http://127.0.0.1:${API_PORT}/v1/faces/verify"
echo "    ML:   http://127.0.0.1:${ML_PORT}/healthz"
echo ""
echo "  Logs:  tail -f $LOG_DIR/api.log $LOG_DIR/ml.log"
echo "  Stop:  press Ctrl+C"
echo "================================================================"

# Stay in the foreground; if either service dies, exit (trap cleans up the other).
# (Portable to macOS bash 3.2, which lacks `wait -n`.)
while kill -0 "$ML_PID" 2>/dev/null && kill -0 "$API_PID" 2>/dev/null; do
  sleep 2
done
echo "A service exited; shutting down the other."
