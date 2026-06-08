#!/usr/bin/env bash
# Start the ZepIris API (:8000) pointed at the Docker infra (MinIO + Milvus) and
# the local ML service. No pyenv/poetry shell needed — uses the project venv.
#
# Every value can be overridden by exporting it first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x "$ROOT/.venv/bin/zepiris-api" ]; then
  echo "ERROR: $ROOT/.venv not found. Run scripts/setup.sh first." >&2
  exit 1
fi

# Free port 8000 (only the listening process).
PID="$(lsof -nP -iTCP:8000 -sTCP:LISTEN -t 2>/dev/null || true)"
if [ -n "$PID" ]; then
  echo "Stopping existing API on :8000 (pid $PID)…"
  kill "$PID" 2>/dev/null || true
  sleep 1
fi

export ZEPIRIS_ML_INFERENCE_SERVICE_URL="${ZEPIRIS_ML_INFERENCE_SERVICE_URL:-http://localhost:8001}"
export ZEPIRIS_MINIO_ENDPOINT="${ZEPIRIS_MINIO_ENDPOINT:-localhost:9002}"
export ZEPIRIS_MINIO_ACCESS_KEY="${ZEPIRIS_MINIO_ACCESS_KEY:-minioadmin}"
export ZEPIRIS_MINIO_SECRET_KEY="${ZEPIRIS_MINIO_SECRET_KEY:-minioadmin}"
export ZEPIRIS_MINIO_BUCKET="${ZEPIRIS_MINIO_BUCKET:-zepiris}"
export ZEPIRIS_MINIO_SECURE="${ZEPIRIS_MINIO_SECURE:-false}"
export ZEPIRIS_MILVUS_HOST="${ZEPIRIS_MILVUS_HOST:-localhost}"
export ZEPIRIS_MILVUS_PORT="${ZEPIRIS_MILVUS_PORT:-19530}"
export ZEPIRIS_MILVUS_COLLECTION="${ZEPIRIS_MILVUS_COLLECTION:-zepiris_faces}"
export ZEPIRIS_MILVUS_EMBEDDING_DIM="${ZEPIRIS_MILVUS_EMBEDDING_DIM:-512}"

echo "Starting API on http://localhost:8000  (UI: http://localhost:8000/ui)"
exec "$ROOT/.venv/bin/zepiris-api"
