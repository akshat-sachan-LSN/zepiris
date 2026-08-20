#!/usr/bin/env bash
# One command to run the whole ZepIris stack locally:
#   1. Docker infra (MinIO + etcd + Milvus)
#   2. ML inference service (:8001) in the background  -> logs/ml.log
#   3. API (:8000) in the foreground  (Ctrl+C to stop everything)
#
# Backend only — no UI is served. Check health at http://localhost:8000/healthz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

echo "▶ [1/3] Starting infra containers (minio, etcd, milvus)…"
docker-compose up -d minio etcd milvus

echo "▶ Waiting for Milvus to report healthy…"
until [ "$(docker inspect -f '{{.State.Health.Status}}' zepiris-milvus 2>/dev/null)" = "healthy" ]; do
  sleep 3
done
echo "  infra ready."

echo "▶ [2/3] Starting ML inference service (:8001) → logs/ml.log"
nohup "$ROOT/scripts/run_ml.sh" >logs/ml.log 2>&1 &
echo $! >logs/ml.pid

cleanup() {
  echo
  echo "Stopping ML service…"
  kill "$(cat logs/ml.pid 2>/dev/null)" 2>/dev/null || true
}
trap cleanup EXIT

echo "▶ Waiting for ML service health…"
until curl -sf http://localhost:8001/healthz >/dev/null 2>&1; do sleep 2; done
echo "  ML ready (logs/ml.log)."

echo "▶ [3/3] Starting API (:8000).  Docs → http://localhost:8000/docs   (Ctrl+C to stop)"
"$ROOT/scripts/run_api.sh"
