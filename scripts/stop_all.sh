#!/usr/bin/env bash
# Stop the local ZepIris stack: API (:8000), ML service (:8001), and infra containers.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for port in 8000 8001; do
  PID="$(lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -n "$PID" ]; then
    echo "Stopping service on :$port (pid $PID)…"
    kill "$PID" 2>/dev/null || true
  fi
done

echo "Stopping infra containers (keeps data volumes)…"
docker-compose stop minio etcd milvus 2>/dev/null || true
echo "Done."
