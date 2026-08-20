#!/usr/bin/env bash
# One-time local setup for ZepIris on macOS (Apple Silicon friendly).
# Creates the project venv on Python 3.12, installs base+dev deps via Poetry,
# then installs the ML stack (torch/insightface/onnx) from PyPI — the pinned
# pytorch-cpu source has no arm64-mac wheels, so we bypass it here.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Locate Poetry (official installer puts it in ~/.local/bin).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found. Install it with:"
  echo "  curl -sSL https://install.python-poetry.org | python3 -"
  exit 1
fi

# Pick a Python 3.12 interpreter (pyenv preferred).
PY312=""
if command -v pyenv >/dev/null 2>&1 && pyenv prefix 3.12 >/dev/null 2>&1; then
  PY312="$(pyenv prefix 3.12)/bin/python3.12"
elif command -v python3.12 >/dev/null 2>&1; then
  PY312="$(command -v python3.12)"
else
  echo "Python 3.12 not found. Install it (e.g. 'pyenv install 3.12')." >&2
  exit 1
fi
echo "Using Python: $PY312"

poetry env use "$PY312"
echo "Installing base + dev dependencies…"
poetry install --with dev

echo "Installing ML stack from PyPI (torch/insightface/onnx)…"
"$ROOT/.venv/bin/pip" install \
  "torch>=2.2.0" "torchvision>=0.17.0" \
  "insightface>=0.7.3,<0.8.0" "onnx>=1.14.0,<1.22.0" "onnxruntime>=1.17.0"

echo
echo "✓ Setup complete."
echo "  Start everything:  ./scripts/run_all.sh"
echo "  Seed the gallery:  .venv/bin/python scripts/seed_gallery.py"
echo "  API docs:          http://localhost:8000/docs"
