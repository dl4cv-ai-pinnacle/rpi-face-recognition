#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sudo apt-get update
sudo apt-get install -y \
  curl \
  unzip \
  ripgrep \
  python3-onnx \
  python3-opencv \
  python3-onnxruntime \
  python3-scipy \
  python3-skimage \
  python3-picamera2 \
  libcamera-tools

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${PATH}"

echo "Syncing dev environment with uv..."
uv sync --project "${ROOT_DIR}" --group dev
echo "Installing git hooks (pre-commit)..."
uv run --project "${ROOT_DIR}" --group dev pre-commit install \
  --config "${ROOT_DIR}/.pre-commit-config.yaml"

echo "Bootstrap complete."
python3 - <<'PY'
import cv2
import onnx
import onnxruntime
print("cv2:", cv2.__version__)
print("onnx:", onnx.__version__)
print("onnxruntime:", onnxruntime.__version__)
PY
uv run --project "${ROOT_DIR}" --group dev ruff --version
uv run --project "${ROOT_DIR}" --group dev pyright --version
