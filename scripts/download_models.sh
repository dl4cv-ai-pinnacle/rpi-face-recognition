#!/usr/bin/env bash
# Download ONNX model files for the face recognition pipeline.
#
# - buffalo_sc pack (SCRFD + MobileFaceNet) from InsightFace
# - UltraFace-slim-320 from Ultra-Light-Fast-Generic-Face-Detector
#
# Usage: bash scripts/download_models.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"

mkdir -p "${MODELS_DIR}/buffalo_sc"

# -- buffalo_sc (insightface SCRFD + MobileFaceNet) --
BUFFALO_DIR="${MODELS_DIR}/buffalo_sc"
if [[ -f "${BUFFALO_DIR}/det_500m.onnx" && -f "${BUFFALO_DIR}/w600k_mbf.onnx" ]]; then
  echo "[OK] buffalo_sc models already present"
else
  BUFFALO_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
  BUFFALO_ZIP="${MODELS_DIR}/buffalo_sc.zip"
  echo "[DOWNLOAD] Fetching buffalo_sc model pack..."
  curl -L "${BUFFALO_URL}" -o "${BUFFALO_ZIP}"
  echo "[EXTRACT] Unpacking..."
  unzip -o "${BUFFALO_ZIP}" -d "${BUFFALO_DIR}" >/dev/null
  rm -f "${BUFFALO_ZIP}"
  echo "[OK] buffalo_sc extracted"
fi

# -- UltraFace-slim-320 (fast fallback detector) --
ULTRAFACE_PATH="${MODELS_DIR}/version-slim-320.onnx"
if [[ -f "${ULTRAFACE_PATH}" ]]; then
  echo "[OK] UltraFace model already present"
else
  ULTRAFACE_URL="https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-slim-320.onnx"
  echo "[DOWNLOAD] Fetching UltraFace-slim-320..."
  curl -L "${ULTRAFACE_URL}" -o "${ULTRAFACE_PATH}"
  echo "[OK] UltraFace downloaded"
fi

echo ""
echo "=== Model files ==="
ls -lh "${BUFFALO_DIR}"/*.onnx 2>/dev/null || true
ls -lh "${ULTRAFACE_PATH}" 2>/dev/null || true
echo "=== Done ==="
