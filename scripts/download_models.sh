#!/usr/bin/env bash
# Download ONNX model files for the face recognition pipeline.
# Models come from InsightFace buffalo_sc model pack.
#
# Origin: Avdieienko scripts/download_models.sh + Valenia scripts/download_models.sh (merged)
#
# Usage: bash scripts/download_models.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models/buffalo_sc"
ZIP_PATH="${ROOT_DIR}/models/buffalo_sc.zip"
URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"

mkdir -p "${MODELS_DIR}"

if [[ -f "${MODELS_DIR}/det_500m.onnx" && -f "${MODELS_DIR}/w600k_mbf.onnx" ]]; then
  echo "Models already present in ${MODELS_DIR}"
  exit 0
fi

echo "Downloading buffalo_sc model pack..."
curl -L "${URL}" -o "${ZIP_PATH}"
echo "Extracting to ${MODELS_DIR}..."
unzip -o "${ZIP_PATH}" -d "${MODELS_DIR}" >/dev/null
echo "Done:"
ls -lh "${MODELS_DIR}"/*.onnx 2>/dev/null || echo "WARNING: No .onnx files found"
