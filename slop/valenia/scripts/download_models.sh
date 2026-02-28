#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"
ZIP_PATH="${MODELS_DIR}/buffalo_sc.zip"
TARGET_DIR="${MODELS_DIR}/buffalo_sc"
URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"

mkdir -p "${MODELS_DIR}"

if [[ -f "${TARGET_DIR}/det_500m.onnx" && -f "${TARGET_DIR}/w600k_mbf.onnx" ]]; then
  echo "Models already present in ${TARGET_DIR}"
  exit 0
fi

echo "Downloading buffalo_sc model pack..."
curl -L "${URL}" -o "${ZIP_PATH}"
echo "Extracting to ${TARGET_DIR}..."
unzip -o "${ZIP_PATH}" -d "${TARGET_DIR}" >/dev/null
echo "Done:"
ls -lh "${TARGET_DIR}"
