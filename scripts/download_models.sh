#!/usr/bin/env bash
#
# Download pre-converted MNN model files for the face recognition pipeline.
# Usage: bash scripts/download_models.sh
#
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

# TODO: Replace these URLs with actual hosted model locations.
# Models need to be converted from ONNX to MNN format using:
#   MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode biz
#
# Sources:
#   SCRFD-500M ONNX:     https://github.com/deepinsight/insightface/tree/master/detection/scrfd
#   MobileFaceNet ONNX:  https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

SCRFD_URL="${SCRFD_URL:-}"
MOBILEFACENET_URL="${MOBILEFACENET_URL:-}"

download_if_missing() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    if [ -f "$dest" ]; then
        echo "[OK] $name already exists, skipping"
        return
    fi

    if [ -z "$url" ]; then
        echo "[SKIP] $name — no URL set. Convert from ONNX manually:"
        echo "       MNNConvert -f ONNX --modelFile <onnx_file> --MNNModel $dest --bizCode biz"
        return
    fi

    echo "[DOWNLOAD] $name from $url"
    wget -q --show-progress -O "$dest" "$url"
    echo "[OK] $name downloaded"
}

echo "=== Downloading models to $MODELS_DIR ==="
download_if_missing "$SCRFD_URL" "$MODELS_DIR/scrfd_500m.mnn"
download_if_missing "$MOBILEFACENET_URL" "$MODELS_DIR/mobilefacenet.mnn"
echo "=== Done ==="
