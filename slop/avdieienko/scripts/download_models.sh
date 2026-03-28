#!/usr/bin/env bash
#
# Download ONNX model files for the face recognition pipeline.
# Models come from InsightFace buffalo_sc model pack.
#
# Usage: bash scripts/download_models.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
TMP_DIR="/tmp/insightface_models"

mkdir -p "$MODELS_DIR" "$TMP_DIR"

BUFFALO_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
BUFFALO_ZIP="$TMP_DIR/buffalo_sc.zip"

# Download buffalo_sc pack if needed
if [ ! -f "$MODELS_DIR/det_500m.onnx" ] || [ ! -f "$MODELS_DIR/w600k_mbf.onnx" ]; then
    echo "[DOWNLOAD] Fetching buffalo_sc model pack..."
    wget -q --show-progress -O "$BUFFALO_ZIP" "$BUFFALO_URL"

    echo "[EXTRACT] Unpacking models..."
    unzip -o "$BUFFALO_ZIP" -d "$TMP_DIR/buffalo_sc"

    # Copy the models we need
    find "$TMP_DIR/buffalo_sc" -name "det_500m.onnx" -exec cp {} "$MODELS_DIR/det_500m.onnx" \;
    find "$TMP_DIR/buffalo_sc" -name "w600k_mbf.onnx" -exec cp {} "$MODELS_DIR/w600k_mbf.onnx" \;

    echo "[OK] Models extracted to $MODELS_DIR"
else
    echo "[OK] Models already exist, skipping download"
fi

# Verify
echo ""
echo "=== Model files ==="
ls -lh "$MODELS_DIR"/*.onnx 2>/dev/null || echo "WARNING: No .onnx files found!"
echo "=== Done ==="
