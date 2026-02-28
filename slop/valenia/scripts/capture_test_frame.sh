#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT_DIR}/data/test_frame.jpg}"
mkdir -p "$(dirname "$OUT")"

rpicam-still \
  --timeout 1200 \
  --width 640 \
  --height 480 \
  --nopreview \
  --output "$OUT"

echo "Saved: $OUT"
