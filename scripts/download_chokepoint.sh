#!/usr/bin/env bash
# Download and prepare the ChokePoint dataset for video benchmarks.
#
# ChokePoint requires registration at:
#   http://arma.sourceforge.net/chokepoint/
#
# After downloading, place the archive(s) in data/ and run this script
# to extract and organize them for benchmark_video.py.
#
# Expected result:
#   data/chokepoint/
#       frames/           (extracted frame images)
#       ground_truth/     (annotation files)
#       enrollment/       (reference images per subject for gallery enrollment)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/chokepoint"

mkdir -p "$DATA_DIR/frames"
mkdir -p "$DATA_DIR/ground_truth"
mkdir -p "$DATA_DIR/enrollment"

echo "=== ChokePoint Dataset Setup ==="
echo ""
echo "The ChokePoint dataset requires manual download after registration."
echo ""
echo "1. Register at: http://arma.sourceforge.net/chokepoint/"
echo "2. Download the dataset archives"
echo "3. Place archives in: $DATA_DIR/"
echo "4. Re-run this script to extract"
echo ""

# Check for archives to extract.
ARCHIVES=("$DATA_DIR"/*.zip "$DATA_DIR"/*.tar.gz "$DATA_DIR"/*.tar)
FOUND=0
for archive in "${ARCHIVES[@]}"; do
    [ -e "$archive" ] || continue
    FOUND=1
    echo "Extracting: $archive"
    case "$archive" in
        *.zip)     unzip -o -q "$archive" -d "$DATA_DIR/frames/" ;;
        *.tar.gz)  tar -xzf "$archive" -C "$DATA_DIR/frames/" ;;
        *.tar)     tar -xf "$archive" -C "$DATA_DIR/frames/" ;;
    esac
done

if [ "$FOUND" -eq 0 ]; then
    echo "No archives found in $DATA_DIR/"
    echo "Please download the dataset and place archives there."
    exit 1
fi

# Count what we have.
N_FRAMES=$(find "$DATA_DIR/frames" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
N_GT=$(find "$DATA_DIR/ground_truth" -name "*.txt" -o -name "*.xml" 2>/dev/null | wc -l)

echo ""
echo "=== Summary ==="
echo "Frames found: $N_FRAMES"
echo "GT files found: $N_GT"
echo "Data directory: $DATA_DIR"
echo ""
echo "To run video benchmarks:"
echo "  uv run --python 3.13 python scripts/benchmark_video.py \\"
echo "      --config config.yaml --chokepoint-dir data/chokepoint"
