# rpi-face-recognition

Modular face recognition pipeline for Raspberry Pi 5. Swappable backends (detection, alignment, matching), live MJPEG dashboard, gallery with unknown-capture workflow, FAISS matching.

## Quick Start (Raspberry Pi)

```bash
# 1. Clone and enter
git clone https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition.git
cd rpi-face-recognition

# 2. Install system deps (Pi only)
sudo apt install -y python3-picamera2 python3-opencv

# 3. Install Python deps
uv sync --python 3.13

# 4. Download models
bash scripts/download_models.sh

# 5. Install insightface (default detector)
uv pip install insightface

# 6. Start the live server
uv run --python 3.13 python -m server.app
```

Open `http://<pi-ip>:8080/` in a browser to see the live dashboard.

## Development (macOS/Linux)

```bash
# Install deps (no camera — Pi only)
uv sync --python 3.13

# Run tests
uv run --python 3.13 pytest

# Type check
uv run --python 3.13 pyright src/ server/

# Lint
uv run --python 3.13 ruff check src/ server/ tests/
```

## Configuration

All settings in `config.yaml`. Key options:

```yaml
detection:
  backend: "insightface"    # or "ultraface" for 3x faster, lower accuracy

alignment:
  method: "cv2"             # or "skimage" for research flexibility

embedding:
  quantize_int8: false      # set true for ~3% faster inference
```

## Architecture

```
Camera → Detection → Tracking → Alignment → Embedding → FAISS Matching → Gallery
```

- **Detection:** insightface SCRFD (default, accurate) or UltraFace (fast fallback)
- **Alignment:** cv2 LMEDS (default) or skimage SimilarityTransform
- **Embedding:** MobileFaceNet ArcFace, 512-dim, optional INT8
- **Matching:** FAISS IndexFlatIP over quality-weighted mean templates
- **Gallery:** Filesystem-backed. Enroll, match, auto-capture unknowns, promote, enrich

See `docs/ARCHITECTURE.md` for details. See `docs/ATTRIBUTIONS.md` for component origins.

## Project Structure

```
src/                    Pipeline modules (Protocols, config, detection, embedding, etc.)
server/                 Live HTTP dashboard (MJPEG stream, gallery UI, metrics)
scripts/                Model download, benchmarks
tests/                  Behavioral tests with DI stubs
docs/                   Architecture, attributions, benchmark findings
config.yaml             Default configuration
slop/                   Archived individual prototypes (read-only)
```

## Team

- **Valenia** — gallery, live runtime, tracking, quality, metrics, benchmarks, tests
- **Shalaiev** — insightface SCRFD, UltraFace, cv2 alignment, display, FAISS+SQLite, graceful degradation
- **Avdieienko** — YAML config, clean architecture, DI wiring, model download, documentation
