# rpi-face-recognition

Raspberry Pi 5 modular face recognition system with face memory.

Real-time face detection, alignment, embedding extraction, and matching — all running locally on-device.

## Pipeline

```
Picamera2 → SCRFD-500M → ArcFace Align → MobileFaceNet → FAISS Match → Display
(640x480)   (detection)   (112x112 crop)  (512-d embed)   (cosine sim)  (OpenCV)
```

## Requirements

- Raspberry Pi 5 with Raspberry Pi OS Bookworm
- Python 3.11+
- Picamera2-compatible camera module

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition.git
cd rpi-face-recognition

# 2. Create virtual environment
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 3. Install system packages
sudo apt install python3-picamera2 python3-opencv

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Download ONNX models (SCRFD-500M + MobileFaceNet)
bash scripts/download_models.sh
```

## Model Files

The pipeline uses ONNX models from the InsightFace `buffalo_sc` pack:

| Model | File | Description |
|-------|------|-------------|
| SCRFD-500MF | `models/det_500m.onnx` | Face detection (640x640 input) |
| MobileFaceNet | `models/w600k_mbf.onnx` | Face embedding (112x112 → 512-d) |

Models are downloaded automatically by `scripts/download_models.sh`.

## Usage

**Run from a NoMachine desktop terminal** (OpenCV window needs a display):

```bash
cd ~/rpi-face-recognition
source venv/bin/activate
python main.py                     # default config
python main.py --config my.yaml    # custom config
```

Press `q` to quit.

## Configuration

All settings are in `config.yaml`:

- **capture** — camera resolution and detection downscale size
- **detection** — model path, confidence/NMS thresholds
- **alignment** — output crop size (112)
- **embedding** — model path and dimension (512)
- **matching** — database paths, cosine similarity threshold
- **display** — window settings, colors, labels

## Project Structure

```
├── main.py                  # Entry point
├── config.yaml              # Configuration
├── requirements.txt         # Python dependencies
├── src/
│   ├── config.py            # Config loader
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── capture/             # Frame capture (Picamera2)
│   ├── detection/           # Face detection (SCRFD via ONNX Runtime)
│   ├── alignment/           # Face alignment (ArcFace template)
│   ├── embedding/           # Embedding extraction (MobileFaceNet via ONNX Runtime)
│   ├── matching/            # FAISS + SQLite face database
│   └── display/             # OpenCV renderer
├── models/                  # ONNX model files (gitignored)
├── data/                    # Face database (gitignored)
└── scripts/                 # Utility scripts
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
