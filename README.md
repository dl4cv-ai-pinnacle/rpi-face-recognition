# rpi-face-recognition

Raspberry Pi 5 modular face recognition system with face memory.

Real-time face detection, alignment, embedding extraction, and matching — all running locally on-device.

## Pipeline

```
Picamera2 → SCRFD-500M → ArcFace Align → MobileFaceNet → FAISS Match → Display
(640x480)   (detection)   (112x112 crop)  (128-d embed)   (cosine sim)  (OpenCV)
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

# 5. Build and install MNN (no aarch64 wheel available)
sudo apt install cmake build-essential
git clone https://github.com/alibaba/MNN.git /tmp/MNN && cd /tmp/MNN
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMNN_ARM82=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_PYTHON=ON
make -j4
cd ../pymnn/pip_package && python3 setup.py install
cd -

# 6. Prepare models (convert from ONNX to MNN format)
bash scripts/download_models.sh
```

## Model Preparation

The pipeline requires two MNN models in the `models/` directory:

| Model | File | Source |
|-------|------|--------|
| SCRFD-500M | `models/scrfd_500m.mnn` | [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) |
| MobileFaceNet | `models/mobilefacenet.mnn` | [InsightFace ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) |

Convert ONNX models to MNN:
```bash
MNNConvert -f ONNX --modelFile scrfd_500m.onnx --MNNModel models/scrfd_500m.mnn --bizCode biz
MNNConvert -f ONNX --modelFile mobilefacenet.onnx --MNNModel models/mobilefacenet.mnn --bizCode biz
```

## Usage

```bash
python main.py                     # default config
python main.py --config my.yaml    # custom config
```

Press `q` to quit.

## Configuration

All settings are in `config.yaml`:

- **capture** — camera resolution and detection downscale size
- **detection** — model path, confidence/NMS thresholds
- **alignment** — output crop size (112)
- **embedding** — model path and dimension (128)
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
│   ├── detection/           # Face detection (SCRFD)
│   ├── alignment/           # Face alignment (ArcFace template)
│   ├── embedding/           # Embedding extraction (MobileFaceNet)
│   ├── matching/            # FAISS + SQLite face database
│   └── display/             # OpenCV renderer
├── models/                  # MNN model files (gitignored)
├── data/                    # Face database (gitignored)
└── scripts/                 # Utility scripts
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
