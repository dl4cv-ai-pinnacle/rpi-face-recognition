# Setup Guide

Step-by-step instructions to get the face recognition system running on a Raspberry Pi 5.

## Prerequisites

**Hardware:**
- Raspberry Pi 5 (4GB or 8GB RAM)
- Raspberry Pi Camera Module (v2, v3, or HQ Camera)
- microSD card (16GB+) with Raspberry Pi OS Bookworm (64-bit)
- Display connected (HDMI) or remote desktop access

**Software:**
- Raspberry Pi OS Bookworm (64-bit) — comes with Python 3.11+
- Internet connection for installing packages

**Remote access (pick one):**
- NoMachine — recommended for desktop GUI (OpenCV window)
- SSH — for terminal-only work (no display window)

## Step 1: Connect to your Raspberry Pi

**Option A: NoMachine (recommended)**

Install NoMachine on both your PC and the RPi. Connect to the RPi desktop. Open a terminal from the desktop.

**Option B: SSH**

```bash
ssh pi@<your-rpi-ip>
```

> Note: If running via SSH, the OpenCV display window will not appear. You can set `show_window: false` in `config.yaml` to suppress the window and run headless.

## Step 2: Clone the repository

```bash
cd ~
git clone https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition.git
cd rpi-face-recognition
git checkout feature/core-pipeline
```

## Step 3: Install system packages

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-pip python3-venv python3-numpy wget unzip
```

These packages provide:
- `python3-picamera2` — camera interface (requires libcamera backend)
- `python3-opencv` — OpenCV with GTK GUI support (needed for `cv2.imshow`)
- `wget`, `unzip` — for downloading models

## Step 4: Create a virtual environment

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

The `--system-site-packages` flag is important — it allows the venv to access `picamera2` and `opencv` which were installed via `apt`.

> Every time you open a new terminal, you need to activate the venv:
> ```bash
> cd ~/rpi-face-recognition
> source venv/bin/activate
> ```

## Step 5: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `numpy` — array operations
- `PyYAML` — config file parsing
- `scikit-image` — face alignment transform
- `faiss-cpu` — embedding similarity search
- `onnxruntime` — neural network inference (pre-built aarch64 wheel, no compilation needed)

Note: `opencv-python-headless` is listed in requirements.txt but will be satisfied by the system `python3-opencv` package. If pip tries to install the headless version, uninstall it so the system version (with GUI) takes priority:
```bash
pip uninstall -y opencv-python-headless
```

## Step 6: Download the models

```bash
bash scripts/download_models.sh
```

This downloads the InsightFace `buffalo_sc` model pack (~16MB) and extracts:
- `models/det_500m.onnx` — SCRFD-500MF face detection model
- `models/w600k_mbf.onnx` — MobileFaceNet face embedding model

Verify the models are in place:
```bash
ls -lh models/*.onnx
```

Expected output:
```
-rw-r--r-- 1 pi pi 2.2M ... models/det_500m.onnx
-rw-r--r-- 1 pi pi  13M ... models/w600k_mbf.onnx
```

## Step 7: Run the system

**From a NoMachine desktop terminal:**

```bash
cd ~/rpi-face-recognition
source venv/bin/activate
python main.py
```

A window titled "Face Recognition" will appear showing the camera feed with bounding boxes around detected faces.

Press **q** to quit.

**Custom config:**
```bash
python main.py --config my_custom_config.yaml
```

## Configuration

All settings are in `config.yaml`. Key parameters you might want to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capture.resolution` | `[640, 480]` | Camera resolution |
| `capture.detection_resolution` | `[320, 240]` | Downscale size for detection (lower = faster) |
| `detection.confidence_threshold` | `0.5` | Min score to count as a face (0.0–1.0) |
| `detection.nms_threshold` | `0.4` | NMS overlap threshold (lower = fewer duplicates) |
| `matching.threshold` | `0.4` | Min cosine similarity to match a known face (0.0–1.0) |
| `display.show_window` | `true` | Set to `false` for headless/SSH operation |
| `log_level` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |

## Troubleshooting

### Camera not detected

```bash
# Check if camera is connected
libcamera-hello --list
```

If no cameras are listed:
1. Make sure the ribbon cable is firmly seated
2. Enable the camera in `raspi-config` → Interface Options → Camera
3. Reboot: `sudo reboot`

### "No module named 'picamera2'"

The Picamera2 library must be installed via apt, not pip:
```bash
sudo apt install python3-picamera2
```

Make sure your venv was created with `--system-site-packages`.

### OpenCV window not appearing

This happens when running via SSH (no display). Solutions:
1. Use NoMachine or VNC for remote desktop access
2. Set `display.show_window: false` in `config.yaml` to run headless
3. If using NoMachine and still no window, make sure you opened the terminal from within the NoMachine desktop session

### "Rebuild the library with Windows, GTK+ 2.x or Cocoa support"

The pip-installed `opencv-python-headless` doesn't have GUI support. Fix:
```bash
pip uninstall -y opencv-python-headless
sudo apt install -y python3-opencv
```

### Models not found

```bash
# Re-download models
bash scripts/download_models.sh

# Verify
ls -la models/
```

### Out of memory during inference

Reduce the detection resolution in `config.yaml`:
```yaml
capture:
  detection_resolution: [160, 120]  # smaller = less memory
```

### Low FPS

- Lower `capture.detection_resolution` (e.g., `[160, 120]`)
- Increase `detection.confidence_threshold` to filter more aggressively
- Close other running applications on the RPi

## Project Structure

```
rpi-face-recognition/
├── main.py              # Entry point — run this
├── config.yaml          # All settings
├── requirements.txt     # Python dependencies
├── docs/
│   ├── architecture.md  # How the system works
│   └── setup.md         # This file
├── src/                 # Source code (modular pipeline)
├── models/              # ONNX model files (gitignored)
├── data/                # Face database files (gitignored)
└── scripts/
    └── download_models.sh
```

See [architecture.md](architecture.md) for details on how each pipeline stage works.
