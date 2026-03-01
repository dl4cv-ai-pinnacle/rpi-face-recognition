# Implementation Summary — Core Pipeline (Stages 1–4)

## Overview

The Butler face recognition system has been implemented as a modular Python pipeline running on Raspberry Pi 5. The system captures live video, detects faces, extracts embeddings, matches against a local database, and supports face enrollment — all running on-device with no cloud dependencies.

## Implemented Pipeline

| Stage | Component | Implementation |
|-------|-----------|----------------|
| 1. Frame Capture | Picamera2 wrapper | RGB888 at 640x480 (configurable), continuous video mode, context manager for resource safety |
| 2. Face Detection | SCRFD-500M (primary) / UltraFace-slim (fallback) | Swappable via `typing.Protocol` interface; SCRFD via insightface, UltraFace via raw ONNX Runtime; both return coordinates in original frame space |
| 3. Face Alignment | ArcFace 5-point alignment | Similarity transform from detected landmarks to standard ArcFace template → 112x112 normalized crop |
| 4. Embedding Extraction | MobileFaceNet (`w600k_mbf.onnx`) | 512-dim L2-normalized embeddings via ONNX Runtime; model ships with buffalo_sc pack |
| 5. Matching & Storage | FAISS (`IndexFlatIP`) + SQLite | Cosine similarity search on normalized vectors; SQLite for identity metadata; numpy fallback if FAISS unavailable |
| 6. Display & HUD | OpenCV window | Bounding boxes, landmarks, identity labels, confidence scores, FPS/resolution/timestamp, detector and embedder metadata |

## Subcommands

| Command | Description |
|---------|-------------|
| `uv run main.py run` | Live preview with detection and recognition (default) |
| `uv run main.py enroll --name "Name"` | Enrollment mode — captures one embedding per second, stores to database |

## Configuration

Django-style settings module (`butler/settings.py`) loaded from `.env` via `python-dotenv`. All pipeline components are configurable:

| Setting | Default | Description |
|---------|---------|-------------|
| `FACE_DETECTOR` | `scrfd` | `scrfd` or `ultraface` |
| `FACE_EMBEDDER` | `mobilefacenet` | Embedding model selection |
| `CAPTURE_RESOLUTION` | `640x480` | Camera capture resolution |
| `DETECTION_RESOLUTION` | `320x240` | Internal detector inference resolution |
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `RECOGNITION_THRESHOLD` | `0.4` | Cosine similarity threshold for identity matching |

## Project Structure

```
code/
├── main.py                          # Entry point (run / enroll subcommands)
├── pyproject.toml                   # Dependencies: opencv, picamera2, insightface, onnxruntime, faiss-cpu, python-dotenv
├── .env.example                     # Configuration template
├── butler/
│   ├── settings.py                  # Django-style settings from .env
│   ├── camera.py                    # Picamera2 wrapper
│   ├── pipeline.py                  # Pipeline orchestrator (run + enroll modes)
│   ├── display.py                   # OpenCV display with detection/recognition/enrollment overlays
│   ├── detectors/
│   │   ├── base.py                  # FaceDetector Protocol, Detection, Landmark
│   │   ├── scrfd.py                 # SCRFD-500M via insightface
│   │   └── ultraface.py            # UltraFace-slim via ONNX Runtime
│   └── recognition/
│       ├── base.py                  # FaceEmbedder Protocol, RecognizedFace
│       ├── align.py                 # ArcFace 5-point alignment (112x112 crop)
│       ├── mobilefacenet.py         # MobileFaceNet embedder via ONNX Runtime
│       └── database.py             # FAISS + SQLite face database
├── models/                          # Downloaded model files (gitignored)
└── data/                            # SQLite database (gitignored)
```

## Design Principles

- **Protocol-based modularity** — Face detectors and embedding extractors implement `typing.Protocol` interfaces, enabling independent swapping without inheritance coupling
- **Lazy initialization** — Models load on first use, not at import time; missing models degrade gracefully to display-only mode
- **Coordinate encapsulation** — Detectors operate at reduced resolution internally but return coordinates in the original frame's pixel space; callers never handle scaling
- **Settings hierarchy** — `.env` provides defaults, CLI arguments override them
