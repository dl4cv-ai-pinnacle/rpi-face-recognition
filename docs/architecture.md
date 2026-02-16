# Architecture

## Overview

The system performs real-time face recognition on a Raspberry Pi 5. It captures video from a camera, detects faces, extracts embeddings, and matches them against a stored database — all running locally on-device with no cloud APIs.

## Pipeline

```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
│  Capture   │───>│ Detection  │───>│ Alignment  │───>│  Embedding   │───>│  Matching  │───>│ Display  │
│ Picamera2  │    │ SCRFD-500M │    │  ArcFace   │    │MobileFaceNet │    │FAISS+SQLite│    │  OpenCV  │
│  640x480   │    │  640x640   │    │  112x112   │    │   512-dim    │    │ cosine sim │    │  render  │
└───────────┘    └───────────┘    └───────────┘    └──────────────┘    └───────────┘    └──────────┘
```

Each stage is a separate module under `src/` with a protocol (interface), making components swappable.

## Pipeline Stages

### 1. Frame Capture (`src/capture/`)

**What it does:** Captures RGB frames from the Raspberry Pi camera module.

**Implementation:** `PiCamera2Capture` wraps the Picamera2 library, which uses the libcamera backend on RPi OS Bookworm.

**Config:**
- Resolution: 640x480 (full frame)
- Detection resolution: 320x240 (downscaled for faster detection)
- Format: RGB888

**Protocol:** Any class with `read() -> ndarray` and `release() -> None` can replace it.

### 2. Face Detection (`src/detection/`)

**What it does:** Locates faces in an image and returns bounding boxes with confidence scores.

**Model: SCRFD-500MF** (Sample and Computation Redistribution for Efficient Face Detection)
- Paper: Guo et al., "Sample and Computation Redistribution for Efficient Face Detection", ICLR 2022
- Source: [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- File: `models/det_500m.onnx` (from InsightFace `buffalo_sc` pack)
- Input: 640x640 RGB image (letterboxed)
- Output: 6 tensors — scores and bounding boxes at 3 FPN strides (8, 16, 32)
- Anchors: 2 per grid position
- Parameters: ~500K

**How it works:**
1. Input image is letterbox-resized to 640x640 and normalized: `(pixel - 127.5) / 128.0`
2. The model outputs are grouped by type: `[score_8, score_16, score_32, bbox_8, bbox_16, bbox_32]`
3. Each stride level produces a feature map (80x80, 40x40, 20x20) with 2 anchors per position
4. Bounding boxes are decoded as distances from anchor centers, scaled by the stride
5. Non-Maximum Suppression (NMS) removes duplicate detections

**Note:** The `det_500m.onnx` model does NOT output keypoint landmarks. When keypoints are unavailable, the system estimates 5-point landmarks from bounding box proportions (eyes at 30%/70% width, 35% height, etc.). For better alignment accuracy, use a model with keypoint outputs (e.g., `det_500m_kps` from `buffalo_s` pack).

**Inference:** ONNX Runtime with CPUExecutionProvider.

### 3. Face Alignment (`src/alignment/`)

**What it does:** Warps the detected face region into a standardized 112x112 crop for consistent embedding extraction.

**Method: ArcFace Similarity Transform**
- Uses 5 reference landmark positions from the ArcFace training template
- Fits a similarity transform (rotation + scale + translation) from detected landmarks to the reference
- Applies `cv2.warpAffine` to produce the aligned crop

**Reference landmarks (for 112x112 output):**
```
Left eye:     (38.29, 51.70)
Right eye:    (73.53, 51.50)
Nose tip:     (56.03, 71.74)
Left mouth:   (41.55, 92.37)
Right mouth:  (70.73, 92.20)
```

**Library:** scikit-image `SimilarityTransform` for transform estimation.

### 4. Embedding Extraction (`src/embedding/`)

**What it does:** Converts a 112x112 aligned face crop into a compact numerical vector (embedding) that represents the face's identity.

**Model: MobileFaceNet (w600k_mbf)**
- Paper: Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices", 2018
- Source: [InsightFace ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
- File: `models/w600k_mbf.onnx` (from InsightFace `buffalo_sc` pack, trained on WebFace600K)
- Input: 112x112 RGB image, normalized to [-1, 1]
- Output: 512-dimensional L2-normalized embedding vector
- Parameters: ~0.99M
- Accuracy: 99.55% on LFW benchmark

**How it works:**
1. Aligned face is normalized: `(pixel / 255 - 0.5) / 0.5` → range [-1, 1]
2. Converted from HWC to NCHW format
3. Model outputs a 512-dim vector
4. Vector is L2-normalized (unit length) so cosine similarity equals dot product

**Inference:** ONNX Runtime with CPUExecutionProvider.

### 5. Matching & Storage (`src/matching/`)

**What it does:** Searches a database of known face embeddings and returns the closest match.

**Components:**

**FAISS (Facebook AI Similarity Search)**
- Index type: `IndexFlatIP` (exact inner product search)
- Since embeddings are L2-normalized, inner product = cosine similarity
- Searches the entire database exhaustively (fast enough for <10,000 faces)
- Persisted to disk as `data/faces.index`

**SQLite**
- Stores metadata: face_id, name, created_at, updated_at
- Face IDs map 1:1 to FAISS index rows
- Persisted as `data/faces.db`

**Matching logic:**
1. Query embedding is L2-normalized
2. FAISS finds the nearest neighbor by cosine similarity
3. If similarity >= threshold (default 0.4), return the match with person's name
4. Otherwise, return "Unknown"

### 6. Display (`src/display/`)

**What it does:** Renders detection results on the video frame and displays it.

**Features:**
- Green bounding boxes around detected faces
- Name label (or "Unknown") above each box
- Confidence score next to recognized names
- Real-time FPS counter
- Graceful fallback if no GUI backend is available

**Library:** OpenCV (`cv2.imshow`, `cv2.rectangle`, `cv2.putText`).

## Configuration

All parameters are in `config.yaml` and loaded into typed Python dataclasses (`src/config.py`). This includes model paths, detection thresholds, camera resolution, display settings, and logging level. No hardcoded magic numbers in the pipeline code.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ONNX Runtime over MNN** | MNN has no pre-built aarch64 Python wheel; PyMNN build is broken on RPi 5. ONNX Runtime has pip-installable aarch64 wheels. |
| **Protocol over ABC** | Python `Protocol` provides structural subtyping — any class with the right methods works, no inheritance needed. |
| **Detection on 320x240** | Camera captures 640x480, but detection runs on a downscaled 320x240 image (then letterboxed to 640x640 for the model). Fewer pixels = faster inference. Coordinates are scaled back to full resolution. |
| **FAISS IndexFlatIP** | Exact search with L2-normalized vectors. Fast enough for <10K faces. No approximate indexing needed at this scale. |
| **Config-driven** | One YAML file controls everything. No CLI flags for thresholds — change `config.yaml` instead. |

## Tools & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| ONNX Runtime | >=1.17.0 | Neural network inference |
| OpenCV | >=4.8.0 | Image processing, display |
| NumPy | >=1.24.0 | Array operations |
| FAISS | >=1.7.4 | Embedding similarity search |
| scikit-image | >=0.21.0 | Similarity transform for alignment |
| PyYAML | >=6.0 | Config file parsing |
| Picamera2 | system | Camera capture (RPi OS) |
| SQLite3 | stdlib | Face metadata storage |

## File Structure

```
src/
├── config.py              # AppConfig dataclass + YAML loader
├── pipeline.py            # Pipeline class: step() and run()
├── capture/
│   ├── base.py            # FrameCapture protocol
│   └── picamera2.py       # Picamera2 implementation
├── detection/
│   ├── base.py            # Detection dataclass + FaceDetector protocol
│   └── scrfd.py           # SCRFD-500M with ONNX Runtime
├── alignment/
│   └── align.py           # ArcFace 5-point alignment
├── embedding/
│   ├── base.py            # EmbeddingExtractor protocol
│   └── mobilefacenet.py   # MobileFaceNet with ONNX Runtime
├── matching/
│   └── database.py        # FaceDatabase (FAISS + SQLite)
└── display/
    └── renderer.py        # OpenCV frame renderer
```

## References

1. Guo et al., "Sample and Computation Redistribution for Efficient Face Detection" (SCRFD), ICLR 2022
2. Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices", 2018
3. Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
4. InsightFace model zoo: https://github.com/deepinsight/insightface/tree/master/model_zoo
