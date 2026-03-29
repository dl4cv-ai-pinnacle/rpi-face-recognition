# Attributions — What Came from Where and Why

This document tracks the origin of every component in the unified pipeline. Three team members independently built face-recognition prototypes; this pipeline merges the best of each.

## Component Origins

### Detection

| Component | File | Source | Why this version |
|---|---|---|---|
| SCRFD detector (default) | `src/detection/insightface.py` | **Shalaiev** `butler/detectors/scrfd.py` | Battle-tested `insightface` library — maintained, handles model downloads, ~10 lines vs 212-line hand-rolled decoder |
| UltraFace fallback | `src/detection/ultraface.py` | **Shalaiev** `butler/detectors/ultraface.py` | Fast fallback (~7ms vs ~22ms). Hand-written NMS is correct and self-contained |
| Detector factory + graceful fallback | `src/detection/__init__.py` | **Shalaiev** `butler/detectors/__init__.py` pattern | Lazy-import factory dispatching on config. `create_detector_safe()` returns None on failure instead of crashing (graceful degradation) |

### Alignment

| Component | File | Source | Why this version |
|---|---|---|---|
| cv2 aligner (default) | `src/alignment.py` `Cv2Aligner` | **Shalaiev** `butler/recognition/align.py` | `estimateAffinePartial2D` with LMEDS — zero extra deps, robust estimator |
| skimage aligner | `src/alignment.py` `SkimageAligner` | **Valenia** `src/face_align.py` | `SimilarityTransform` — handles 128-aligned sizes, good for research flexibility |
| Center-crop fallback | `src/alignment.py` `center_crop_fallback` | **New** (designed from analysis) | Honest fallback for UltraFace (no landmarks). Center-crop + resize is more predictable than synthetic landmark estimation |

### Embedding

| Component | File | Source | Why this version |
|---|---|---|---|
| ArcFace/MobileFaceNet | `src/embedding/arcface.py` | **Valenia** `src/arcface.py` | Cleanest implementation — reads input size from ONNX graph, `blobFromImage` with `swapRB=True` handles BGR→RGB implicitly |

### Gallery & Matching

| Component | File | Source | Why this version |
|---|---|---|---|
| GalleryStore | `src/gallery.py` | **Valenia** `src/gallery.py` | Only implementation with full identity lifecycle: unknowns, promote, enrich, merge. Filesystem-backed, thread-safe with RLock |
| FAISS IndexFlatIP search | `src/gallery.py` `_rebuild_gallery_index` | **Avdieienko** `src/matching/database.py` | FAISS cosine search over mean templates — more scalable than np.dot loop |
| Index rebuild-on-write | `src/gallery.py` `_rebuild_gallery_index` | **Shalaiev** `butler/recognition/database.py` | Shalaiev proved the rebuild pattern works: tear down and recreate index on every write. Clean for small galleries |
| Quality-weighted mean templates | `src/gallery.py` `_template_from_samples` | **Valenia** `src/gallery.py` | Better samples (higher quality score) dominate the mean template |
| normalize_embedding | `src/gallery.py` | **Valenia** `src/gallery.py` | Was private `_normalize_embedding`. Collocated with its only consumer |

### Tracking

| Component | File | Source | Why this version |
|---|---|---|---|
| SimpleFaceTracker | `src/tracking.py` | **Valenia** `src/tracking.py` | IoU greedy matching + EMA smoothing on boxes and landmarks. Simplest correct approach for Pi |
| box_iou | `src/tracking.py` | **Valenia** `src/tracking.py` + `src/live_runtime.py` | Deduplicated — was duplicated across two files |

### Live Runtime

| Component | File | Source | Why this version |
|---|---|---|---|
| LiveRuntime | `src/live.py` | **Valenia** `src/live_runtime.py` | Strided detection, selective embedding refresh, grace period for known faces, enrichment logic |
| RLock + swap_pipeline() | `src/live.py` | **New** (designed for GUI hot-swap) | Thread-safe pipeline replacement. Build new pipeline before lock, swap reference, clear tracker state |

### Quality Scoring

| Component | File | Source | Why this version |
|---|---|---|---|
| FaceQuality + compute_face_quality | `src/quality.py` | **Valenia** `src/quality.py` | Geometric mean of confidence x area — gates enrichment quality |

### Metrics & System Stats

| Component | File | Source | Why this version |
|---|---|---|---|
| MemoryStats, CpuUsageSampler, SystemStats | `src/metrics.py` | **Valenia** `src/runtime_utils.py` | Pi-specific telemetry: CPU, SoC temp, GPU busy, RSS |
| LiveMetricsCollector | `src/metrics.py` | **Valenia** `src/live_runtime.py` | Rolling averages for ~20 metrics, periodic JSON persistence |

### Display

| Component | File | Source | Why this version |
|---|---|---|---|
| draw_hud, draw_detection_hud | `src/display.py` | **Shalaiev** `butler/display.py` | Richest HUD — FPS, resolution, timestamp, detector/embedder names |
| Double-pass text rendering | `src/display.py` `annotate_faces` | **Valenia** `src/live_runtime.py` `annotate_in_place` | Dark outline then colored foreground — readable on any background |
| Display class with GUI fallback | `src/display.py` `Display` | **Shalaiev** `butler/display.py` + **Avdieienko** `src/display/renderer.py` | Shalaiev's window management + Avdieienko's cv2.error catch when no GUI backend |

### Capture

| Component | File | Source | Why this version |
|---|---|---|---|
| PiCamera2Capture | `src/capture.py` | **Avdieienko** `src/capture/picamera2.py` + **Shalaiev** `butler/camera.py` | Avdieienko's error handling (try/except, returns None) + Shalaiev's context manager + video_configuration |

### Configuration

| Component | File | Source | Why this version |
|---|---|---|---|
| YAML → frozen dataclass tree | `src/config.py` | **Avdieienko** `src/config.py` | Typed dataclass hierarchy > env vars (Shalaiev) or argparse-only (Valenia). Frozen prevents mutation bugs |
| Config expanded to all stages | `src/config.py` | **New** (extended Avdieienko's pattern) | Added tracking, gallery, live, server, metrics, display sections |

### Contracts (Protocols)

| Component | File | Source | Why this version |
|---|---|---|---|
| Protocol definitions | `src/contracts.py` | **Valenia** `src/contracts.py` | Richest set — 9+ Protocols covering every DI boundary |
| @runtime_checkable | `src/contracts.py` | **Shalaiev** `butler/detectors/base.py` | Enables isinstance() checks for graceful degradation |
| FrameCapture Protocol | `src/contracts.py` | **Avdieienko** `src/capture/base.py` | Camera as a swappable Protocol |
| AlignerLike Protocol | `src/contracts.py` | **New** | Enables swappable alignment backends (cv2/skimage) |
| name property on Protocols | `src/contracts.py` | **Shalaiev** `butler/detectors/base.py` | Useful for display/logging which backend is active |

### Design Patterns

| Pattern | Source | Where applied |
|---|---|---|
| Protocol-based DI (no base classes) | **Valenia** | Every swappable stage |
| @runtime_checkable for graceful degradation | **Shalaiev** | All Protocols |
| Full DI wiring in one place | **Avdieienko** `main.py` | `server/app.py` `build_runtime()` |
| `Pipeline.step()` testable single-frame API | **Avdieienko** `src/pipeline.py` | `src/pipeline.py` — alias for `process_frame()` |
| Frozen config dataclasses | **Avdieienko** `src/config.py` | `src/config.py` |
| Lazy-import factory with config dispatch | **Shalaiev** `butler/detectors/__init__.py` | `src/detection/__init__.py`, `src/embedding/__init__.py` |

### Quantization

| Component | File | Source | Why this version |
|---|---|---|---|
| quantize_onnx_model + QuantizationReport | `src/quantization.py` | **Valenia** `src/runtime_utils.py` | Dynamic INT8 quantization. Measured accuracy: 94.72% vs 94.75% on LFW |

### ONNX Session Management

| Component | File | Source | Why this version |
|---|---|---|---|
| suppress_stderr_fd | `src/onnx_session.py` | **Valenia** `src/runtime_utils.py` | FD-level redirect to suppress ONNX Runtime C++ noise on ARM |

### Server

| Component | File | Source | Why this version |
|---|---|---|---|
| CameraStreamer | `server/streamer.py` | **Valenia** `scripts/live_camera_server.py` | Condition-variable frame publishing, MJPEG backpressure handling |
| HTTP handler + gallery pages | `server/handlers.py` | **Valenia** `scripts/live_camera_server.py` | Full enrollment UI, unknown inbox, identity detail, all CRUD routes |
| Live dashboard (HTML/CSS/JS) | `server/templates/index.html` | **Valenia** `scripts/live_camera_server.py` | Dark-mode responsive dashboard with auto-refreshing metrics |
| Server bootstrap | `server/app.py` | **Valenia** `scripts/live_camera_server.py` | Adapted to use unified config instead of argparse |

### Scripts

| Component | File | Source |
|---|---|---|
| Model download | `scripts/download_models.sh` | **Avdieienko** + **Valenia** merged |

### Tooling

| Component | File | Source |
|---|---|---|
| Pyright strict + Ruff + pre-commit | `pyproject.toml`, `.pre-commit-config.yaml` | **Valenia** `slop/valenia/pyproject.toml` |
| Test stubs and DI pattern | `tests/conftest.py` | **Valenia** `tests/conftest.py` pattern |
