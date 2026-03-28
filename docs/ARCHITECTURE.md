# Architecture

## Pipeline Flow

```
Camera Frame (640x480 BGR)
    |
    v
Detection (insightface SCRFD | UltraFace)
    |
    v
Tracking (IoU greedy matching + EMA smoothing)
    |
    v
Alignment (cv2 | skimage | center-crop fallback)
    |
    v
Embedding (MobileFaceNet ArcFace, 512-dim)
    |
    v
Matching (FAISS IndexFlatIP over mean templates)
    |
    v
Gallery (enroll / match / capture unknown / promote / enrich)
```

## Swappable Backends

Backends are selected via `config.yaml` and can be changed at runtime through the GUI settings panel.

| Stage | Default | Alternative | Config key |
|---|---|---|---|
| Detection | insightface SCRFD | UltraFace-slim-320 | `detection.backend` |
| Alignment | cv2 (LMEDS) | skimage (SimilarityTransform) | `alignment.method` |
| Embedding | MobileFaceNet FP32 | MobileFaceNet INT8 | `embedding.quantize_int8` |

## Key Design Decisions

1. **Protocol-based DI** — every swappable stage is a `@runtime_checkable Protocol`. No base classes, no inheritance. Concrete implementations are structurally typed.

2. **Frozen YAML config** — all parameters in `config.yaml`, parsed into frozen dataclasses. No module-level settings mutation.

3. **FAISS with mean templates** — each identity has one template (quality-weighted mean of samples). FAISS index rebuilt on every gallery write.

4. **Thread-safe hot-swap** — `LiveRuntime.swap_pipeline()` acquires an RLock, swaps the pipeline reference, clears tracker state. HTTP handler builds new pipeline before acquiring lock.

5. **BGR convention** — OpenCV-native throughout. ONNX models handle RGB via `blobFromImage(swapRB=True)`.

## Module Map

```
src/
  contracts.py      Protocols + type aliases + shared dataclasses
  config.py         YAML -> frozen dataclass tree
  onnx_session.py   ONNX Runtime noise suppression
  detection/        Detector backends (insightface, ultraface)
  embedding/        Embedder backends (arcface)
  alignment.py      Face alignment (cv2, skimage, center-crop)
  pipeline.py       FacePipeline + factory
  gallery.py        GalleryStore + FAISS + unknowns workflow
  tracking.py       SimpleFaceTracker + box_iou
  quality.py        Face quality scoring
  metrics.py        System telemetry + rolling metrics
  display.py        HUD + face annotations
  capture.py        Picamera2 wrapper
  live.py           LiveRuntime orchestrator
  quantization.py   INT8 ONNX quantization

server/
  app.py            HTTP server entry point
  handlers.py       Request handlers + HTML rendering
  streamer.py       MJPEG camera streamer
  templates/        Static HTML pages
```

## What Is Intentionally Avoided

- No plugin registry or auto-discovery — backends are explicit in factory functions
- No ORM or migration framework — gallery uses filesystem + FAISS
- No async/await — stdlib threading is sufficient for Pi's 4 cores
- No template engine — f-string HTML rendering, no Jinja dependency
- No config file hot-reloading — GUI settings panel rebuilds components explicitly
