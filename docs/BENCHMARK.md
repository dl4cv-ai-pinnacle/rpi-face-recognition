# Benchmark Report

**Date:** 2026-04-01
**Hardware:** Raspberry Pi 5 (4 GiB RAM)
**Branch:** `feat/#10-video-benchmarks`
**Issue:** #10

## Dataset

**FANVID** — celebrity video clips with per-frame bounding box and identity annotations from HuggingFace.

- Resolution: 180x320 (LR variant) — matches Pi camera scenarios
- Clips used: 50 (from 488 available)
- Total frames per run: ~8,660
- Ground-truth subjects per run: 10 unique identities
- Gallery enrollment: 29 from mugshots + 3 auto-enrolled from clip frames = 32 total

## Methodology

Each benchmark run:

1. Builds a fresh `FacePipeline` from config
2. Creates an **isolated temp gallery** (`tempfile.mkdtemp`) — no cross-run contamination
3. Enrolls identities from mugshot images, then auto-enrolls remaining identities from clip frames
4. Processes each clip with a fresh `LiveRuntime` (tracker state reset between clips)
5. Collects per-frame timing and tracking records
6. Computes temporal metrics (IDSW, TCR, fragmentation, TTFI, UFR) and throughput metrics

Latency is reported separately for **detection frames** (where SCRFD + embedding run) and **tracking-only frames** (where the tracker holds/predicts boxes). With `det_every=3`, only 1/3 of frames incur detection cost.

### Metrics glossary

| Metric | Description |
|--------|-------------|
| **FPS** | Sustained frames per second (wall time / total frames) |
| **Det p50/p95** | Detection-frame latency percentiles in ms |
| **IDSW** | ID switches — times a track's assigned identity changes |
| **TCR** | Temporal consistency ratio: `1 - IDSW / tracked_frames` |
| **Frag** | Fragmentation — times a GT trajectory is interrupted |
| **TTFI** | Time to first identification (frames from first appearance to correct match) |
| **UFR** | Unknown fragmentation rate: `unknown_slugs / GT_subjects` |

## Ablation Results

### Insightface backend — tracker x det_every x threshold grid

| Tracker | det_every | Threshold | FPS | Det p50 ms | Det p95 ms | IDSW | Frag | UFR | TTFI mean | TTFI=-1 | Peak RSS MiB |
|---------|-----------|-----------|-----|-----------|-----------|------|------|-----|-----------|---------|-------------|
| simple  | 3         | 0.3       | 32.1 | 80.3     | 133.6     | 0    | 79   | 6.1 | 21.0      | 4/10    | 316         |
| simple  | 3         | 0.4       | 33.8 | 79.7     | 127.2     | 0    | 79   | 6.7 | 57.0      | 4/10    | 319         |
| simple  | 5         | 0.3       | 49.5 | 88.6     | 137.6     | 0    | 83   | 5.8 | 43.2      | 4/10    | 318         |
| simple  | 5         | 0.4       | 51.3 | 85.6     | 137.5     | 0    | 83   | 5.9 | 57.0      | 4/10    | 318         |
| kalman  | 3         | 0.3       | 30.8 | 87.7     | 136.3     | 0    | 74   | 5.2 | 38.7      | 4/10    | 319         |
| kalman  | 3         | 0.4       | 32.5 | 84.0     | 130.5     | 0    | 74   | 5.9 | 57.0      | 4/10    | 319         |
| kalman  | 5         | 0.3       | 48.9 | 89.4     | 139.8     | 0    | 79   | 4.9 | 43.2      | 4/10    | 318         |
| kalman  | 5         | 0.4       | 50.5 | 87.0     | 140.7     | 0    | 79   | 5.6 | 57.0      | 4/10    | 318         |

### Backend comparison (simple tracker, det_every=3, threshold=0.3)

| Backend     | FPS   | Det p50 ms | Det p95 ms | Enrolled | UFR  | TTFI mean | TTFI=-1 | Peak RSS MiB |
|-------------|-------|-----------|-----------|----------|------|-----------|---------|-------------|
| insightface | 32.1  | 80.3      | 133.6     | 32/48    | 6.1  | 21.0      | 4/10    | 316         |
| scrfd       | 31.4  | 86.8      | 132.7     | 32/48    | 6.0  | 21.0      | 4/10    | 278         |
| ultraface   | 112.1 | 12.3      | 59.6      | 0/48     | 11.4 | -1.0      | 10/10   | 256         |

### Alignment and embedding (kalman, det_every=3, threshold=0.3)

| Variant         | FPS  | Det p50 ms | Avg embed ms | UFR | TTFI mean | Peak RSS MiB |
|-----------------|------|-----------|-------------|-----|-----------|-------------|
| cv2 + FP32      | 30.8 | 87.7      | 23.0        | 5.2 | 38.7      | 319         |
| skimage + FP32  | 27.1 | 98.6      | 24.7        | 5.2 | 38.7      | 318         |
| cv2 + INT8      | 28.2 | 94.8      | 25.3        | 5.1 | 38.7      | 316         |

## Analysis

### Tracker: Kalman vs Simple

Kalman tracking consistently produces **lower fragmentation** (74 vs 79 at `det_every=3`) and **lower UFR** (best: 4.9 vs 5.8). The Kalman filter predicts box positions during tracking-only frames rather than holding them static, maintaining smoother trajectories. The cost is ~2 FPS (30.8 vs 32.1), which is negligible relative to the quality improvement.

### Detection frequency: det_every=3 vs 5

Setting `det_every=5` boosts throughput by ~55% (32 FPS to 49-51 FPS) since detection runs on only 20% of frames instead of 33%. The trade-off is slower first identification — Deepika Padukone's TTFI increases from 55 to 188 frames because embedding refreshes happen less frequently. Fragmentation increases slightly (79 to 83 with simple tracker). For latency-sensitive applications, `det_every=3` is preferable; for throughput-bound scenarios (e.g., batch video processing), `det_every=5` is viable.

### Match threshold: 0.3 vs 0.4

Threshold 0.3 is strictly better on this dataset. At 0.4, TTFI mean increases from 21 to 57 frames (Deepika: 55 to 271 frames) with no measurable gain in precision — zero ID switches at both thresholds. The stricter threshold delays recognition without preventing false matches.

### Backend: insightface vs SCRFD vs UltraFace

**SCRFD** (standalone reimplementation) produces identical recognition quality to the insightface library wrapper — same enrollment count (32), same TTFI per subject, same fragmentation. It uses **38 MiB less memory** (278 vs 316 peak RSS) since it avoids loading the insightface library. FPS is within measurement noise (31.4 vs 32.1).

**UltraFace** is 3.5x faster (112 FPS, 12ms detect) but **cannot perform recognition** at 180x320 resolution. It produces no facial landmarks, forcing center-crop fallback for alignment. The resulting embeddings are too poor for gallery matching — zero enrollments succeed, all subjects remain unidentified.

### Alignment: cv2 vs skimage

Recognition quality is identical (same TTFI, same UFR, same fragmentation). skimage is **12% slower** — detection-frame p50 increases from 87.7ms to 98.6ms (~11ms overhead from scipy's `SimilarityTransform`). There is no reason to prefer skimage over cv2 on this hardware.

### Embedding: FP32 vs INT8

INT8 quantization produces identical recognition quality (same TTFI per subject, marginally better UFR 5.1 vs 5.2). However, INT8 is **slightly slower** on the Pi 5's Cortex-A76 CPU — avg embed 25.3ms vs 23.0ms. This is expected: ONNX Runtime's INT8 codepath benefits from x86 VNNI instructions that ARM lacks, and the quantize/dequantize overhead exceeds the savings from reduced arithmetic precision. Memory savings are negligible (3 MiB). INT8 is not recommended on this hardware.

### Invariants across all configurations

- **Zero ID switches** — the pipeline never confuses one enrolled identity for another
- **4 subjects always TTFI=-1** (Roger Federer, Rihanna, Chimamanda Ngozi Adichie, Iman Vellani) — this is an enrollment problem (auto-enrollment likely captures a distractor face), not a tracker or threshold issue
- **Memory stays under 320 MiB** — well within the 4 GiB cap
- **TCR = 1.0** — perfect temporal consistency (no identity flicker)

### Unresolved: TTFI=-1 subjects

Four subjects are never correctly identified across all configurations. The likely cause is that `auto_enroll_from_clips()` captures a distractor face (interviewer, bystander) in the enrollment frame instead of the target celebrity. These subjects have no mugshot images available, so auto-enrollment is their only enrollment path. A targeted fix would add GT-guided enrollment (use the GT bounding box to select the correct face in the enrollment frame).

## Pipeline Latency Benchmark

Single-image benchmark on a 180x320 FANVID frame, 50 iterations with 5 warmup runs, `det_every=3`.

| Metric | Value |
|--------|-------|
| Avg loop latency | 36.6 ms |
| Avg FPS | 27.3 |
| Avg detection (when run) | 80.8 ms |
| Avg embedding per frame | 9.0 ms |
| Faces detected per frame | 1 |
| Model init RSS delta | 117.6 MiB |
| Peak RSS | 261.8 MiB |

## LFW Verification Benchmark

Standard LFW protocol: train threshold on dev-train split (1100 match + 1100 mismatch pairs), evaluate on dev-test split (500 + 500 pairs). Deep-funneled images, insightface SCRFD detection, cv2 alignment, FP32 ArcFace embedding.

| Metric | Value |
|--------|-------|
| Test accuracy | **96.88%** |
| ROC-AUC | 0.9674 |
| EER | 5.63% (@ threshold 0.107) |
| TAR @ FAR <= 1% | 93.99% (@ threshold 0.256) |
| Best threshold (train) | 0.224 |
| Face detection rate | 99.76% (17/4992 dropped) |
| Peak RSS | 278.5 MiB |

Compared to Valenia's Milestone 2 baseline (94.75% FP32), the unified pipeline achieves **+2.13 percentage points** improvement. This is likely due to the deep-funneled image variant (better pre-alignment) and improved alignment implementation.

## Recommended configuration

For the Raspberry Pi 5 live dashboard use case:

```yaml
detection:
  backend: "insightface"  # or "scrfd" for lower memory
tracking:
  method: "kalman"
live:
  det_every: 3
  match_threshold: 0.3
```

This configuration provides the best balance: lowest UFR (5.2), lowest fragmentation (74), fast recognition (TTFI mean 38.7), and ~31 FPS — sufficient for real-time display at the `target_fps: 15` setting.

## How to reproduce

```bash
# Download FANVID dataset (50 clips)
uv run --python 3.13 python scripts/download_fanvid.py \
    --output-dir data/fanvid --max-clips 50

# Video benchmark (FANVID ablation)
uv run --python 3.13 python scripts/benchmark.py video \
    --config config.yaml --fanvid-dir data/fanvid --max-clips 50 \
    --override metrics.json_path= \
    --override tracking.method=kalman --override live.det_every=3 \
    --override live.match_threshold=0.3 \
    --output-json data/ablation_kalman_d3_t03.json

# Pipeline latency (single image)
uv run --python 3.13 python scripts/benchmark.py pipeline \
    --config config.yaml --mode image --image <path_to_image> \
    --runs 50 --override metrics.json_path= \
    --output-json data/bench_pipeline.json

# LFW verification (requires LFW deep-funneled images + pairs files)
uv run --python 3.13 python scripts/benchmark.py lfw \
    --config config.yaml \
    --lfw-dir data/lfw/lfw-deepfunneled \
    --train-pairs data/lfw/pairsDevTrain.txt \
    --test-pairs data/lfw/pairsDevTest.txt \
    --override metrics.json_path= \
    --output-json data/bench_lfw.json
```
