# Milestone 2 -- Technical Report

**Project:** Real-Time Face Recognition on Raspberry Pi 5
**Setup:** valenia
**Date:** 2026-03-01
**Repository:** [github.com/dl4cv-ai-pinnacle/rpi-face-recognition](https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition)
**Branch:** [feat/#1-valenia](https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition/tree/feat/%231-valenia)

*This report was generated with the help of Claude Opus 4.6.*

---

## 1. System Overview

| Component | Spec |
|-----------|------|
| Board | Raspberry Pi 5, 8 GB RAM |
| OS | Debian 13 (trixie), kernel 6.12 |
| Camera | Raspberry Pi Camera Module (via `rpicam`) |
| Inference | ONNX Runtime CPUExecutionProvider (ARM Cortex-A76, 4 cores) |
| Language | Python 3.13 |
| Package mgmt | `uv` with `pyproject.toml` |
| Quality gates | ruff format + ruff check + pyright + pytest (pre-commit) |

---

## 2. Pipeline Architecture

```
                         +-----------+
                         |  Camera   |
                         | (rpicam)  |
                         +-----+-----+
                               |
                          BGR frame
                               |
                   +-----------v-----------+
                   |    SCRFD-500M         |
                   |    Face Detector      |
                   |    (~21 ms / frame)   |
                   +-----------+-----------+
                               |
                   boxes + 5-point landmarks
                               |
              +----------------v----------------+
              |       IoU Face Tracker          |
              |   - greedy IoU box matching     |
              |   - exponential box smoothing   |
              |   - configurable hold frames    |
              +----------------+----------------+
                               |
                    tracked faces (with IDs)
                               |
              +----------------v----------------+
              |    Embedding Refresh Logic       |
              |    - new track? -> embed         |
              |    - stale / moved? -> re-embed  |
              |    - stable known? -> reuse      |
              +----------------+----------------+
                               |
                          (if embedding)
                               |
              +----------------v----------------+
              |    ArcFace 5-point Alignment     |
              |    -> 112x112 BGR crop           |
              +----------------+----------------+
                               |
              +----------------v----------------+
              |    MobileFaceNet Embedder        |
              |    512-d L2-normalized           |
              |    (~28 ms / face)               |
              +----------------+----------------+
                               |
                     512-d embedding
                               |
         +---------------------v---------------------+
         |          Gallery Matching                  |
         |  cosine similarity vs enrolled templates   |
         |  threshold: 0.228 (from LFW calibration)  |
         +-----+-------------------+-----------------+
               |                   |
          matched              not matched
               |                   |
    +----------v--------+   +-----v-----------+
    | Identity Label    |   | Auto-Capture    |
    | + auto-enrichment |   | Unknown Inbox   |
    +-------------------+   +-----------------+
```

The pipeline is composed of swappable components (`FacePipeline` = detector +
embedder). A factory (`pipeline_factory.py`) builds variants from CLI flags,
making A/B comparisons trivial (e.g. FP32 vs INT8 embedder on the same image).

---

## 3. Models

| Model | Task | Output | Size |
|-------|------|--------|------|
| SCRFD-500M (`det_500m.onnx`) | Face detection | Boxes + 5-point landmarks | ~2 MB |
| MobileFaceNet (`w600k_mbf.onnx`) | Embedding | 512-d float32 vector | 12.99 MB |
| MobileFaceNet INT8 (`w600k_mbf.int8.onnx`) | Embedding | 512-d float32 vector | 8.39 MB |

Both from the InsightFace `buffalo_sc` pack. INT8 model produced via
`onnxruntime.quantization.quantize_dynamic` (MatMul + Gemm, per-tensor QInt8).

---

## 4. Runtime Optimizations

These make the pipeline viable on a 4-core ARM CPU with no GPU:

### Detect Cadence (`det_every`)

The face detector is the most expensive stage (~21 ms). With `--det-every N`,
detection runs only every Nth frame. Between passes, the tracker holds and
smooths boxes via exponential averaging. At `det-every 2` detector cost is
effectively halved with minimal visual impact.

### Selective Embedding Refresh

Embedding costs ~28 ms per face. Rather than re-embedding every face every
frame, the runtime applies three rules:

- **New track** -- always embed (first sighting)
- **Stale / moved** -- re-embed after N frames (default 5) or when box IoU
  vs last embedded position drops below 0.85. Known high-confidence identities
  get a tighter re-embed cadence (2 frames max) to keep labels responsive.
- **Stable known** -- reuse cached embedding and match result entirely

In a typical 2-face scene this cuts embedding work by 50-75%.

### Identity Grace Period

When a recognized identity momentarily drops below threshold (brief head turn,
partial occlusion), the system holds the label for one frame if: the prior
match was confirmed with margin >= 0.08, only one frame elapsed, and the box
hasn't moved (IoU >= 0.90). Prevents label flickering.

### Track Smoothing

Boxes and landmarks are exponentially smoothed (factor 0.65) across frames.
Reduces visual jitter and makes the IoU-based "has the face moved?" check
less noisy.

### Quality-Gated Auto-Enrichment

Live frames automatically strengthen enrolled identities, but only when:

1. Match confidence exceeds threshold by margin >= 0.10
2. Face quality >= 0.40 (geometric mean of detector confidence and normalized
   face area -- rejects tiny, blurry, or partial crops)
3. Cooldown of 30s since last enrichment for the same identity
4. Cosine similarity < 0.95 against all existing samples (diversity check)
5. If at 48-sample cap, the lowest-quality sample is evicted

Templates are quality-weighted averages -- better samples contribute more
to the matching vector.

### RAM Cap

Configurable `--ram-cap-mb` checked every frame via `/proc/self/status`.
Clean exit before the OOM killer fires.

### systemd Service

The live system runs as a `systemd` unit managed by a helper script
(`manage_live_camera_service.sh up/down/restart/status/logs`). This means
the Pi boots into face recognition mode automatically, restarts on crash,
and logs go to `journalctl`.

### Live Metrics Collection

The runtime records a rich set of per-frame metrics and exposes them both
in the web dashboard and as a JSON file (`/metrics.json`):

- **Latency breakdown:** detection ms, tracking ms, embedding ms per frame
- **Throughput:** output FPS, processing FPS, average loop time
- **Face stats:** face count, fresh tracks, recognized faces, embedding
  refreshes vs reuses per frame
- **Hardware health:** CPU usage %, 1/5/15-min load averages, SoC temperature,
  GPU usage (when available), current and peak RSS

All of these are shown live in the browser dashboard with tooltip
explanations. The metrics JSON is also written to disk periodically for
offline analysis. This is not a formal video benchmark (we don't measure
recognition accuracy over a clip), but it gives good operational visibility
into how the hardware performs under real load. Whether we need a dedicated
video benchmark or whether these live metrics are sufficient is something
we'll evaluate in the next phase.

---

## 5. LFW Validation

Standard face verification benchmark. View2 protocol: 6000 pairs, 10-fold
cross-validation, threshold selected on train folds.

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| View2 10-fold accuracy | 94.75% | 94.72% | -0.03 pp |
| ROC-AUC | 0.9325 | 0.9326 | +0.0001 |
| EER | 11.20% | 11.20% | 0 |
| TAR @ FAR <= 1% | 88.00% | 88.00% | 0 |
| Avg detect latency | 20.84 ms | 21.65 ms | +0.81 ms |
| Avg embed latency | 28.64 ms | 28.01 ms | -0.63 ms |
| Peak RSS | 271.94 MB | 265.94 MB | -6.00 MB |

**Takeaway:** INT8 is a free lunch -- 35% smaller model, ~0.6 ms faster
embeddings, 6 MB less RAM, zero accuracy cost.

---

## 6. Image Benchmarks (Raspberry Pi 5)

### Single-Image Latency

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| Avg loop latency | 70.39 ms | 54.27 ms | -16.12 ms |
| Avg FPS | 14.21 | 18.43 | +4.22 (1.30x) |
| Avg detect latency | 23.44 ms | 21.84 ms | -1.60 ms |
| Avg embed latency | 28.88 ms | 27.89 ms | -0.99 ms |
| Peak RSS | 233.95 MB | 228.25 MB | -5.70 MB |

### A/B Variant Comparison (same session)

| Variant | Avg loop ms | Avg FPS |
|---------|-------------|---------|
| FP32 | 58.81 | 17.00 |
| INT8 | 52.78 | 18.95 |

### Gap: Video-Level Recognition Benchmark

We don't yet have a formal benchmark that measures recognition accuracy and
stability on a recorded video clip. The image benchmark captures per-frame
latency, and the live metrics give good visibility into hardware performance
(FPS, latency breakdown, CPU/temp/memory) under real load -- but neither
evaluates tracking stability or identity assignment accuracy across a
multi-second sequence. We will consider whether the live metrics already
cover enough for our use case or whether a dedicated video benchmark is
needed.

---

## 7. Gallery and Auto-Detection

### How It Works

```
  Live frame --> detect + track + embed
      |
  gallery.match(embedding, threshold=0.228)
      |
  +---+---+
  |       |
matched  unmatched
  |       |
  |   gallery.capture_unknown(embedding, crop)
  |       |
  |   match against unknown templates (threshold=0.36)
  |       |
  |   +---+---+
  |   |       |
  |  known   new
  |  unknown  unknown
  |   |       |
  |  update   create
  |  record   new record
  |
  +-> auto-enrich (if quality + diversity + cooldown pass)
```

### Storage Layout

```
data/gallery/
  alice/
    meta.json        # name, slug, sample_count, sample_qualities
    template.npy     # 512-d quality-weighted mean template
    samples.npy      # Nx512 embedding matrix
    upload_001.jpg   # stored face crops
  _unknowns/
    unknown-0001/
      meta.json      # slug, display_name, timestamps
      template.npy
      samples.npy
      capture_001.jpg
```

### Known Problems

1. **Family member confusion** -- my mum was repeatedly labelled as me during
   initial setups. Our main countermeasure was collecting more diverse
   embeddings per person (auto-enrichment), making each template more
   discriminative over time. This helped, but the thresholds and heuristics
   were tuned empirically in a rush rather than grounded in literature
2. **Unknown fragmentation** -- same person across sessions creates multiple
   unknown entries (merge-unknowns helps post-hoc, but the root cause is a
   loose unknown-matching threshold)
3. **Template pollution** -- low-quality auto-enrichment samples can degrade
   the identity template despite quality gating

### What We Did About It

- Quality scoring gates enrichment on face size and detector confidence
- Diversity filter prevents near-duplicate embedding inflation
- Cooldown and sample cap with worst-quality eviction
- Merge-unknowns UI consolidates duplicate inbox entries
- Quality-weighted template averaging

### What Still Needs Work

- Proper literature review on template adaptation and open-set identification
  -- current tuning was empirical and rushed
- Per-identity adaptive thresholds
- Stronger quality model (pose, blur)
- User-confirmed enrichment option
- Video-level recognition stability evaluation

---

## 8. Test Suite

33 tests across 4 files, all running with stubbed `cv2` and protocol-based
dependency injection (no real models or camera needed):

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_gallery_store.py` | 17 | Enroll, match, promote, merge, rename, delete, enrich, upload |
| `test_live_runtime.py` | 9 | Tracking, embedding cache, enrichment, metrics, delegation |
| `test_pipeline_factory.py` | 3 | Size parsing, path resolution, factory injection |
| `test_quality.py` | 4 | Quality scoring edge cases |

---

## 9. Reproducibility

All benchmark runs produce JSON artifacts in `docs/metrics/`. To reproduce:

```bash
# One-time setup
./slop/valenia/scripts/bootstrap_pi.sh
./slop/valenia/scripts/download_models.sh
./slop/valenia/scripts/download_lfw_dataset.sh

# LFW validation
python3 -u slop/valenia/scripts/evaluate_lfw.py \
  --view2-pairs data/lfw/pairs.txt \
  --output-json docs/metrics/lfw_view2_fp32.json

# Image benchmark
python3 slop/valenia/scripts/benchmark_pipeline.py \
  --mode image --image data/lena.jpg \
  --warmup-runs 5 --runs 50 \
  --output-json docs/metrics/image_benchmark.json

# Live system
python3 slop/valenia/scripts/live_camera_server.py --det-every 2
```

Full setup instructions: [`slop/valenia/README.md`](https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition/blob/feat/%231-valenia/slop/valenia/README.md)
