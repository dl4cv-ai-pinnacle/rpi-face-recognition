# Raspberry Pi 5 Execution Plan

This plan combines the two reports from issue `#1` and is tuned for this machine:
- Raspberry Pi 5 (8GB)
- Debian 13 (trixie), kernel `6.12.62+rpt-rpi-2712`

## Final Baseline Stack

- Detection: `SCRFD-500M` (`det_500m.onnx`)
- Alignment: ArcFace 5-point affine alignment (`112x112`)
- Embedding: `MobileFaceNet` (`w600k_mbf.onnx`)
- Tracking (phase 2): lightweight IoU/centroid, then optional Norfair
- Matching/storage (phase 3): cosine similarity + SQLite/FAISS
- Runtime now: ONNX Runtime (CPU) for fast bring-up
- Future optimization path: MNN/ncnn + FP16/INT8 after FP32 baseline is stable

## Why This Direction

- Both reports converge on `SCRFD + MobileFaceNet` for Pi edge balance.
- `buffalo_sc` gives a matched lightweight detector+recognizer pair.
- We should validate real Pi numbers first, then optimize.
- Report evidence suggests 10-15 FPS target is realistic for 1-3 faces with skip-frame detection.

## Phase Plan

1. Phase 0: Bring-up and instrumentation (today)
- Confirm camera, thermals, and model loading.
- Add repeatable benchmark scripts.
- Produce first latency/FPS measurements on this Pi.

2. Phase 1: Baseline pipeline correctness (next)
- Run detect -> align -> embed on camera frames.
- Add basic thresholded matching against a tiny local gallery.
- Validate preprocessing compatibility (RGB/BGR, normalization, template).

3. Phase 2: Real-time stability and tracking
- Run detection every `N` frames (`N=2..4`), track in between.
- Cache embeddings per track and refresh on interval/change.
- Measure FPS vs face count (1, 2, 3+).

4. Phase 3: Face memory and actions
- Enrollment flow with quality checks (pose/blur/min box size).
- Store identities + metadata in SQLite (FAISS when gallery grows).
- Event/action hooks (greetings, logging, automation trigger).

5. Phase 4: Optimization and hardening
- Try FP16/INT8 only after baseline is reliable.
- Compare ONNX Runtime vs MNN/ncnn for actual Pi throughput.
- Add thermal guardrails and long-run soak tests.

## Technical Guardrails

- Keep detector/alignment/embedding preprocessing coupled to ArcFace conventions.
- Treat 0.5 cosine threshold as initial default; tune with collected positives/negatives.
- Delay anti-spoofing until core pipeline is stable (add when required by threat model).
- Log timing per stage, not just total FPS, for each experiment run.
- Keep static quality gate green before benchmark commits:
  - `uv run --group dev pre-commit run --all-files`

## Success Criteria (Near-Term)

- Offline validation baseline (LFW) is reproducible and includes:
  - Verification accuracy metrics (train/test + view2 10-fold)
  - Threshold estimates for cosine matching
  - Per-stage runtime (detection and embedding)
- Camera pipeline runs continuously without crashes for 10+ minutes (when camera is attached).
- Runtime benchmark report includes:
  - Detection latency (ms)
  - Embedding latency per face (ms)
  - End-to-end FPS with skip-frame settings
- Recognition smoke test works with at least one enrolled identity.

## Immediate Actions Started

- Reports downloaded and imported under `docs/imported/`.
- `buffalo_sc` model bundle downloaded under `models/buffalo_sc/`.
- Initial benchmark scripts added in this repo for first on-device tests.
- LFW offline validation pipeline added:
  - Dataset bootstrap: `scripts/download_lfw_dataset.sh`
  - Validator: `scripts/evaluate_lfw.py`
  - Baseline metrics: `docs/LFW_VALIDATION.md`, `docs/metrics/`
