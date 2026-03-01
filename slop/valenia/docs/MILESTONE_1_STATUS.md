# Milestone 1 Status

Status checked against `MILESTONE_1.md`.

## Covered Now

- Stage 1: frame capture
  - `slop/valenia/scripts/benchmark_pipeline.py`
  - `slop/valenia/scripts/capture_test_frame.sh`
- Stage 2: face detection (`SCRFD`)
  - `slop/valenia/src/scrfd.py`
- Stage 4: face alignment
  - `slop/valenia/src/face_align.py`
- Stage 5: embedding extraction (`MobileFaceNet`)
  - `slop/valenia/src/arcface.py`
- Baseline pipeline wiring
  - `slop/valenia/src/pipeline.py`
- Stage 6: multi-face tracking
  - `slop/valenia/src/tracking.py`
  - `slop/valenia/src/live_runtime.py`
- Offline validation with reproducible metrics
  - `slop/valenia/scripts/evaluate_lfw.py`
  - `slop/valenia/docs/metrics/`
- Live camera browser stream (ready to run when a camera is attached)
  - `slop/valenia/scripts/live_camera_server.py`
  - `slop/valenia/docs/LIVE_CAMERA_SERVER.md`
- Stage 7 baseline: gallery enrollment + matching
  - `slop/valenia/src/gallery.py`
  - `slop/valenia/src/live_runtime.py`
- Recognition against enrolled identities
  - `slop/valenia/src/gallery.py`
  - `slop/valenia/scripts/live_camera_server.py`
- Verified quantized runtime delta on this Pi
  - `slop/valenia/docs/BENCHMARK_FINDINGS_2026-02-28.md`
  - `slop/valenia/docs/metrics/lfw_view2_int8_2026-02-28.json`

## Not Done Yet

- Stage 3: MNN/ncnn runtime comparison
- Stage 7 follow-up: SQLite/FAISS-backed storage (current gallery is filesystem-backed)
- Stage 8: TTS/actions
- Stage 9: anti-spoofing
- Face memory / remembering mode

## Current Milestone Read

- The Milestone 1 baseline pipeline is implemented and measured.
- Live tracking, enrollment, and named matching are available in the current software path.
- The remaining gap is the heavier follow-on work: alternate runtimes, durable storage, actions, and anti-spoofing.
