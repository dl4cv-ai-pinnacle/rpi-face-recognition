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
- Offline validation with reproducible metrics
  - `slop/valenia/scripts/evaluate_lfw.py`
  - `slop/valenia/docs/metrics/`
- Live camera browser stream (ready to run when a camera is attached)
  - `slop/valenia/scripts/live_camera_server.py`
  - `slop/valenia/docs/LIVE_CAMERA_SERVER.md`

## Not Done Yet

- Stage 3: MNN/ncnn runtime comparison
- Stage 6: multi-face tracking
- Stage 7: gallery, SQLite, FAISS, enrollment storage
- Stage 8: TTS/actions
- Stage 9: anti-spoofing
- Face memory / remembering mode
- Recognition against enrolled identities
- Verified quantized runtime delta on this Pi

## Current Milestone Read

- The Milestone 1 baseline pipeline is implemented and measured.
- The full milestone vision is not complete yet.
- The next buildable step is gallery + enrollment, then tracking.
