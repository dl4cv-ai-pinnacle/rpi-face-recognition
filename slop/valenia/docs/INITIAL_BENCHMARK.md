# Initial Benchmark (2026-02-12)

This file is the first bring-up snapshot. For the current measured baseline and
INT8 comparison, use `slop/valenia/docs/BENCHMARK_FINDINGS_2026-02-28.md`.

## Context

- Device: Raspberry Pi 5 (8GB)
- OS: Debian 13 (trixie)
- Models: `buffalo_sc` (`det_500m.onnx`, `w600k_mbf.onnx`)
- Runtime: ONNX Runtime CPU (`python3-onnxruntime` package)

## Commands Run

```bash
python3 slop/valenia/scripts/benchmark_pipeline.py \
  --mode image \
  --image slop/valenia/data/lena.jpg \
  --runs 30 \
  --det-every 3 \
  --save-output slop/valenia/data/lena_annotated.jpg
```

## Results

- Average loop latency: `22.60 ms`
- Average FPS: `44.25`
- Average detection latency (when run): `21.43 ms`
- Average embedding latency total/frame: `9.22 ms`
- Average faces detected/frame: `1.00`

## Notes

- The benchmark is currently image-mode only.
- Camera check returned `No cameras available!` via `rpicam-hello --list-cameras`.
- Pipeline code is ready for live camera benchmark as soon as the sensor is detected.

## What Changed Since Then

- The live path now uses `slop/valenia/src/live_runtime.py` and
  `slop/valenia/scripts/live_camera_server.py`.
- Tracking, gallery enrollment, live matching, and live metrics are now in
  place.
- Use `slop/valenia/docs/LIVE_CAMERA_SERVER.md` for the current live workflow.

## Offline Validation Available

If camera is not attached yet, run dataset-based quality + performance validation:

```bash
./slop/valenia/scripts/download_lfw_dataset.sh
python3 -u slop/valenia/scripts/evaluate_lfw.py \
  --view2-pairs slop/valenia/data/lfw/pairs.txt \
  --output-json slop/valenia/docs/metrics/lfw_dev_view2_baseline.json
```
