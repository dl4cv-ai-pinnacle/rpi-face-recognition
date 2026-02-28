# Initial Benchmark (2026-02-12)

## Context

- Device: Raspberry Pi 5 (8GB)
- OS: Debian 13 (trixie)
- Models: `buffalo_sc` (`det_500m.onnx`, `w600k_mbf.onnx`)
- Runtime: ONNX Runtime CPU (`python3-onnxruntime` package)

## Commands Run

```bash
python3 scripts/benchmark_pipeline.py \
  --mode image \
  --image data/lena.jpg \
  --runs 30 \
  --det-every 3 \
  --save-output data/lena_annotated.jpg
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

## Immediate Next Step

1. Attach/enable camera and verify:
```bash
rpicam-hello --list-cameras
```
2. Run live benchmark:
```bash
python3 scripts/benchmark_pipeline.py --mode camera --frames 300 --det-every 3
```

## Offline Validation Available

If camera is not attached yet, run dataset-based quality + performance validation:

```bash
./scripts/download_lfw_dataset.sh
python3 -u scripts/evaluate_lfw.py --view2-pairs data/lfw/pairs.txt --output-json docs/metrics/lfw_dev_view2_baseline.json
```
