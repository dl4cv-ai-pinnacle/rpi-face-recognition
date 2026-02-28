# Benchmark Findings (2026-02-28)

All runs below were executed on the Raspberry Pi 5 with a 4 GiB RSS cap enforced.

## FP32 Baseline

### Image Benchmark

Command:

```bash
python3 -u slop/valenia/scripts/benchmark_pipeline.py \
  --mode image \
  --image slop/valenia/data/lena.jpg \
  --runs 50 \
  --det-every 1 \
  --ram-cap-mb 4096
```

Results:

- Frames: `50`
- Avg loop latency: `70.39 ms`
- Avg FPS: `14.21`
- Avg detection latency: `23.44 ms`
- Avg embedding latency: `28.88 ms/frame`
- Peak RSS: `233.95 MiB`

### LFW View2 Evaluation

Command:

```bash
python3 -u slop/valenia/scripts/evaluate_lfw.py \
  --view2-pairs data/lfw/pairs.txt \
  --ram-cap-mb 4096 \
  --output-json docs/metrics/lfw_view2_fp32_2026-02-28.json
```

Results:

- Train accuracy: `0.9491`
- Test accuracy: `0.9370`
- ROC-AUC: `0.9325`
- EER: `0.1120`
- TAR@FAR<=1e-2: `0.8800`
- View2 10-fold accuracy: `0.9475` (std `0.0095`)
- Avg detection latency: `20.84 ms/image`
- Avg embedding latency: `28.64 ms/face`
- Preprocess throughput: `19.15 images/s`
- Peak RSS: `271.94 MiB`

JSON artifact:

- `slop/valenia/docs/metrics/lfw_view2_fp32_2026-02-28.json`

## Quantization Status

Attempted dynamic INT8 quantization for the ArcFace model is currently blocked on this machine.

Observed failure:

- Importing `onnxruntime.quantization` fails on Debian's `python3-onnx` +
  `python3-onnxruntime` combination with duplicate ONNX schema registration
  errors and a final `NotImplementedError` around duplicate `LRN` operator names.

Impact:

- No valid INT8 delta is available yet on this Pi.
- The benchmark and evaluation scripts now fail fast with a short actionable
  message instead of a long stack trace when quantization is requested.

Recommended next step:

- Generate the quantized `.onnx` model in a separate compatible environment,
  then evaluate it here by passing `--rec-model <quantized_model_path>`.
