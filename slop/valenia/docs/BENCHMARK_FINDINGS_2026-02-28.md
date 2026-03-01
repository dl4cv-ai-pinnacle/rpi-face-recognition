# Benchmark Findings (2026-02-28)

All runs below were executed on the Raspberry Pi 5 with a 4 GiB RSS cap enforced.

## FP32 Baseline

### Image Benchmark

Command:

```bash
python3 -u slop/valenia/scripts/benchmark_pipeline.py \
  --mode image \
  --image data/lena.jpg \
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

## INT8 Quantized Embedder

The recognition model can be quantized on this Pi with Python `3.13`, but the
conversion must run in an isolated `uv` environment with newer PyPI wheels.
The Debian-packaged `python3-onnx` + `python3-onnxruntime` combination still
fails when importing `onnxruntime.quantization`.

### Quantization Command

```bash
uv run --isolated --python 3.13 \
  --with onnx==1.20.1 \
  --with onnxruntime==1.24.2 \
  python -c 'from pathlib import Path; import time; from onnxruntime.quantization import QuantType, quantize_dynamic; src=Path("slop/valenia/models/buffalo_sc/w600k_mbf.onnx"); dst=Path("slop/valenia/models/buffalo_sc/w600k_mbf.int8.onnx"); t0=time.perf_counter(); quantize_dynamic(model_input=str(src), model_output=str(dst), op_types_to_quantize=["MatMul","Gemm"], weight_type=QuantType.QInt8, per_channel=False); dt=(time.perf_counter()-t0)*1000; print(f"OUTPUT={dst}"); print(f"QUANTIZE_MS={dt:.2f}"); print(f"SRC_SIZE={src.stat().st_size}"); print(f"DST_SIZE={dst.stat().st_size}")'
```

Quantization result:

- Output model: `slop/valenia/models/buffalo_sc/w600k_mbf.int8.onnx`
- Quantization time: `331.57 ms`
- Source size: `12.99 MiB`
- Quantized size: `8.39 MiB`
- Size reduction: `4.59 MiB` (`35.4%`)

### LFW View2 Evaluation (INT8)

Command:

```bash
python3 -u slop/valenia/scripts/evaluate_lfw.py \
  --view2-pairs data/lfw/pairs.txt \
  --rec-model models/buffalo_sc/w600k_mbf.int8.onnx \
  --ram-cap-mb 4096 \
  --output-json docs/metrics/lfw_view2_int8_2026-02-28.json
```

Results:

- Train accuracy: `0.9491`
- Test accuracy: `0.9370`
- ROC-AUC: `0.9326`
- EER: `0.1120`
- TAR@FAR<=1e-2: `0.8800`
- View2 10-fold accuracy: `0.9472` (std `0.0094`)
- Avg detection latency: `21.65 ms/image`
- Avg embedding latency: `28.01 ms/face`
- Preprocess throughput: `19.57 images/s`
- Peak RSS: `265.94 MiB`

FP32 delta:

- Train accuracy: unchanged
- Test accuracy: unchanged
- ROC-AUC: `+0.000058`
- View2 10-fold accuracy: `-0.000333` (`-0.033` percentage points)
- Avg detection latency: `+0.81 ms/image`
- Avg embedding latency: `-0.63 ms/face`
- Peak RSS: `-6.00 MiB`

JSON artifact:

- `slop/valenia/docs/metrics/lfw_view2_int8_2026-02-28.json`

### Image Benchmark (INT8)

Command:

```bash
python3 -u slop/valenia/scripts/benchmark_pipeline.py \
  --mode image \
  --image data/lena.jpg \
  --runs 50 \
  --det-every 1 \
  --rec-model models/buffalo_sc/w600k_mbf.int8.onnx \
  --ram-cap-mb 4096
```

Results:

- Frames: `50`
- Avg loop latency: `54.27 ms`
- Avg FPS: `18.43`
- Avg detection latency: `21.84 ms`
- Avg embedding latency: `27.89 ms/frame`
- Peak RSS: `228.25 MiB`

FP32 delta:

- Avg loop latency: `-16.12 ms`
- Avg FPS: `+4.22` (`1.30x`)
- Avg detection latency: `-1.60 ms`
- Avg embedding latency: `-0.99 ms/frame`
- Peak RSS: `-5.70 MiB`

### Variant Comparison Script (FP32 vs INT8)

Command:

```bash
python3 slop/valenia/scripts/compare_pipeline_variants.py \
  --image data/lena.jpg \
  --runs 30 \
  --a-label fp32 \
  --a-rec-model models/buffalo_sc/w600k_mbf.onnx \
  --b-label int8 \
  --b-rec-model models/buffalo_sc/w600k_mbf.int8.onnx \
  --output-json docs/metrics/fp32_vs_int8_image_compare.json
```

Results:

- FP32 avg loop latency: `58.81 ms`
- INT8 avg loop latency: `52.78 ms`
- FP32 avg FPS: `17.00`
- INT8 avg FPS: `18.95`
- Delta loop latency: `-6.03 ms`
- Delta FPS: `+1.94`

JSON artifact:

- `slop/valenia/docs/metrics/fp32_vs_int8_image_compare.json`

## Practical Takeaway

- Python `3.13` is viable for quantization on this Pi.
- The blocker is only the Debian ONNX package set used by system `python3`.
- Runtime inference can stay on the current system stack.
- Quantization should use `uv --isolated`, then the generated INT8 model can be
  passed into the existing scripts with `--rec-model`.
