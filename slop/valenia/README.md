# Valenia Setup

This is the active Raspberry Pi face-recognition setup.

## Quick Start

Run commands from the repo root.
When a script accepts file paths, the path values are resolved relative to
`slop/valenia/` unless you pass an absolute path.

1. Install system packages and `uv` tooling:

```bash
./slop/valenia/scripts/bootstrap_pi.sh
```

2. Download baseline models:

```bash
./slop/valenia/scripts/download_models.sh
```

3. Download the LFW dataset:

```bash
./slop/valenia/scripts/download_lfw_dataset.sh
```

4. Run offline validation:

```bash
python3 -u slop/valenia/scripts/evaluate_lfw.py \
  --output-json docs/metrics/lfw_dev_baseline.json
```

5. Run full view2 validation:

```bash
python3 -u slop/valenia/scripts/evaluate_lfw.py \
  --view2-pairs data/lfw/pairs.txt \
  --output-json docs/metrics/lfw_dev_view2_baseline.json
```

6. Run the image benchmark:

```bash
python3 slop/valenia/scripts/benchmark_pipeline.py \
  --mode image \
  --image data/lena.jpg \
  --warmup-runs 5 \
  --runs 50 \
  --det-every 1 \
  --save-output data/lena_annotated.jpg \
  --output-json docs/metrics/image_benchmark_latest.json
```

## INT8 Quantization

Create the INT8 model with `uv --isolated`, then run the normal scripts with
`--rec-model`:

```bash
uv run --isolated --python 3.13 \
  --with onnx==1.20.1 \
  --with onnxruntime==1.24.2 \
  python -c 'from pathlib import Path; from onnxruntime.quantization import QuantType, quantize_dynamic; src=Path("slop/valenia/models/buffalo_sc/w600k_mbf.onnx"); dst=Path("slop/valenia/models/buffalo_sc/w600k_mbf.int8.onnx"); quantize_dynamic(model_input=str(src), model_output=str(dst), op_types_to_quantize=["MatMul","Gemm"], weight_type=QuantType.QInt8, per_channel=False)'
```

For the package nuance and measured INT8 deltas, see
`slop/valenia/docs/BENCHMARK_FINDINGS_2026-02-28.md`.

## Tooling

```bash
uv sync --project slop/valenia --group dev
uv run --project slop/valenia --group dev pre-commit install \
  --config slop/valenia/.pre-commit-config.yaml
uv run --project slop/valenia --group dev pytest
uv run --project slop/valenia --group dev pyright -p slop/valenia --pythonpath /usr/bin/python3
uv run --project slop/valenia --group dev pre-commit run \
  --config slop/valenia/.pre-commit-config.yaml \
  --all-files
```

## Variant Comparison

Use the shared variant runner to compare two swappable pipeline specs on the
same image:

```bash
python3 slop/valenia/scripts/compare_pipeline_variants.py \
  --image data/lena.jpg \
  --warmup-runs 5 \
  --runs 30 \
  --a-label fp32 \
  --a-rec-model models/buffalo_sc/w600k_mbf.onnx \
  --b-label int8 \
  --b-rec-model models/buffalo_sc/w600k_mbf.int8.onnx \
  --output-json docs/metrics/fp32_vs_int8_image_compare.json
```

Use `benchmark_pipeline.py` when you need reproducible per-variant RSS numbers.
Use `compare_pipeline_variants.py` for like-for-like latency/FPS comparisons only.

## Live Camera

```bash
rpicam-hello --list-cameras
python3 slop/valenia/scripts/live_camera_server.py \
  --det-every 2 \
  --ram-cap-mb 4096
```

Then open `http://<pi-ip>:8000/` from another device on the same network.
The page keeps the camera feed on the left, with enrollment and a live metrics
dashboard on the right. The dashboard surfaces FPS, CPU load, SoC temperature,
memory, detector cadence, and the raw metrics snapshot remains available at
`http://<pi-ip>:8000/metrics.json`.
Detailed notes: `slop/valenia/docs/LIVE_CAMERA_SERVER.md`

## Notes

- Code: `slop/valenia/src/`
- Workflows: `slop/valenia/scripts/`
- Findings and plans: `slop/valenia/docs/`
- Imported reports: `slop/valenia/docs/imported/`
- Milestone status: `slop/valenia/docs/MILESTONE_1_STATUS.md`
- Latest measured findings: `slop/valenia/docs/BENCHMARK_FINDINGS_2026-02-28.md`
