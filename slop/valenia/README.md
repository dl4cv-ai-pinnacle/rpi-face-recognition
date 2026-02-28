# Valenia Setup

This is the active Raspberry Pi face-recognition setup.

## Quick Start

Run commands from the repo root.

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
  --output-json slop/valenia/docs/metrics/lfw_dev_baseline.json
```

5. Run full view2 validation:

```bash
python3 -u slop/valenia/scripts/evaluate_lfw.py \
  --view2-pairs data/lfw/pairs.txt \
  --output-json slop/valenia/docs/metrics/lfw_dev_view2_baseline.json
```

6. Run the image benchmark:

```bash
python3 slop/valenia/scripts/benchmark_pipeline.py \
  --mode image \
  --image slop/valenia/data/lena.jpg \
  --runs 50 \
  --det-every 1 \
  --save-output slop/valenia/data/lena_annotated.jpg
```

## INT8 Quantization

Python `3.13` works here for quantization. The only broken path is Debian's
system `python3-onnx` + `python3-onnxruntime` import path for
`onnxruntime.quantization`.

Use `uv --isolated` to create the INT8 model, then run the normal scripts with
`--rec-model`:

```bash
uv run --isolated --python 3.13 \
  --with onnx==1.20.1 \
  --with onnxruntime==1.24.2 \
  python -c 'from pathlib import Path; from onnxruntime.quantization import QuantType, quantize_dynamic; src=Path("slop/valenia/models/buffalo_sc/w600k_mbf.onnx"); dst=Path("slop/valenia/models/buffalo_sc/w600k_mbf.int8.onnx"); quantize_dynamic(model_input=str(src), model_output=str(dst), op_types_to_quantize=["MatMul","Gemm"], weight_type=QuantType.QInt8, per_channel=False)'
```

## Tooling

```bash
uv sync --project slop/valenia --group dev
uv run --project slop/valenia --group dev pre-commit install \
  --config slop/valenia/.pre-commit-config.yaml
uv run --project slop/valenia --group dev pyright -p slop/valenia --pythonpath /usr/bin/python3
uv run --project slop/valenia --group dev pre-commit run \
  --config slop/valenia/.pre-commit-config.yaml \
  --all-files
```

## Live Camera

```bash
rpicam-hello --list-cameras
python3 slop/valenia/scripts/live_camera_server.py --ram-cap-mb 4096
```

Then open `http://<pi-ip>:8000/` from another device on the same network.
Detailed notes: `slop/valenia/docs/LIVE_CAMERA_SERVER.md`

## Notes

- Code: `slop/valenia/src/`
- Workflows: `slop/valenia/scripts/`
- Findings and plans: `slop/valenia/docs/`
- Imported reports: `slop/valenia/docs/imported/`
- Milestone status: `slop/valenia/docs/MILESTONE_1_STATUS.md`
- Latest measured findings: `slop/valenia/docs/BENCHMARK_FINDINGS_2026-02-28.md`
