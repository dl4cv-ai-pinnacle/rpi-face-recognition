# LFW Validation Baseline (2026-02-12)

This repository now includes an offline validation pipeline that runs the same core stack used for real-time inference:

1. `SCRFD-500M` face detection
2. ArcFace 5-point alignment (`112x112`)
3. `MobileFaceNet` embedding (`w600k_mbf.onnx`)
4. Cosine-similarity verification on LFW pairs

## Dataset Setup

```bash
./scripts/download_lfw_dataset.sh
```

This downloads and verifies:

- `lfw-funneled.tgz`
- `pairsDevTrain.txt`
- `pairsDevTest.txt`
- `pairs.txt` (official 10-fold view2 protocol)

## Run Validation

Dev train/test split:

```bash
python3 -u scripts/evaluate_lfw.py \
  --output-json docs/metrics/lfw_dev_baseline.json
```

Dev split + official view2 cross-validation:

```bash
python3 -u scripts/evaluate_lfw.py \
  --view2-pairs data/lfw/pairs.txt \
  --output-json docs/metrics/lfw_dev_view2_baseline.json
```

## Protocol

- Train threshold selection:
  - Find the best cosine threshold on `pairsDevTrain.txt` (2200 pairs).
- Holdout evaluation:
  - Report metrics on `pairsDevTest.txt` (1000 pairs) using train-selected threshold.
- View2 cross-validation:
  - Run official 10-fold protocol from `pairs.txt` (6000 pairs).
  - Per fold: select threshold on 9 folds, evaluate on held-out fold.
  - Report mean accuracy and standard deviation across folds.

## Reported Metrics

- `accuracy`: verification accuracy at selected threshold.
- `ROC-AUC`: area under ROC curve.
- `EER`: equal error rate (where FAR and FRR are equal).
- `TAR@FAR<=1e-2`: true accept rate under low false accept constraint.
- `detection latency`: mean detector runtime per image.
- `embedding latency`: mean embedding runtime per detected/aligned face.
- `throughput`: processed images/pairs per second during offline evaluation.

## Baseline Results on This Pi

Machine:
- Raspberry Pi 5 (8GB)
- Debian 13 (trixie)
- ONNX Runtime CPU

From `docs/metrics/lfw_dev_view2_baseline.json`:

- Train best threshold: `0.2280`
- Train accuracy: `0.9491`
- Test accuracy (threshold from train): `0.9370`
- Test ROC-AUC: `0.9326`
- Test EER: `0.1120`
- Test TAR@FAR<=1e-2: `0.8800`
- View2 10-fold accuracy: `0.9475` (std `0.0095`)
- Avg detection latency: `22.11 ms / image`
- Avg embedding latency: `29.17 ms / face`
- Preprocess throughput: `18.96 images/s`

## Notes

- This benchmark is camera-free and fully reproducible on stored images.
- It validates both recognition quality and per-stage runtime before live camera deployment.
- JSON outputs are meant for tracking improvements across model/runtime changes.
- This is a strong baseline protocol but not a production security certification benchmark.
