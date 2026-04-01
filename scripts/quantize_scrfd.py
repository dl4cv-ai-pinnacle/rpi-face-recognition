#!/usr/bin/env python3
"""Static INT8 quantization of SCRFD detection models.

Uses LFW deep-funneled images as calibration data — diverse faces in varied
environments give good activation-range coverage without clipping outliers.

Usage:
    uv run --python 3.13 python scripts/quantize_scrfd.py \
        --model models/det_2.5g.onnx \
        --lfw-dir data/lfw_tmp/lfw-deepfunneled/lfw-deepfunneled \
        --num-calibration 100

Produces: models/det_2.5g.int8.onnx
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

INPUT_H, INPUT_W = 640, 640


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Replicate SCRFDDetector._preprocess (aspect-ratio resize + pad + norm)."""
    orig_h, orig_w = img_bgr.shape[:2]
    im_ratio = orig_h / orig_w
    model_ratio = INPUT_H / INPUT_W

    if im_ratio > model_ratio:
        new_h = INPUT_H
        new_w = int(new_h / im_ratio)
    else:
        new_w = INPUT_W
        new_h = int(new_w * im_ratio)

    resized = cv2.resize(img_bgr, (new_w, new_h))
    det_img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = resized

    blob: np.ndarray = cv2.dnn.blobFromImage(
        det_img,
        scalefactor=1.0 / 128.0,
        size=(INPUT_W, INPUT_H),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
    )
    return blob


def collect_lfw_images(lfw_dir: Path, num_images: int, seed: int = 42) -> list[Path]:
    """Sample LFW images for calibration."""
    all_images = sorted(lfw_dir.rglob("*.jpg"))
    if not all_images:
        log.error("No .jpg images found in %s", lfw_dir)
        sys.exit(1)
    log.info("Found %d LFW images, sampling %d for calibration", len(all_images), num_images)
    rng = random.Random(seed)
    return rng.sample(all_images, min(num_images, len(all_images)))


class SCRFDCalibrationReader:
    """CalibrationDataReader that feeds preprocessed LFW images."""

    def __init__(self, image_paths: list[Path], input_name: str) -> None:
        self._input_name = input_name
        self._blobs: list[np.ndarray] = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is not None:
                self._blobs.append(preprocess_image(img))
        self._iter = iter(self._blobs)
        log.info("Prepared %d calibration samples", len(self._blobs))

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return {self._input_name: next(self._iter)}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = iter(self._blobs)


# ---------------------------------------------------------------------------
# Quantization pipeline
# ---------------------------------------------------------------------------


def get_input_name(model_path: Path) -> str:
    """Read the input tensor name from the ONNX model."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    return sess.get_inputs()[0].name


def run_quantization(
    model_path: Path,
    output_path: Path,
    calibration_reader: SCRFDCalibrationReader,
) -> None:
    """Run quant_pre_process + quantize_static."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.preprocess import quant_pre_process

    # Step 1: Preprocess — shape inference, BN fusion, graph cleanup.
    # This is critical to avoid the duplicate node name bug (ORT #8811).
    preprocessed_path = model_path.with_name(f"{model_path.stem}.preproc.onnx")
    log.info("Preprocessing model for quantization...")
    t0 = time.perf_counter()
    try:
        quant_pre_process(str(model_path), str(preprocessed_path), auto_merge=True)
    except Exception as exc:
        log.warning("Symbolic shape inference failed (%s), retrying without it...", exc)
        quant_pre_process(str(model_path), str(preprocessed_path), skip_symbolic_shape=True)
    log.info("Preprocessing done in %.1fs → %s", time.perf_counter() - t0, preprocessed_path.name)

    # Step 1b: Upgrade opset to ≥13 if needed (per-channel QDQ requires axis attr).
    import onnx

    model = onnx.load(str(preprocessed_path))
    model_opset = model.opset_import[0].version
    if model_opset < 13:
        log.info("Upgrading opset %d → 13 (required for per-channel QDQ)", model_opset)
        model = onnx.version_converter.convert_version(model, 13)
        onnx.save(model, str(preprocessed_path))
    del model

    # Step 2: Static quantization with QDQ format.
    log.info("Running static INT8 quantization (QDQ, MinMax calibration)...")
    t0 = time.perf_counter()
    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    quant_ms = (time.perf_counter() - t0) * 1000.0
    log.info("Quantization done in %.1fs → %s", quant_ms / 1000.0, output_path.name)

    # Cleanup intermediate file.
    preprocessed_path.unlink(missing_ok=True)

    # Report sizes.
    orig_mb = model_path.stat().st_size / (1024 * 1024)
    quant_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(
        "Size: %.2f MiB → %.2f MiB (%.1f%% reduction)",
        orig_mb,
        quant_mb,
        (1 - quant_mb / orig_mb) * 100,
    )


def verify_model(model_path: Path) -> bool:
    """Verify quantized model loads and runs."""
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        dummy = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
        outputs = sess.run(None, {input_name: dummy})
        log.info("Verification OK: %d outputs, model loads and runs", len(outputs))
        return True
    except Exception:
        log.exception("Verification FAILED — quantized model is broken")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Static INT8 quantization of SCRFD models")
    parser.add_argument("--model", type=Path, required=True, help="Path to FP32 SCRFD ONNX model")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: <model>.int8.onnx)")
    parser.add_argument(
        "--lfw-dir",
        type=Path,
        default=Path("data/lfw_tmp/lfw-deepfunneled/lfw-deepfunneled"),
        help="Path to LFW deep-funneled images",
    )
    parser.add_argument("--num-calibration", type=int, default=100, help="Number of calibration images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image sampling")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing quantized model")
    args = parser.parse_args()

    if not args.model.exists():
        log.error("Model not found: %s", args.model)
        sys.exit(1)

    output_path = args.output or args.model.with_name(f"{args.model.stem}.int8.onnx")
    if output_path.exists() and not args.overwrite:
        log.info("Quantized model already exists: %s (use --overwrite to replace)", output_path)
        sys.exit(0)

    # Collect calibration images.
    image_paths = collect_lfw_images(args.lfw_dir, args.num_calibration, args.seed)

    # Read input name from model.
    input_name = get_input_name(args.model)
    log.info("Model input name: %s", input_name)

    # Build calibration reader.
    reader = SCRFDCalibrationReader(image_paths, input_name)

    # Run quantization.
    run_quantization(args.model, output_path, reader)

    # Verify.
    if not verify_model(output_path):
        log.error("Quantized model verification failed!")
        sys.exit(1)

    log.info("Done: %s", output_path)


if __name__ == "__main__":
    main()
