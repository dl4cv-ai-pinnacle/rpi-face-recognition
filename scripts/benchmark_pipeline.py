#!/usr/bin/env python3
"""Benchmark the unified face-recognition pipeline on image or Pi camera.

Ported from slop/valenia/scripts/benchmark_pipeline.py to use the unified
pipeline (src.pipeline.build_pipeline) and YAML config (src.config.load_config).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.contracts import Float32Array, UInt8Array
from src.metrics import MemoryStats, enforce_memory_cap
from src.pipeline import FacePipeline, build_pipeline
from src.quantization import QuantizationReport, quantize_onnx_model

# ---------------------------------------------------------------------------
# Config override helpers
# ---------------------------------------------------------------------------


def _auto_cast(value: str) -> object:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_overrides(raw: dict[str, object], overrides: list[str]) -> None:
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        target = raw
        for part in parts[:-1]:
            target = target[part]  # type: ignore[assignment]
        target[parts[-1]] = _auto_cast(value)  # type: ignore[index]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the unified face-recognition pipeline."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Dotted config override, repeatable. "
            "Examples: live.det_every=5, embedding.quantize_int8=true"
        ),
    )
    parser.add_argument("--mode", choices=["image", "camera"], default="image")
    parser.add_argument("--image", help="Input image path when mode=image")
    parser.add_argument("--runs", type=int, default=50, help="Iterations in image mode")
    parser.add_argument("--frames", type=int, default=300, help="Frames in camera mode")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help=(
            "Unmeasured warmup iterations/frames before collecting timings; "
            "values <0 are treated as 0"
        ),
    )
    parser.add_argument("--save-output", default="", help="Optional annotated output file")
    parser.add_argument("--output-json", default="", help="Optional JSON report path")
    parser.add_argument(
        "--ram-cap-mb",
        type=float,
        default=4096.0,
        help="Abort if current or peak RSS exceeds this limit; <=0 disables the check",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Annotate helper
# ---------------------------------------------------------------------------


def annotate(frame: UInt8Array, boxes: Float32Array, kps: Float32Array | None) -> UInt8Array:
    out = frame.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2, score = box
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        cv2.putText(
            out,
            f"{score:.2f}",
            (int(x1), max(0, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        if kps is not None:
            for point in kps[idx]:
                cv2.circle(out, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    return np.asarray(out, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkMemory:
    startup_current_rss: float
    post_init_current_rss: float
    final_current_rss: float
    final_peak_rss: float
    model_init_delta_rss: float


@dataclass(frozen=True)
class BenchmarkQuantization:
    enabled: bool
    model_path: str
    reused_existing: bool
    quantize_ms: float
    original_size_mb: float
    quantized_size_mb: float
    size_delta_mb: float


@dataclass
class BenchmarkSummary:
    frames_measured: int
    avg_loop_ms: float
    avg_fps: float
    avg_detect_ms_when_run: float
    avg_embed_ms_per_frame: float
    avg_faces_per_frame: float
    memory_mb: BenchmarkMemory
    quantization: BenchmarkQuantization
    config: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_summary(
    loop_ms: list[float],
    det_ms: list[float],
    emb_ms: list[float],
    face_counts: list[int],
    startup_memory: MemoryStats,
    post_init_memory: MemoryStats,
    final_memory: MemoryStats,
    quantization: QuantizationReport | None,
) -> BenchmarkSummary:
    loop_avg = statistics.fmean(loop_ms)
    fps = 1000.0 / loop_avg if loop_avg > 0 else 0.0
    det_avg = statistics.fmean(det_ms) if det_ms else 0.0
    emb_avg = statistics.fmean(emb_ms) if emb_ms else 0.0
    faces_avg = statistics.fmean(face_counts) if face_counts else 0.0
    return BenchmarkSummary(
        frames_measured=len(loop_ms),
        avg_loop_ms=loop_avg,
        avg_fps=fps,
        avg_detect_ms_when_run=det_avg,
        avg_embed_ms_per_frame=emb_avg,
        avg_faces_per_frame=faces_avg,
        memory_mb=BenchmarkMemory(
            startup_current_rss=startup_memory.current_rss_mb,
            post_init_current_rss=post_init_memory.current_rss_mb,
            final_current_rss=final_memory.current_rss_mb,
            final_peak_rss=final_memory.peak_rss_mb,
            model_init_delta_rss=(post_init_memory.current_rss_mb - startup_memory.current_rss_mb),
        ),
        quantization=BenchmarkQuantization(
            enabled=bool(quantization is not None),
            model_path=quantization.quantized_model_path if quantization is not None else "",
            reused_existing=bool(quantization.reused_existing)
            if quantization is not None
            else False,
            quantize_ms=float(quantization.quantize_ms) if quantization else 0.0,
            original_size_mb=float(quantization.original_size_mb) if quantization else 0.0,
            quantized_size_mb=float(quantization.quantized_size_mb) if quantization else 0.0,
            size_delta_mb=float(quantization.size_delta_mb) if quantization else 0.0,
        ),
    )


def print_summary(summary: BenchmarkSummary) -> None:
    memory = summary.memory_mb
    quantization = summary.quantization
    print("\n=== Benchmark summary ===")
    print(f"frames: {summary.frames_measured}")
    print(f"avg loop latency: {summary.avg_loop_ms:.2f} ms")
    print(f"avg FPS: {summary.avg_fps:.2f}")
    print(f"avg detection latency (when run): {summary.avg_detect_ms_when_run:.2f} ms")
    print(f"avg embedding latency total/frame: {summary.avg_embed_ms_per_frame:.2f} ms")
    print(f"avg faces detected/frame: {summary.avg_faces_per_frame:.2f}")
    print(
        "memory RSS (startup -> post-init -> final current): "
        f"{memory.startup_current_rss:.2f} -> {memory.post_init_current_rss:.2f} "
        f"-> {memory.final_current_rss:.2f} MiB"
    )
    print(f"peak RSS: {memory.final_peak_rss:.2f} MiB")
    print(f"model init RSS delta: {memory.model_init_delta_rss:.2f} MiB")
    if quantization.enabled:
        quant_status = "reused existing file" if quantization.reused_existing else "generated"
        print(
            "quantized ArcFace model: "
            f"{quantization.model_path} ({quant_status}, "
            f"{quantization.original_size_mb:.2f} -> "
            f"{quantization.quantized_size_mb:.2f} MiB, "
            f"delta {quantization.size_delta_mb:.2f} MiB)"
        )
        print(f"quantization time: {quantization.quantize_ms:.2f} ms")


# ---------------------------------------------------------------------------
# Benchmark modes
# ---------------------------------------------------------------------------


def run_image_mode(
    args: argparse.Namespace,
    pipeline: FacePipeline,
    det_every: int,
    max_faces: int,
    startup_memory: MemoryStats,
    post_init_memory: MemoryStats,
    quantization: QuantizationReport | None,
    config_snapshot: dict[str, object],
) -> int:
    if not args.image:
        raise ValueError("--image is required in image mode")
    image_path = ROOT / args.image
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image = np.asarray(image, dtype=np.uint8)

    warmup_runs = max(0, args.warmup_runs)
    loop_ms: list[float] = []
    det_ms: list[float] = []
    emb_ms: list[float] = []
    face_counts: list[int] = []
    last_boxes = np.empty((0, 5), dtype=np.float32)
    last_kps: Float32Array | None = None

    total_runs = warmup_runs + max(1, args.runs)
    for idx in range(total_runs):
        t_loop0 = time.perf_counter()
        if idx % det_every == 0:
            frame = pipeline.process_frame(image, max_faces=max_faces)
            last_boxes, last_kps = frame.boxes, frame.kps
            current_det_ms = frame.detect_ms
            current_emb_ms = frame.embed_ms_total
        else:
            current_det_ms = None
            current_emb_ms = 0.0
        t_loop1 = time.perf_counter()
        if idx >= warmup_runs:
            loop_ms.append((t_loop1 - t_loop0) * 1000.0)
            if current_det_ms is not None:
                det_ms.append(current_det_ms)
            emb_ms.append(current_emb_ms)
            face_counts.append(len(last_boxes))
            measured_idx = idx - warmup_runs + 1
            if measured_idx % 25 == 0 or measured_idx == args.runs:
                enforce_memory_cap(args.ram_cap_mb, f"image benchmark iteration {measured_idx}")

    final_memory = enforce_memory_cap(args.ram_cap_mb, "image benchmark completion")
    summary = build_summary(
        loop_ms,
        det_ms,
        emb_ms,
        face_counts,
        startup_memory=startup_memory,
        post_init_memory=post_init_memory,
        final_memory=final_memory,
        quantization=quantization,
    )
    summary.config = {
        "mode": "image",
        "image": args.image,
        "runs": int(args.runs),
        "warmup_runs": warmup_runs,
        "det_every": det_every,
        "max_faces": max_faces,
        "ram_cap_mb": float(args.ram_cap_mb),
        **config_snapshot,
    }
    print_summary(summary)
    if args.save_output:
        save_output_path = ROOT / args.save_output
        save_output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = annotate(image, last_boxes, last_kps)
        cv2.imwrite(str(save_output_path), annotated)
        print(f"Saved annotation: {save_output_path}")
    if args.output_json:
        out_path = ROOT / args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")
        print(f"Saved JSON report: {out_path}")
    return 0


def run_camera_mode(
    args: argparse.Namespace,
    pipeline: FacePipeline,
    det_every: int,
    max_faces: int,
    width: int,
    height: int,
    startup_memory: MemoryStats,
    post_init_memory: MemoryStats,
    quantization: QuantizationReport | None,
    config_snapshot: dict[str, object],
) -> int:
    from picamera2 import Picamera2  # type: ignore[import-untyped]

    try:
        picam2 = Picamera2()
    except IndexError:
        print("No camera detected by Picamera2. Check cable/enablement and rerun.")
        return 3

    cam_config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(cam_config)
    picam2.start()

    # Warm up camera for stable timings.
    time.sleep(1.0)

    warmup_runs = max(0, args.warmup_runs)
    loop_ms: list[float] = []
    det_ms: list[float] = []
    emb_ms: list[float] = []
    face_counts: list[int] = []
    last_boxes = np.empty((0, 5), dtype=np.float32)
    last_kps: Float32Array | None = None
    last_frame_bgr: UInt8Array | None = None

    measured_frames = max(1, args.frames)
    total_frames = warmup_runs + measured_frames
    try:
        for idx in range(total_frames):
            t_loop0 = time.perf_counter()
            frame_rgb = np.asarray(picam2.capture_array(), dtype=np.uint8)
            frame_bgr = np.asarray(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
            last_frame_bgr = frame_bgr

            if idx % det_every == 0:
                frame = pipeline.process_frame(frame_bgr, max_faces=max_faces)
                last_boxes, last_kps = frame.boxes, frame.kps
                current_det_ms = frame.detect_ms
                current_emb_ms = frame.embed_ms_total
            else:
                current_det_ms = None
                current_emb_ms = 0.0

            t_loop1 = time.perf_counter()
            if idx >= warmup_runs:
                loop_ms.append((t_loop1 - t_loop0) * 1000.0)
                if current_det_ms is not None:
                    det_ms.append(current_det_ms)
                emb_ms.append(current_emb_ms)
                face_counts.append(len(last_boxes))
                measured_idx = idx - warmup_runs + 1
                if measured_idx % 25 == 0 or measured_idx == measured_frames:
                    enforce_memory_cap(
                        args.ram_cap_mb, f"camera benchmark iteration {measured_idx}"
                    )

    finally:
        picam2.stop()

    final_memory = enforce_memory_cap(args.ram_cap_mb, "camera benchmark completion")
    summary = build_summary(
        loop_ms,
        det_ms,
        emb_ms,
        face_counts,
        startup_memory=startup_memory,
        post_init_memory=post_init_memory,
        final_memory=final_memory,
        quantization=quantization,
    )
    summary.config = {
        "mode": "camera",
        "frames": measured_frames,
        "warmup_runs": warmup_runs,
        "det_every": det_every,
        "max_faces": max_faces,
        "width": width,
        "height": height,
        "ram_cap_mb": float(args.ram_cap_mb),
        **config_snapshot,
    }
    print_summary(summary)
    if args.save_output and last_frame_bgr is not None:
        save_output_path = ROOT / args.save_output
        save_output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = annotate(last_frame_bgr, last_boxes, last_kps)
        cv2.imwrite(str(save_output_path), annotated)
        print(f"Saved annotation: {save_output_path}")
    if args.output_json:
        out_path = ROOT / args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")
        print(f"Saved JSON report: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        args = parse_args()
        startup_memory = enforce_memory_cap(args.ram_cap_mb, "startup")

        # Load YAML, apply CLI overrides, then build typed config.
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _apply_overrides(raw, args.override)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(raw, tmp)
            tmp_path = Path(tmp.name)
        try:
            config = load_config(tmp_path)
        finally:
            tmp_path.unlink()

        # Optional INT8 quantization (driven by config.embedding.quantize_int8).
        quantization: QuantizationReport | None = None
        embedding_model = ROOT / config.embedding.model_path
        if not embedding_model.exists():
            print(f"Missing embedding model: {embedding_model}")
            print("Run: ./scripts/download_models.sh")
            return 2

        if config.embedding.quantize_int8:
            try:
                _, quantization = quantize_onnx_model(embedding_model)
            except RuntimeError as exc:
                print(exc)
                return 5

        # Build unified pipeline from config.
        pipeline = build_pipeline(config)
        post_init_memory = enforce_memory_cap(args.ram_cap_mb, "pipeline initialization")

        # Extract live parameters from config.
        det_every = max(1, config.live.det_every)
        max_faces = config.live.max_faces
        width, height = config.capture.resolution

        # Snapshot of relevant config values for the report.
        config_snapshot: dict[str, object] = {
            "config_file": str(config_path),
            "detection_backend": config.detection.backend,
            "detection_confidence_threshold": config.detection.confidence_threshold,
            "alignment_method": config.alignment.method,
            "embedding_model_path": config.embedding.model_path,
            "embedding_quantize_int8": config.embedding.quantize_int8,
        }

        if args.mode == "image":
            return run_image_mode(
                args,
                pipeline,
                det_every=det_every,
                max_faces=max_faces,
                startup_memory=startup_memory,
                post_init_memory=post_init_memory,
                quantization=quantization,
                config_snapshot=config_snapshot,
            )
        return run_camera_mode(
            args,
            pipeline,
            det_every=det_every,
            max_faces=max_faces,
            width=width,
            height=height,
            startup_memory=startup_memory,
            post_init_memory=post_init_memory,
            quantization=quantization,
            config_snapshot=config_snapshot,
        )
    except MemoryError as exc:
        print(exc)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
