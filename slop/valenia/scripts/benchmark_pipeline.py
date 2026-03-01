#!/usr/bin/env python3
"""Benchmark SCRFD + ArcFace baseline on image or Pi camera."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from contracts import PipelineLike
from pipeline_factory import PipelineSpec, build_face_pipeline, parse_size, resolve_project_path
from runtime_utils import (
    Float32Array,
    MemoryStats,
    QuantizationReport,
    UInt8Array,
    enforce_memory_cap,
    quantize_onnx_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-model", default="models/buffalo_sc/det_500m.onnx")
    parser.add_argument("--rec-model", default="models/buffalo_sc/w600k_mbf.onnx")
    parser.add_argument("--det-size", default="320x320")
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--max-faces", type=int, default=3)
    parser.add_argument("--det-every", type=int, default=3)
    parser.add_argument("--mode", choices=["image", "camera"], default="image")
    parser.add_argument("--image", help="Input image path when mode=image")
    parser.add_argument("--runs", type=int, default=50, help="Iterations in image mode")
    parser.add_argument("--frames", type=int, default=300, help="Frames in camera mode")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--save-output", default="", help="Optional annotated output file")
    parser.add_argument(
        "--ram-cap-mb",
        type=float,
        default=4096.0,
        help="Abort if current or peak RSS exceeds this limit; <=0 disables the check",
    )
    parser.add_argument(
        "--quantize-rec",
        action="store_true",
        help="Generate or reuse an INT8 dynamic-quantized ArcFace model before benchmarking",
    )
    parser.add_argument(
        "--quantized-rec-model",
        default="",
        help="Optional output path for the generated quantized ArcFace model",
    )
    parser.add_argument(
        "--force-requantize",
        action="store_true",
        help="Rebuild the quantized ArcFace model even if the output file already exists",
    )
    return parser.parse_args()


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


def print_summary(
    loop_ms: list[float],
    det_ms: list[float],
    emb_ms: list[float],
    face_counts: list[int],
    startup_memory: MemoryStats,
    post_init_memory: MemoryStats,
    final_memory: MemoryStats,
    quantization: QuantizationReport | None,
) -> None:
    loop_avg = statistics.fmean(loop_ms)
    fps = 1000.0 / loop_avg if loop_avg > 0 else 0.0
    det_avg = statistics.fmean(det_ms) if det_ms else 0.0
    emb_avg = statistics.fmean(emb_ms) if emb_ms else 0.0
    faces_avg = statistics.fmean(face_counts) if face_counts else 0.0
    print("\n=== Benchmark summary ===")
    print(f"frames: {len(loop_ms)}")
    print(f"avg loop latency: {loop_avg:.2f} ms")
    print(f"avg FPS: {fps:.2f}")
    print(f"avg detection latency (when run): {det_avg:.2f} ms")
    print(f"avg embedding latency total/frame: {emb_avg:.2f} ms")
    print(f"avg faces detected/frame: {faces_avg:.2f}")
    print(
        "memory RSS (startup -> post-init -> final current): "
        f"{startup_memory.current_rss_mb:.2f} -> {post_init_memory.current_rss_mb:.2f} "
        f"-> {final_memory.current_rss_mb:.2f} MiB"
    )
    print(f"peak RSS: {final_memory.peak_rss_mb:.2f} MiB")
    print(
        "model init RSS delta: "
        f"{post_init_memory.current_rss_mb - startup_memory.current_rss_mb:.2f} MiB"
    )
    if quantization is not None:
        quant_status = "reused existing file" if quantization.reused_existing else "generated"
        print(
            "quantized ArcFace model: "
            f"{quantization.quantized_model_path} ({quant_status}, "
            f"{quantization.original_size_mb:.2f} -> {quantization.quantized_size_mb:.2f} MiB, "
            f"delta {quantization.size_delta_mb:.2f} MiB)"
        )
        print(f"quantization time: {quantization.quantize_ms:.2f} ms")


def run_image_mode(
    args: argparse.Namespace,
    pipeline: PipelineLike,
    startup_memory: MemoryStats,
    post_init_memory: MemoryStats,
    quantization: QuantizationReport | None,
) -> int:
    if not args.image:
        raise ValueError("--image is required in image mode")
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    image = np.asarray(image, dtype=np.uint8)

    loop_ms: list[float] = []
    det_ms: list[float] = []
    emb_ms: list[float] = []
    face_counts: list[int] = []
    last_boxes = np.empty((0, 5), dtype=np.float32)
    last_kps: Float32Array | None = None

    for idx in range(args.runs):
        t_loop0 = time.perf_counter()
        if idx % args.det_every == 0:
            frame = pipeline.process_frame(image, max_faces=args.max_faces)
            last_boxes, last_kps = frame.boxes, frame.kps
            det_ms.append(frame.detect_ms)
            emb_ms.append(frame.embed_ms_total)
        else:
            emb_ms.append(0.0)
        t_loop1 = time.perf_counter()
        loop_ms.append((t_loop1 - t_loop0) * 1000.0)
        face_counts.append(len(last_boxes))
        if (idx + 1) % 25 == 0 or idx + 1 == args.runs:
            enforce_memory_cap(args.ram_cap_mb, f"image benchmark iteration {idx + 1}")

    final_memory = enforce_memory_cap(args.ram_cap_mb, "image benchmark completion")
    print_summary(
        loop_ms,
        det_ms,
        emb_ms,
        face_counts,
        startup_memory=startup_memory,
        post_init_memory=post_init_memory,
        final_memory=final_memory,
        quantization=quantization,
    )
    if args.save_output:
        annotated = annotate(image, last_boxes, last_kps)
        cv2.imwrite(args.save_output, annotated)
        print(f"Saved annotation: {args.save_output}")
    return 0


def run_camera_mode(
    args: argparse.Namespace,
    pipeline: PipelineLike,
    startup_memory: MemoryStats,
    post_init_memory: MemoryStats,
    quantization: QuantizationReport | None,
) -> int:
    from picamera2 import Picamera2

    try:
        picam2 = Picamera2()
    except IndexError:
        print("No camera detected by Picamera2. Check cable/enablement and rerun.")
        return 3

    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # Warm up camera for stable timings.
    time.sleep(1.0)

    loop_ms: list[float] = []
    det_ms: list[float] = []
    emb_ms: list[float] = []
    face_counts: list[int] = []
    last_boxes = np.empty((0, 5), dtype=np.float32)
    last_kps: Float32Array | None = None
    last_frame_bgr: UInt8Array | None = None

    try:
        for idx in range(args.frames):
            t_loop0 = time.perf_counter()
            frame_rgb = np.asarray(picam2.capture_array(), dtype=np.uint8)
            frame_bgr = np.asarray(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
            last_frame_bgr = frame_bgr

            if idx % args.det_every == 0:
                frame = pipeline.process_frame(frame_bgr, max_faces=args.max_faces)
                last_boxes, last_kps = frame.boxes, frame.kps
                det_ms.append(frame.detect_ms)
                emb_ms.append(frame.embed_ms_total)
            else:
                emb_ms.append(0.0)

            t_loop1 = time.perf_counter()
            loop_ms.append((t_loop1 - t_loop0) * 1000.0)
            face_counts.append(len(last_boxes))
            if (idx + 1) % 25 == 0 or idx + 1 == args.frames:
                enforce_memory_cap(args.ram_cap_mb, f"camera benchmark iteration {idx + 1}")

    finally:
        picam2.stop()

    final_memory = enforce_memory_cap(args.ram_cap_mb, "camera benchmark completion")
    print_summary(
        loop_ms,
        det_ms,
        emb_ms,
        face_counts,
        startup_memory=startup_memory,
        post_init_memory=post_init_memory,
        final_memory=final_memory,
        quantization=quantization,
    )
    if args.save_output and last_frame_bgr is not None:
        annotated = annotate(last_frame_bgr, last_boxes, last_kps)
        cv2.imwrite(args.save_output, annotated)
        print(f"Saved annotation: {args.save_output}")
    return 0


def main() -> int:
    try:
        args = parse_args()
        startup_memory = enforce_memory_cap(args.ram_cap_mb, "startup")
        det_model = resolve_project_path(ROOT, args.det_model)
        rec_model = resolve_project_path(ROOT, args.rec_model)

        if not det_model.exists() or not rec_model.exists():
            print("Missing model files. Run: ./scripts/download_models.sh")
            return 2

        quantization: QuantizationReport | None = None
        if args.quantize_rec or args.quantized_rec_model:
            quantized_rec_path = (
                resolve_project_path(ROOT, args.quantized_rec_model)
                if args.quantized_rec_model
                else None
            )
            try:
                rec_model, quantization = quantize_onnx_model(
                    rec_model,
                    quantized_path=quantized_rec_path,
                    overwrite=args.force_requantize,
                )
            except RuntimeError as exc:
                print(exc)
                return 5

        pipeline = build_face_pipeline(
            PipelineSpec(
                det_model=det_model,
                rec_model=rec_model,
                det_size=parse_size(args.det_size),
                det_thresh=args.det_thresh,
                max_faces=args.max_faces,
            )
        )
        post_init_memory = enforce_memory_cap(args.ram_cap_mb, "pipeline initialization")

        if args.mode == "image":
            return run_image_mode(
                args,
                pipeline,
                startup_memory=startup_memory,
                post_init_memory=post_init_memory,
                quantization=quantization,
            )
        return run_camera_mode(
            args,
            pipeline,
            startup_memory=startup_memory,
            post_init_memory=post_init_memory,
            quantization=quantization,
        )
    except MemoryError as exc:
        print(exc)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
