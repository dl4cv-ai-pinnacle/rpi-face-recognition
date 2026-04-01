#!/usr/bin/env python3
"""Unified benchmark CLI for the face-recognition pipeline.

Subcommands:
    pipeline  Single-image or Pi camera latency + memory
    video     Video-level tracking quality + identity metrics (FANVID or raw video)
    lfw       LFW verification accuracy with 10-fold cross-validation
    compare   A/B comparison of two pipeline configurations

Usage examples:
    # Pipeline latency on a single image
    uv run --python 3.13 python scripts/benchmark.py pipeline \
        --config config.yaml --image data/test_frame.jpg --runs 50

    # Video benchmark on FANVID dataset
    uv run --python 3.13 python scripts/benchmark.py video \
        --config config.yaml --fanvid-dir data/fanvid

    # LFW verification
    uv run --python 3.13 python scripts/benchmark.py lfw \
        --config config.yaml --lfw-dir data/lfw/lfw_funneled \
        --train-pairs data/lfw/pairsDevTrain.txt --test-pairs data/lfw/pairsDevTest.txt

    # Compare FP32 vs INT8
    uv run --python 3.13 python scripts/benchmark.py compare \
        --config-a config.yaml --override-a embedding.quantize_int8=false --label-a fp32 \
        --config-b config.yaml --override-b embedding.quantize_int8=true --label-b int8
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking.config import load_config_with_overrides  # noqa: E402
from src.benchmarking.ground_truth import (  # noqa: E402
    GTFrame,
    VideoClip,
    auto_enroll_from_clips,
    enroll_gallery_from_dir,
    load_fanvid_gt,
    load_gt_json,
)
from src.benchmarking.video_metrics import (  # noqa: E402
    TrackRecord,
    compute_temporal_metrics,
    compute_throughput,
    match_tracks_to_gt,
)
from src.config import AppConfig  # noqa: E402
from src.contracts import Float32Array, UInt8Array  # noqa: E402
from src.gallery import GalleryStore  # noqa: E402
from src.live import LiveRuntime  # noqa: E402
from src.metrics import MemoryStats, enforce_memory_cap, get_memory_stats  # noqa: E402
from src.pipeline import FacePipeline, build_pipeline  # noqa: E402
from src.quantization import QuantizationReport, quantize_onnx_model  # noqa: E402

# ===================================================================
# Common argparse parent
# ===================================================================

_COMMON_PARSER = argparse.ArgumentParser(add_help=False)
_COMMON_PARSER.add_argument("--config", default="config.yaml", help="Pipeline config YAML path")
_COMMON_PARSER.add_argument(
    "--override",
    action="append",
    default=[],
    help="Dotted config override, repeatable. E.g. live.det_every=5",
)
_COMMON_PARSER.add_argument("--output-json", default="", help="JSON output path")
_COMMON_PARSER.add_argument(
    "--ram-cap-mb",
    type=float,
    default=4096.0,
    help="Abort if RSS exceeds this limit (MiB); <=0 disables",
)


def _save_json(data: object, path_str: str) -> None:
    if not path_str:
        return
    out = Path(path_str) if Path(path_str).is_absolute() else ROOT / path_str
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report: {out}")


# ===================================================================
# Subcommand: pipeline
# ===================================================================


@dataclass(frozen=True)
class _PipelineMemory:
    startup_current_rss: float
    post_init_current_rss: float
    final_current_rss: float
    final_peak_rss: float
    model_init_delta_rss: float


@dataclass(frozen=True)
class _PipelineQuantization:
    enabled: bool
    model_path: str
    reused_existing: bool
    quantize_ms: float
    original_size_mb: float
    quantized_size_mb: float
    size_delta_mb: float


@dataclass
class _PipelineSummary:
    frames_measured: int
    avg_loop_ms: float
    avg_fps: float
    avg_detect_ms_when_run: float
    avg_embed_ms_per_frame: float
    avg_faces_per_frame: float
    memory_mb: _PipelineMemory
    quantization: _PipelineQuantization
    config: dict[str, object] = field(default_factory=dict)


def _build_pipeline_summary(
    loop_ms: list[float],
    det_ms: list[float],
    emb_ms: list[float],
    face_counts: list[int],
    startup_memory: MemoryStats,
    post_init_memory: MemoryStats,
    final_memory: MemoryStats,
    quantization: QuantizationReport | None,
) -> _PipelineSummary:
    loop_avg = statistics.fmean(loop_ms)
    fps = 1000.0 / loop_avg if loop_avg > 0 else 0.0
    return _PipelineSummary(
        frames_measured=len(loop_ms),
        avg_loop_ms=loop_avg,
        avg_fps=fps,
        avg_detect_ms_when_run=statistics.fmean(det_ms) if det_ms else 0.0,
        avg_embed_ms_per_frame=statistics.fmean(emb_ms) if emb_ms else 0.0,
        avg_faces_per_frame=statistics.fmean(face_counts) if face_counts else 0.0,
        memory_mb=_PipelineMemory(
            startup_current_rss=startup_memory.current_rss_mb,
            post_init_current_rss=post_init_memory.current_rss_mb,
            final_current_rss=final_memory.current_rss_mb,
            final_peak_rss=final_memory.peak_rss_mb,
            model_init_delta_rss=post_init_memory.current_rss_mb - startup_memory.current_rss_mb,
        ),
        quantization=_PipelineQuantization(
            enabled=quantization is not None,
            model_path=quantization.quantized_model_path if quantization else "",
            reused_existing=bool(quantization.reused_existing) if quantization else False,
            quantize_ms=float(quantization.quantize_ms) if quantization else 0.0,
            original_size_mb=float(quantization.original_size_mb) if quantization else 0.0,
            quantized_size_mb=float(quantization.quantized_size_mb) if quantization else 0.0,
            size_delta_mb=float(quantization.size_delta_mb) if quantization else 0.0,
        ),
    )


def _print_pipeline_summary(s: _PipelineSummary) -> None:
    m = s.memory_mb
    q = s.quantization
    print("\n=== Benchmark summary ===")
    print(f"frames: {s.frames_measured}")
    print(f"avg loop latency: {s.avg_loop_ms:.2f} ms")
    print(f"avg FPS: {s.avg_fps:.2f}")
    print(f"avg detection latency (when run): {s.avg_detect_ms_when_run:.2f} ms")
    print(f"avg embedding latency total/frame: {s.avg_embed_ms_per_frame:.2f} ms")
    print(f"avg faces detected/frame: {s.avg_faces_per_frame:.2f}")
    print(
        f"memory RSS (startup -> post-init -> final current): "
        f"{m.startup_current_rss:.2f} -> {m.post_init_current_rss:.2f} "
        f"-> {m.final_current_rss:.2f} MiB"
    )
    print(f"peak RSS: {m.final_peak_rss:.2f} MiB")
    print(f"model init RSS delta: {m.model_init_delta_rss:.2f} MiB")
    if q.enabled:
        status = "reused existing file" if q.reused_existing else "generated"
        print(
            f"quantized ArcFace model: {q.model_path} ({status}, "
            f"{q.original_size_mb:.2f} -> {q.quantized_size_mb:.2f} MiB, "
            f"delta {q.size_delta_mb:.2f} MiB)"
        )
        print(f"quantization time: {q.quantize_ms:.2f} ms")


def _annotate_frame(frame: UInt8Array, boxes: Float32Array, kps: Float32Array | None) -> UInt8Array:
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


def _run_pipeline_image(
    pipeline: FacePipeline,
    image: UInt8Array,
    runs: int,
    warmup_runs: int,
    det_every: int,
    max_faces: int,
    ram_cap_mb: float,
) -> tuple[list[float], list[float], list[float], list[int], Float32Array, Float32Array | None]:
    loop_ms: list[float] = []
    det_ms: list[float] = []
    emb_ms: list[float] = []
    face_counts: list[int] = []
    last_boxes = np.empty((0, 5), dtype=np.float32)
    last_kps: Float32Array | None = None

    for idx in range(warmup_runs + max(1, runs)):
        t0 = time.perf_counter()
        if idx % det_every == 0:
            frame = pipeline.process_frame(image, max_faces=max_faces)
            last_boxes, last_kps = frame.boxes, frame.kps
            current_det_ms: float | None = frame.detect_ms
            current_emb_ms = frame.embed_ms_total
        else:
            current_det_ms = None
            current_emb_ms = 0.0
        t1 = time.perf_counter()
        if idx >= warmup_runs:
            loop_ms.append((t1 - t0) * 1000.0)
            if current_det_ms is not None:
                det_ms.append(current_det_ms)
            emb_ms.append(current_emb_ms)
            face_counts.append(len(last_boxes))
            measured_idx = idx - warmup_runs + 1
            if measured_idx % 25 == 0 or measured_idx == runs:
                enforce_memory_cap(ram_cap_mb, f"image benchmark iteration {measured_idx}")

    return loop_ms, det_ms, emb_ms, face_counts, last_boxes, last_kps


def _run_pipeline_camera(
    pipeline: FacePipeline,
    frames: int,
    warmup_runs: int,
    det_every: int,
    max_faces: int,
    width: int,
    height: int,
    ram_cap_mb: float,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[int],
    Float32Array,
    Float32Array | None,
    UInt8Array | None,
]:
    from picamera2 import Picamera2  # type: ignore[import-untyped]

    try:
        picam2 = Picamera2()
    except IndexError:
        print("No camera detected by Picamera2. Check cable/enablement and rerun.")
        sys.exit(3)

    cam_config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(cam_config)
    picam2.start()
    time.sleep(1.0)  # stabilise camera

    loop_ms: list[float] = []
    det_ms: list[float] = []
    emb_ms: list[float] = []
    face_counts: list[int] = []
    last_boxes = np.empty((0, 5), dtype=np.float32)
    last_kps: Float32Array | None = None
    last_frame_bgr: UInt8Array | None = None

    measured_frames = max(1, frames)
    try:
        for idx in range(warmup_runs + measured_frames):
            t0 = time.perf_counter()
            frame_rgb = np.asarray(picam2.capture_array(), dtype=np.uint8)
            frame_bgr = np.asarray(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
            last_frame_bgr = frame_bgr

            if idx % det_every == 0:
                frame = pipeline.process_frame(frame_bgr, max_faces=max_faces)
                last_boxes, last_kps = frame.boxes, frame.kps
                current_det_ms: float | None = frame.detect_ms
                current_emb_ms = frame.embed_ms_total
            else:
                current_det_ms = None
                current_emb_ms = 0.0

            t1 = time.perf_counter()
            if idx >= warmup_runs:
                loop_ms.append((t1 - t0) * 1000.0)
                if current_det_ms is not None:
                    det_ms.append(current_det_ms)
                emb_ms.append(current_emb_ms)
                face_counts.append(len(last_boxes))
                measured_idx = idx - warmup_runs + 1
                if measured_idx % 25 == 0 or measured_idx == measured_frames:
                    enforce_memory_cap(ram_cap_mb, f"camera benchmark iteration {measured_idx}")
    finally:
        picam2.stop()

    return loop_ms, det_ms, emb_ms, face_counts, last_boxes, last_kps, last_frame_bgr


def _add_pipeline_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("pipeline", parents=[_COMMON_PARSER], help="Single-image / camera")
    p.add_argument("--mode", choices=["image", "camera"], default="image")
    p.add_argument("--image", help="Input image path (required for image mode)")
    p.add_argument("--runs", type=int, default=50, help="Iterations in image mode")
    p.add_argument("--frames", type=int, default=300, help="Frames in camera mode")
    p.add_argument("--warmup-runs", type=int, default=5)
    p.add_argument("--save-output", default="", help="Save annotated image to this path")


def cmd_pipeline(args: argparse.Namespace) -> int:
    startup_memory = enforce_memory_cap(args.ram_cap_mb, "startup")
    config = load_config_with_overrides(args.config, args.override)

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

    pipeline = build_pipeline(config)
    post_init_memory = enforce_memory_cap(args.ram_cap_mb, "pipeline initialization")

    det_every = max(1, config.live.det_every)
    max_faces = config.live.max_faces
    warmup_runs = max(0, args.warmup_runs)

    config_snapshot: dict[str, object] = {
        "config_file": str(args.config),
        "detection_backend": config.detection.backend,
        "detection_confidence_threshold": config.detection.confidence_threshold,
        "alignment_method": config.alignment.method,
        "embedding_model_path": config.embedding.model_path,
        "embedding_quantize_int8": config.embedding.quantize_int8,
    }

    if args.mode == "image":
        if not args.image:
            print("--image is required in image mode")
            return 1
        image_path = ROOT / args.image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path}")
            return 2
        image = np.asarray(image, dtype=np.uint8)

        loop_ms, det_ms, emb_ms, face_counts, last_boxes, last_kps = _run_pipeline_image(
            pipeline,
            image,
            max(1, args.runs),
            warmup_runs,
            det_every,
            max_faces,
            args.ram_cap_mb,
        )
        final_memory = enforce_memory_cap(args.ram_cap_mb, "image benchmark completion")
        summary = _build_pipeline_summary(
            loop_ms,
            det_ms,
            emb_ms,
            face_counts,
            startup_memory,
            post_init_memory,
            final_memory,
            quantization,
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
        _print_pipeline_summary(summary)
        if args.save_output:
            out_path = ROOT / args.save_output
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), _annotate_frame(image, last_boxes, last_kps))
            print(f"Saved annotation: {out_path}")
        _save_json(asdict(summary), args.output_json)
    else:
        width, height = config.capture.resolution
        (
            loop_ms,
            det_ms,
            emb_ms,
            face_counts,
            last_boxes,
            last_kps,
            last_frame,
        ) = _run_pipeline_camera(
            pipeline,
            max(1, args.frames),
            warmup_runs,
            det_every,
            max_faces,
            width,
            height,
            args.ram_cap_mb,
        )
        final_memory = enforce_memory_cap(args.ram_cap_mb, "camera benchmark completion")
        summary = _build_pipeline_summary(
            loop_ms,
            det_ms,
            emb_ms,
            face_counts,
            startup_memory,
            post_init_memory,
            final_memory,
            quantization,
        )
        summary.config = {
            "mode": "camera",
            "frames": max(1, args.frames),
            "warmup_runs": warmup_runs,
            "det_every": det_every,
            "max_faces": max_faces,
            "width": width,
            "height": height,
            "ram_cap_mb": float(args.ram_cap_mb),
            **config_snapshot,
        }
        _print_pipeline_summary(summary)
        if args.save_output and last_frame is not None:
            out_path = ROOT / args.save_output
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), _annotate_frame(last_frame, last_boxes, last_kps))
            print(f"Saved annotation: {out_path}")
        _save_json(asdict(summary), args.output_json)

    return 0


# ===================================================================
# Subcommand: video
# ===================================================================


@dataclass
class _VideoBenchResult:
    config: dict[str, object] = field(default_factory=dict)
    video_metrics: dict[str, object] | None = None
    throughput: dict[str, object] = field(default_factory=dict)
    memory_mb: dict[str, float] = field(default_factory=dict)


def _run_video_benchmark(
    config: AppConfig,
    *,
    clips: list[VideoClip] | None = None,
    video_path: Path | None = None,
    video_gt_frames: list[GTFrame] | None = None,
    enroll_dir: Path | None = None,
    warmup_frames: int = 10,
    ram_cap_mb: float = 4096.0,
) -> _VideoBenchResult:
    startup_memory = enforce_memory_cap(ram_cap_mb, "startup")
    pipeline = build_pipeline(config)
    # Use a fresh temp gallery so results are not contaminated by previous runs.
    gallery_tmp = tempfile.mkdtemp(prefix="bench_gallery_")
    gallery = GalleryStore(
        root_dir=Path(gallery_tmp),
        embedding_dim=config.embedding.embedding_dim,
    )
    post_init_memory = enforce_memory_cap(ram_cap_mb, "pipeline initialization")

    enrolled: list[str] = []
    if enroll_dir is not None and enroll_dir.exists():
        enrolled = enroll_gallery_from_dir(enroll_dir, gallery, pipeline)
        print(f"Enrolled {len(enrolled)} identities from mugshots")

    # Auto-enroll from clip frames for identities missing mugshots.
    if clips:
        enrolled_set = set(enrolled)
        auto = auto_enroll_from_clips(clips, gallery, pipeline, enrolled_set)
        if auto:
            enrolled.extend(auto)
            print(f"Auto-enrolled {len(auto)} identities from clip frames: {auto}")

    # Collect timing + tracking data across all clips/frames.
    loop_ms: list[float] = []
    detect_loop_ms: list[float] = []  # loop times for detection frames only
    detect_ms: list[float] = []  # component times for detection frames only
    track_ms: list[float] = []
    embed_ms: list[float] = []
    all_frame_records: list[list[TrackRecord]] = []
    all_gt_ordered: list[GTFrame] = []
    unknown_slugs_seen: set[str] = set()
    total_processed = 0
    has_gt = False

    # Build a list of (frame_iterator, gt_frames, track_id_offset) segments.
    segments: list[tuple[list[GTFrame], int]] = []  # (gt_frames, id_offset)

    if clips:
        has_gt = True
        offset = 0
        for clip in clips:
            segments.append((clip.gt_frames, offset))
            offset += 1_000_000  # generous offset to avoid track_id collision
    elif video_path is not None:
        gt = video_gt_frames or []
        has_gt = bool(gt)
        segments.append((gt, 0))

    wall_t0 = time.perf_counter()

    if clips:
        for seg_idx, clip in enumerate(clips):
            gt_frames, id_offset = segments[seg_idx]
            gt_by_frame = {gt.frame_index: gt for gt in gt_frames}
            runtime = LiveRuntime(pipeline=pipeline, gallery=gallery, config=config)

            for seq_idx, frame_path in enumerate(clip.frame_paths):
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                frame_bgr = np.asarray(img, dtype=np.uint8)
                t0 = time.perf_counter()
                result = runtime.process_frame(frame_bgr)
                t1 = time.perf_counter()
                total_processed += 1

                if total_processed > warmup_frames:
                    frame_ms = (t1 - t0) * 1000.0
                    loop_ms.append(frame_ms)
                    is_detect_frame = result.overlay.detect_ms > 0
                    if is_detect_frame:
                        detect_loop_ms.append(frame_ms)
                        detect_ms.append(result.overlay.detect_ms)
                        track_ms.append(result.overlay.track_ms)
                        embed_ms.append(result.overlay.embed_ms_total)

                    gt_frame = gt_by_frame.get(seq_idx)
                    gt_faces = gt_frame.faces if gt_frame is not None else []
                    all_gt_ordered.append(
                        gt_frame if gt_frame is not None else GTFrame(frame_index=0, faces=[])
                    )

                    track_boxes = [f.track.box for f in result.overlay.faces]
                    track_to_gt = match_tracks_to_gt(track_boxes, gt_faces)

                    records: list[TrackRecord] = []
                    for t_idx, overlay_face in enumerate(result.overlay.faces):
                        gt_subject = track_to_gt.get(t_idx)
                        assigned_name = None
                        if overlay_face.match is not None and overlay_face.match.matched:
                            assigned_name = overlay_face.match.name
                        # Only count unknowns for GT-matched faces (not distractors).
                        if (
                            gt_subject is not None
                            and overlay_face.match is not None
                            and overlay_face.match.slug is not None
                            and overlay_face.match.slug.startswith("unknown-")
                        ):
                            unknown_slugs_seen.add(overlay_face.match.slug)
                        records.append(
                            TrackRecord(
                                track_id=overlay_face.track.track_id + id_offset,
                                gt_subject=gt_subject,
                                assigned_name=assigned_name,
                            )
                        )
                    all_frame_records.append(records)

                if total_processed > 0 and total_processed % 100 == 0:
                    enforce_memory_cap(ram_cap_mb, f"frame {total_processed}")

            if clips and (seg_idx + 1) % 50 == 0:
                print(f"  Processed {seg_idx + 1}/{len(clips)} clips ...")

    elif video_path is not None:
        gt_frames_list = segments[0][0] if segments else []
        gt_by_frame = {gt.frame_index: gt for gt in gt_frames_list}
        runtime = LiveRuntime(pipeline=pipeline, gallery=gallery, config=config)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            msg = f"Failed to open video: {video_path}"
            raise RuntimeError(msg)

        frame_idx = 0
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break
            frame_bgr = np.asarray(raw_frame, dtype=np.uint8)
            t0 = time.perf_counter()
            result = runtime.process_frame(frame_bgr)
            t1 = time.perf_counter()
            total_processed += 1

            if total_processed > warmup_frames:
                frame_ms = (t1 - t0) * 1000.0
                loop_ms.append(frame_ms)
                is_detect_frame = result.overlay.detect_ms > 0
                if is_detect_frame:
                    detect_loop_ms.append(frame_ms)
                    detect_ms.append(result.overlay.detect_ms)
                    track_ms.append(result.overlay.track_ms)
                    embed_ms.append(result.overlay.embed_ms_total)

                if has_gt:
                    gt_frame = gt_by_frame.get(frame_idx)
                    gt_faces = gt_frame.faces if gt_frame is not None else []
                    all_gt_ordered.append(
                        gt_frame if gt_frame is not None else GTFrame(frame_index=0, faces=[])
                    )

                    track_boxes = [f.track.box for f in result.overlay.faces]
                    track_to_gt = match_tracks_to_gt(track_boxes, gt_faces)

                    records_v: list[TrackRecord] = []
                    for t_idx, overlay_face in enumerate(result.overlay.faces):
                        gt_subject = track_to_gt.get(t_idx)
                        assigned_name = None
                        if overlay_face.match is not None and overlay_face.match.matched:
                            assigned_name = overlay_face.match.name
                        if (
                            gt_subject is not None
                            and overlay_face.match is not None
                            and overlay_face.match.slug is not None
                            and overlay_face.match.slug.startswith("unknown-")
                        ):
                            unknown_slugs_seen.add(overlay_face.match.slug)
                        records_v.append(
                            TrackRecord(
                                track_id=overlay_face.track.track_id,
                                gt_subject=gt_subject,
                                assigned_name=assigned_name,
                            )
                        )
                    all_frame_records.append(records_v)

            if total_processed > 0 and total_processed % 100 == 0:
                enforce_memory_cap(ram_cap_mb, f"frame {total_processed}")
            frame_idx += 1

        cap.release()

    wall_seconds = time.perf_counter() - wall_t0
    final_memory = get_memory_stats()
    throughput = compute_throughput(
        loop_ms, detect_loop_ms, detect_ms, track_ms, embed_ms, wall_seconds,
    )

    video_metrics_dict: dict[str, object] | None = None
    if has_gt and all_frame_records:
        temporal = compute_temporal_metrics(all_frame_records, all_gt_ordered, unknown_slugs_seen)
        video_metrics_dict = {
            "id_switches": temporal.id_switches,
            "temporal_consistency_ratio": temporal.temporal_consistency_ratio,
            "fragmentation": temporal.fragmentation,
            "ttfi_frames": temporal.ttfi_frames,
            "ttfi_mean_frames": temporal.ttfi_mean_frames,
            "unknown_slugs_created": temporal.unknown_slugs_created,
            "unique_gt_subjects": temporal.unique_gt_subjects,
            "unknown_fragmentation_rate": temporal.unknown_fragmentation_rate,
        }

    result = _VideoBenchResult(
        config={
            "tracking_method": config.tracking.method,
            "det_every": config.live.det_every,
            "max_faces": config.live.max_faces,
            "embed_refresh_frames": config.live.embed_refresh_frames,
            "embed_refresh_iou": config.live.embed_refresh_iou,
            "detection_backend": config.detection.backend,
            "quantize_int8": config.embedding.quantize_int8,
            "gallery_enrolled": len(enrolled),
        },
        video_metrics=video_metrics_dict,
        throughput=asdict(throughput),
        memory_mb={
            "startup_rss": round(startup_memory.current_rss_mb, 2),
            "post_init_rss": round(post_init_memory.current_rss_mb, 2),
            "final_rss": round(final_memory.current_rss_mb, 2),
            "peak_rss": round(final_memory.peak_rss_mb, 2),
        },
    )
    # Clean up temp gallery.
    shutil.rmtree(gallery_tmp, ignore_errors=True)
    return result


def _print_video_result(result: _VideoBenchResult) -> None:
    print("\n=== Video benchmark summary ===")
    print(f"config: {json.dumps(result.config, indent=2)}")

    tp = result.throughput
    det_frames = tp.get("detection_frames", 0)
    trk_frames = tp.get("tracking_only_frames", 0)
    print("\n--- Throughput ---")
    print(
        f"frames: {tp.get('total_frames', 0)} "
        f"({det_frames} detection, {trk_frames} tracking-only)"
    )
    print(f"wall time: {tp.get('total_wall_seconds', 0):.2f} s")
    print(f"sustained FPS: {tp.get('sustained_fps', 0):.2f}")
    print(
        f"all-frames latency p50/p95/p99: {tp.get('latency_p50_ms', 0):.1f} / "
        f"{tp.get('latency_p95_ms', 0):.1f} / {tp.get('latency_p99_ms', 0):.1f} ms"
    )
    print(
        f"detection-frames latency p50/p95/p99: "
        f"{tp.get('detect_latency_p50_ms', 0):.1f} / "
        f"{tp.get('detect_latency_p95_ms', 0):.1f} / "
        f"{tp.get('detect_latency_p99_ms', 0):.1f} ms"
    )
    print(
        f"avg per detection frame: detect {tp.get('avg_detect_ms', 0):.1f} ms, "
        f"track {tp.get('avg_track_ms', 0):.1f} ms, "
        f"embed {tp.get('avg_embed_ms', 0):.1f} ms"
    )

    mem = result.memory_mb
    print("\n--- Memory ---")
    print(
        f"RSS (startup -> post-init -> final): "
        f"{mem.get('startup_rss', 0):.1f} -> {mem.get('post_init_rss', 0):.1f} "
        f"-> {mem.get('final_rss', 0):.1f} MiB"
    )
    print(f"peak RSS: {mem.get('peak_rss', 0):.1f} MiB")

    if result.video_metrics is not None:
        vm = result.video_metrics
        print("\n--- Temporal metrics ---")
        print(f"ID switches: {vm.get('id_switches', 0)}")
        print(f"temporal consistency ratio: {vm.get('temporal_consistency_ratio', 0):.4f}")
        print(f"fragmentation: {vm.get('fragmentation', 0)}")
        print(f"TTFI mean (frames): {vm.get('ttfi_mean_frames', -1)}")
        print(f"unknown slugs created: {vm.get('unknown_slugs_created', 0)}")
        print(f"unique GT subjects: {vm.get('unique_gt_subjects', 0)}")
        print(f"unknown fragmentation rate: {vm.get('unknown_fragmentation_rate', 0):.4f}")
        ttfi = vm.get("ttfi_frames", {})
        if ttfi:
            print(f"TTFI per subject: {json.dumps(ttfi, indent=2)}")


def _add_video_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("video", parents=[_COMMON_PARSER], help="Video tracking metrics")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to video file (.mp4, .avi, etc.)")
    group.add_argument("--fanvid-dir", help="Path to FANVID dataset root")
    p.add_argument("--ground-truth", help="JSON ground truth (for --video mode)")
    p.add_argument("--enroll-dir", help="Enrollment images: <dir>/<subject>/<img>.jpg")
    p.add_argument("--max-clips", type=int, default=0, help="Limit FANVID clips (0 = all)")
    p.add_argument("--split", choices=["train", "test", "both"], default="both")
    p.add_argument("--warmup-frames", type=int, default=10)


def cmd_video(args: argparse.Namespace) -> int:
    config = load_config_with_overrides(args.config, args.override)

    clips: list[VideoClip] | None = None
    video_path: Path | None = None
    gt_frames: list[GTFrame] | None = None
    enroll_dir: Path | None = Path(args.enroll_dir) if args.enroll_dir else None

    if args.video:
        video_path = Path(args.video) if Path(args.video).is_absolute() else ROOT / args.video
        if args.ground_truth:
            gt_path = (
                ROOT / args.ground_truth
                if not Path(args.ground_truth).is_absolute()
                else Path(args.ground_truth)
            )
            gt_frames = load_gt_json(gt_path)
    elif args.fanvid_dir:
        fv_dir = (
            ROOT / args.fanvid_dir
            if not Path(args.fanvid_dir).is_absolute()
            else Path(args.fanvid_dir)
        )
        clips = load_fanvid_gt(fv_dir, max_clips=args.max_clips, split=args.split)
        if not clips:
            print("No FANVID clips found (frames not downloaded?)")
            return 2
        print(f"Loaded {len(clips)} FANVID clips")
        if enroll_dir is None:
            candidate = fv_dir / "enrollment"
            if candidate.exists():
                enroll_dir = candidate

    result = _run_video_benchmark(
        config,
        clips=clips,
        video_path=video_path,
        video_gt_frames=gt_frames,
        enroll_dir=enroll_dir,
        warmup_frames=max(0, args.warmup_frames),
        ram_cap_mb=args.ram_cap_mb,
    )
    _print_video_result(result)
    _save_json(
        {
            "config": result.config,
            "video_metrics": result.video_metrics,
            "throughput": result.throughput,
            "memory_mb": result.memory_mb,
        },
        args.output_json,
    )
    return 0


# ===================================================================
# Subcommand: lfw
# ===================================================================


@dataclass(frozen=True)
class _PairSample:
    path1: Path
    path2: Path
    is_same: int


@dataclass
class _EmbeddingRecord:
    embedding: Float32Array | None
    detect_ms: float
    embed_ms: float
    had_face: bool
    image_ok: bool


def _parse_pairs_file(
    pairs_file: Path,
    lfw_dir: Path,
    max_pairs: int = 0,
) -> list[_PairSample]:
    lines = [ln.strip() for ln in pairs_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    start_idx = 1 if lines[0].split()[0].isdigit() else 0
    samples: list[_PairSample] = []
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) == 3:
            person, idx1, idx2 = parts[0], int(parts[1]), int(parts[2])
            samples.append(
                _PairSample(
                    path1=lfw_dir / person / f"{person}_{idx1:04d}.jpg",
                    path2=lfw_dir / person / f"{person}_{idx2:04d}.jpg",
                    is_same=1,
                )
            )
        elif len(parts) == 4:
            p1, i1, p2, i2 = parts[0], int(parts[1]), parts[2], int(parts[3])
            samples.append(
                _PairSample(
                    path1=lfw_dir / p1 / f"{p1}_{i1:04d}.jpg",
                    path2=lfw_dir / p2 / f"{p2}_{i2:04d}.jpg",
                    is_same=0,
                )
            )
        else:
            raise RuntimeError(f"Unexpected pairs format line: {line}")
        if 0 < max_pairs <= len(samples):
            break
    return samples


def _parse_view2_pairs(
    pairs_file: Path,
    lfw_dir: Path,
    max_pairs: int = 0,
) -> tuple[list[_PairSample], int, int]:
    lines = [ln.strip() for ln in pairs_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    header = lines[0].split()
    if len(header) < 2 or not header[0].isdigit() or not header[1].isdigit():
        raise RuntimeError("Expected first line: '<folds> <pairs_per_class>'")
    folds = int(header[0])
    pairs_per_class = int(header[1])
    samples = _parse_pairs_file(pairs_file, lfw_dir, max_pairs)
    # Remove the re-parsed header line sample count if any
    return samples, folds, pairs_per_class * 2


def _extract_embedding(
    image_path: Path,
    pipeline: FacePipeline,
) -> _EmbeddingRecord:
    image = cv2.imread(str(image_path))
    if image is None:
        return _EmbeddingRecord(None, 0.0, 0.0, had_face=False, image_ok=False)
    image = np.asarray(image, dtype=np.uint8)
    det, det_ms = pipeline.detect(image)
    if det.kps is None or len(det.boxes) == 0:
        return _EmbeddingRecord(None, det_ms, 0.0, had_face=False, image_ok=True)
    best_idx = int(np.argmax(det.boxes[:, 4]))
    emb, emb_ms = pipeline.embed_from_kps(image, det.kps[best_idx])
    return _EmbeddingRecord(emb, det_ms, emb_ms, had_face=True, image_ok=True)


def _score_pairs(
    pairs: list[_PairSample],
    cache: dict[str, _EmbeddingRecord],
) -> tuple[Float32Array, npt.NDArray[np.int32], int]:
    scores: list[float] = []
    labels: list[int] = []
    dropped = 0
    for pair in pairs:
        r1 = cache[str(pair.path1)]
        r2 = cache[str(pair.path2)]
        if r1.embedding is None or r2.embedding is None:
            dropped += 1
            continue
        scores.append(float(np.dot(r1.embedding, r2.embedding)))
        labels.append(pair.is_same)
    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32), dropped


def _accuracy_at_threshold(
    scores: Float32Array, labels: npt.NDArray[np.int32], threshold: float
) -> float:
    preds = scores >= threshold
    return float(np.mean((preds.astype(np.int32) == labels).astype(np.float32)))


def _find_best_threshold(
    scores: Float32Array,
    labels: npt.NDArray[np.int32],
    steps: int,
) -> tuple[float, float]:
    if len(scores) == 0:
        return 0.5, 0.0
    thresholds = np.linspace(-1.0, 1.0, num=steps, dtype=np.float32)
    best_thr, best_acc = 0.5, -1.0
    for thr in thresholds:
        acc = _accuracy_at_threshold(scores, labels, float(thr))
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, best_acc


def _roc_curve(
    scores: Float32Array,
    labels: npt.NDArray[np.int32],
    steps: int,
) -> tuple[Float32Array, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    thresholds = np.linspace(1.0, -1.0, num=steps, dtype=np.float32)
    positives = max(1, int(np.sum(labels == 1)))
    negatives = max(1, int(np.sum(labels == 0)))
    tpr_list: list[float] = []
    fpr_list: list[float] = []
    for thr in thresholds:
        preds = scores >= thr
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tpr_list.append(tp / positives)
        fpr_list.append(fp / negatives)
    return thresholds, np.array(fpr_list, dtype=np.float64), np.array(tpr_list, dtype=np.float64)


def _compute_auc(fpr: npt.NDArray[np.float64], tpr: npt.NDArray[np.float64]) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def _compute_eer(
    thresholds: Float32Array,
    fpr: npt.NDArray[np.float64],
    tpr: npt.NDArray[np.float64],
) -> tuple[float, float]:
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = (float(fpr[idx]) + float(fnr[idx])) / 2.0
    return eer, float(thresholds[idx])


def _tar_at_far(
    thresholds: Float32Array,
    fpr: npt.NDArray[np.float64],
    tpr: npt.NDArray[np.float64],
    target_far: float,
) -> tuple[float, float]:
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        return 0.0, float(thresholds[-1])
    idx = valid[int(np.argmax(tpr[valid]))]
    return float(tpr[idx]), float(thresholds[idx])


def _evaluate_view2_cv(
    scores: Float32Array,
    labels: npt.NDArray[np.int32],
    folds: int,
    fold_size: int,
    threshold_steps: int,
) -> tuple[float, float, float]:
    available_folds = min(folds, len(scores) // fold_size)
    if available_folds < 2:
        return 0.0, 0.0, 0.0
    fold_accs: list[float] = []
    fold_thresholds: list[float] = []
    for fold_idx in range(available_folds):
        start = fold_idx * fold_size
        end = start + fold_size
        test_s, test_l = scores[start:end], labels[start:end]
        train_s = np.concatenate([scores[:start], scores[end:]], axis=0)
        train_l = np.concatenate([labels[:start], labels[end:]], axis=0)
        thr, _ = _find_best_threshold(train_s, train_l, steps=max(101, threshold_steps))
        fold_accs.append(_accuracy_at_threshold(test_s, test_l, thr))
        fold_thresholds.append(thr)
    return (
        float(statistics.fmean(fold_accs)),
        float(np.std(np.array(fold_accs, dtype=np.float64))),
        float(statistics.fmean(fold_thresholds)),
    )


def _add_lfw_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("lfw", parents=[_COMMON_PARSER], help="LFW verification accuracy")
    p.add_argument("--lfw-dir", default="data/lfw/lfw_funneled")
    p.add_argument("--train-pairs", default="data/lfw/pairsDevTrain.txt")
    p.add_argument("--test-pairs", default="data/lfw/pairsDevTest.txt")
    p.add_argument("--view2-pairs", default="", help="Optional view2 pairs.txt")
    p.add_argument("--max-train-pairs", type=int, default=0)
    p.add_argument("--max-test-pairs", type=int, default=0)
    p.add_argument("--max-view2-pairs", type=int, default=0)
    p.add_argument("--view2-folds", type=int, default=10)
    p.add_argument("--view2-pairs-per-class", type=int, default=300)
    p.add_argument("--threshold-steps", type=int, default=2001)


def cmd_lfw(args: argparse.Namespace) -> int:
    startup_memory = enforce_memory_cap(args.ram_cap_mb, "startup")
    config = load_config_with_overrides(args.config, args.override)

    lfw_dir = Path(args.lfw_dir) if Path(args.lfw_dir).is_absolute() else ROOT / args.lfw_dir
    train_pf = (
        Path(args.train_pairs) if Path(args.train_pairs).is_absolute() else ROOT / args.train_pairs
    )
    test_pf = (
        Path(args.test_pairs) if Path(args.test_pairs).is_absolute() else ROOT / args.test_pairs
    )
    view2_pf: Path | None = None
    if args.view2_pairs:
        view2_pf = (
            Path(args.view2_pairs)
            if Path(args.view2_pairs).is_absolute()
            else ROOT / args.view2_pairs
        )

    for path in [lfw_dir, train_pf, test_pf] + ([view2_pf] if view2_pf else []):
        if not path.exists():
            print(f"Missing required path: {path}")
            return 2

    pipeline = build_pipeline(config)
    post_init_memory = enforce_memory_cap(args.ram_cap_mb, "pipeline initialization")

    train_pairs = _parse_pairs_file(train_pf, lfw_dir, max(0, args.max_train_pairs))
    test_pairs = _parse_pairs_file(test_pf, lfw_dir, max(0, args.max_test_pairs))
    view2_pairs: list[_PairSample] = []
    view2_folds = max(2, args.view2_folds)
    view2_fold_size = max(2, args.view2_pairs_per_class) * 2
    if view2_pf is not None:
        view2_pairs, parsed_folds, parsed_fold_size = _parse_view2_pairs(
            view2_pf,
            lfw_dir,
            max(0, args.max_view2_pairs),
        )
        if args.max_view2_pairs == 0:
            view2_folds, view2_fold_size = parsed_folds, parsed_fold_size

    all_paths = sorted(
        {str(p.path1) for p in train_pairs + test_pairs + view2_pairs}
        | {str(p.path2) for p in train_pairs + test_pairs + view2_pairs}
    )
    print(f"train: {len(train_pairs)}, test: {len(test_pairs)}, images: {len(all_paths)}")

    # Extract embeddings.
    cache: dict[str, _EmbeddingRecord] = {}
    det_ms_list: list[float] = []
    emb_ms_list: list[float] = []
    t0 = time.perf_counter()
    for idx, path_str in enumerate(all_paths, 1):
        rec = _extract_embedding(Path(path_str), pipeline)
        cache[path_str] = rec
        if rec.image_ok:
            det_ms_list.append(rec.detect_ms)
        if rec.embedding is not None:
            emb_ms_list.append(rec.embed_ms)
        if idx % 250 == 0 or idx == len(all_paths):
            enforce_memory_cap(args.ram_cap_mb, f"LFW embedding pass {idx}")
        if idx % 500 == 0 or idx == len(all_paths):
            print(f"  processed: {idx}/{len(all_paths)}")
    preprocess_s = time.perf_counter() - t0

    # Score pairs.
    train_scores, train_labels, train_dropped = _score_pairs(train_pairs, cache)
    test_scores, test_labels, test_dropped = _score_pairs(test_pairs, cache)
    view2_scores = np.array([], dtype=np.float32)
    view2_labels = np.array([], dtype=np.int32)
    view2_dropped = 0
    if view2_pf is not None:
        view2_scores, view2_labels, view2_dropped = _score_pairs(view2_pairs, cache)

    # Metrics.
    best_thr, train_acc = _find_best_threshold(
        train_scores,
        train_labels,
        max(101, args.threshold_steps),
    )
    test_acc = _accuracy_at_threshold(test_scores, test_labels, best_thr)
    thresholds, test_fpr, test_tpr = _roc_curve(
        test_scores,
        test_labels,
        max(101, args.threshold_steps),
    )
    test_auc = _compute_auc(test_fpr, test_tpr)
    test_eer, eer_thr = _compute_eer(thresholds, test_fpr, test_tpr)
    tar_1e2, tar_thr = _tar_at_far(thresholds, test_fpr, test_tpr, target_far=1e-2)
    view2_cv_acc, view2_cv_std, view2_cv_thr = 0.0, 0.0, 0.0
    if view2_pf is not None and len(view2_scores) > 0:
        view2_cv_acc, view2_cv_std, view2_cv_thr = _evaluate_view2_cv(
            view2_scores,
            view2_labels,
            view2_folds,
            view2_fold_size,
            args.threshold_steps,
        )

    final_memory = enforce_memory_cap(args.ram_cap_mb, "LFW evaluation completion")
    face_ok = sum(1 for r in cache.values() if r.embedding is not None)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731

    # Print.
    print("\n=== LFW Validation Summary ===")
    print(f"train usable: {len(train_scores)}/{len(train_pairs)} (dropped {train_dropped})")
    print(f"test usable: {len(test_scores)}/{len(test_pairs)} (dropped {test_dropped})")
    if view2_pf:
        print(f"view2 usable: {len(view2_scores)}/{len(view2_pairs)} (dropped {view2_dropped})")
    print(f"face detection+embedding rate: {fmt(face_ok / max(1, len(all_paths)))}")
    print(f"best threshold (train): {fmt(best_thr)}")
    print(f"train acc: {fmt(train_acc)}  |  test acc: {fmt(test_acc)}")
    print(f"test ROC-AUC: {fmt(test_auc)}")
    print(f"test EER: {fmt(test_eer)} @ thr {fmt(eer_thr)}")
    print(f"test TAR@FAR<=1e-2: {fmt(tar_1e2)} @ thr {fmt(tar_thr)}")
    if view2_pf and len(view2_scores) > 0:
        print(
            f"view2 10-fold: {fmt(view2_cv_acc)} (std {fmt(view2_cv_std)}) thr {fmt(view2_cv_thr)}"
        )
    print(
        f"memory RSS: {startup_memory.current_rss_mb:.1f} -> "
        f"{post_init_memory.current_rss_mb:.1f} -> {final_memory.current_rss_mb:.1f} MiB, "
        f"peak {final_memory.peak_rss_mb:.1f} MiB"
    )

    _save_json(
        {
            "metrics": {
                "best_threshold_train": float(best_thr),
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "test_auc": float(test_auc),
                "test_eer": float(test_eer),
                "test_tar_far_1e-2": float(tar_1e2),
                "view2_cv_accuracy": float(view2_cv_acc),
            },
            "usable": {
                "train": int(len(train_scores)),
                "test": int(len(test_scores)),
                "view2": int(len(view2_scores)),
            },
            "throughput": {
                "preprocess_seconds": float(preprocess_s),
                "images_per_second": float(len(all_paths) / preprocess_s)
                if preprocess_s > 0
                else 0,
            },
        },
        args.output_json,
    )
    return 0


# ===================================================================
# Subcommand: compare
# ===================================================================


@dataclass(frozen=True)
class _VariantSummary:
    label: str
    avg_loop_ms: float
    avg_detect_ms: float
    avg_embed_ms: float
    avg_faces: float
    avg_fps: float
    final_memory: MemoryStats


def _benchmark_variant(
    *,
    label: str,
    pipeline: FacePipeline,
    image: UInt8Array,
    runs: int,
    warmup_runs: int,
    face_limit: int,
    ram_cap_mb: float,
) -> _VariantSummary:
    loop_ms: list[float] = []
    det_ms: list[float] = []
    emb_ms: list[float] = []
    faces: list[int] = []
    final_memory = enforce_memory_cap(ram_cap_mb, f"{label} startup")

    for idx in range(warmup_runs + max(1, runs)):
        t0 = time.perf_counter()
        det, det_latency = pipeline.detect(image)
        embed_total = 0.0
        if det.kps is not None:
            for fi in range(min(len(det.boxes), face_limit)):
                _, el = pipeline.embed_from_kps(image, det.kps[fi])
                embed_total += el
        if idx >= warmup_runs:
            loop_ms.append((time.perf_counter() - t0) * 1000.0)
            det_ms.append(det_latency)
            emb_ms.append(embed_total)
            faces.append(len(det.boxes))
            m_idx = idx - warmup_runs + 1
            if m_idx % 10 == 0 or m_idx == runs:
                final_memory = enforce_memory_cap(ram_cap_mb, f"{label} iter {m_idx}")

    loop_avg = statistics.fmean(loop_ms)
    return _VariantSummary(
        label=label,
        avg_loop_ms=loop_avg,
        avg_detect_ms=statistics.fmean(det_ms) if det_ms else 0.0,
        avg_embed_ms=statistics.fmean(emb_ms) if emb_ms else 0.0,
        avg_faces=statistics.fmean(faces) if faces else 0.0,
        avg_fps=1000.0 / loop_avg if loop_avg > 0 else 0.0,
        final_memory=final_memory,
    )


def _variant_dict(s: _VariantSummary) -> dict[str, float | str]:
    return {
        "label": s.label,
        "avg_loop_ms": round(s.avg_loop_ms, 3),
        "avg_detect_ms": round(s.avg_detect_ms, 3),
        "avg_embed_ms": round(s.avg_embed_ms, 3),
        "avg_faces": round(s.avg_faces, 3),
        "avg_fps": round(s.avg_fps, 3),
    }


def _add_compare_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("compare", help="A/B comparison of two configs")
    p.add_argument("--image", default="data/lena.jpg")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup-runs", type=int, default=5)
    p.add_argument("--ram-cap-mb", type=float, default=4096.0)
    p.add_argument("--output-json", default="")
    p.add_argument("--config-a", default="config.yaml")
    p.add_argument("--override-a", action="append", default=[])
    p.add_argument("--label-a", default="A")
    p.add_argument("--config-b", default="config.yaml")
    p.add_argument("--override-b", action="append", default=[])
    p.add_argument("--label-b", default="B")


def cmd_compare(args: argparse.Namespace) -> int:
    image = cv2.imread(str(ROOT / args.image))
    if image is None:
        print(f"Failed to read image: {ROOT / args.image}")
        return 2

    config_a = load_config_with_overrides(args.config_a, args.override_a)
    config_b = load_config_with_overrides(args.config_b, args.override_b)
    enforce_memory_cap(args.ram_cap_mb, "comparison startup")
    pipeline_a = build_pipeline(config_a)
    pipeline_b = build_pipeline(config_b)
    enforce_memory_cap(args.ram_cap_mb, "comparison initialization")

    frame = np.asarray(image, dtype=np.uint8)
    warmup = max(0, args.warmup_runs)
    runs = max(1, args.runs)
    sa = _benchmark_variant(
        label=args.label_a,
        pipeline=pipeline_a,
        image=frame,
        runs=runs,
        warmup_runs=warmup,
        face_limit=config_a.live.max_faces,
        ram_cap_mb=args.ram_cap_mb,
    )
    sb = _benchmark_variant(
        label=args.label_b,
        pipeline=pipeline_b,
        image=frame,
        runs=runs,
        warmup_runs=warmup,
        face_limit=config_b.live.max_faces,
        ram_cap_mb=args.ram_cap_mb,
    )

    print("\n=== Variant comparison ===")
    print(json.dumps({"a": _variant_dict(sa), "b": _variant_dict(sb)}, indent=2))
    print("\n=== Delta (b - a) ===")
    print(f"avg loop: {sb.avg_loop_ms - sa.avg_loop_ms:+.3f} ms")
    print(f"avg detect: {sb.avg_detect_ms - sa.avg_detect_ms:+.3f} ms")
    print(f"avg embed: {sb.avg_embed_ms - sa.avg_embed_ms:+.3f} ms")
    print(f"avg FPS: {sb.avg_fps - sa.avg_fps:+.3f}")

    _save_json(
        {
            "a": _variant_dict(sa),
            "b": _variant_dict(sb),
            "delta": {
                "avg_loop_ms": sb.avg_loop_ms - sa.avg_loop_ms,
                "avg_detect_ms": sb.avg_detect_ms - sa.avg_detect_ms,
                "avg_embed_ms": sb.avg_embed_ms - sa.avg_embed_ms,
                "avg_fps": sb.avg_fps - sa.avg_fps,
            },
        },
        args.output_json,
    )
    return 0


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified benchmark CLI for the face-recognition pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_pipeline_parser(subparsers)
    _add_video_parser(subparsers)
    _add_lfw_parser(subparsers)
    _add_compare_parser(subparsers)

    args = parser.parse_args()

    try:
        if args.command == "pipeline":
            return cmd_pipeline(args)
        if args.command == "video":
            return cmd_video(args)
        if args.command == "lfw":
            return cmd_lfw(args)
        if args.command == "compare":
            return cmd_compare(args)
        return 1
    except MemoryError as exc:
        print(exc)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
