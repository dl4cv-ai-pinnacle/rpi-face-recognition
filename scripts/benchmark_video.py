#!/usr/bin/env python3
"""Video benchmark: tracking quality + identity metrics on ChokePoint or raw video.

Measures temporal metrics that single-image benchmarks cannot capture:
identity stability (IDSW, TCR), time-to-first-identification (TTFI),
unknown fragmentation rate (UFR), plus throughput and memory.

Usage examples:
    # Arbitrary video with ground truth
    uv run --python 3.13 python scripts/benchmark_video.py \
        --config config.yaml --video data/clip.mp4 \
        --ground-truth data/clip_gt.json --enroll-dir data/enroll/ \
        --output-json results/video.json

    # ChokePoint dataset
    uv run --python 3.13 python scripts/benchmark_video.py \
        --config config.yaml --chokepoint-dir data/chokepoint \
        --output-json results/chokepoint.json

    # Throughput-only (no ground truth)
    uv run --python 3.13 python scripts/benchmark_video.py \
        --config config.yaml --video data/clip.mp4

    # With config overrides
    uv run --python 3.13 python scripts/benchmark_video.py \
        --config config.yaml --video data/clip.mp4 \
        --ground-truth data/clip_gt.json --enroll-dir data/enroll/ \
        --override live.det_every=5 --override tracking.method=kalman
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from src.config import AppConfig, load_config  # noqa: E402
from src.contracts import Float32Array, UInt8Array  # noqa: E402
from src.gallery import GalleryStore  # noqa: E402
from src.live import LiveRuntime  # noqa: E402
from src.metrics import enforce_memory_cap, get_memory_stats  # noqa: E402
from src.pipeline import build_pipeline  # noqa: E402
from src.tracking import box_iou  # noqa: E402

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
        target: object = raw
        for part in parts[:-1]:
            target = target[part]  # type: ignore[index]
        target[parts[-1]] = _auto_cast(value)  # type: ignore[index]


def _load_config_with_overrides(config_path: str, overrides: list[str]) -> AppConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if overrides:
        _apply_overrides(raw, overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(raw, tmp)
        tmp_path = tmp.name
    config = load_config(tmp_path)
    Path(tmp_path).unlink()
    return config


# ---------------------------------------------------------------------------
# Ground truth types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GTFace:
    subject_id: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class GTFrame:
    frame_index: int
    faces: list[GTFace]


# ---------------------------------------------------------------------------
# Ground truth loaders
# ---------------------------------------------------------------------------


def load_gt_json(gt_path: Path) -> list[GTFrame]:
    """Load ground truth from simple JSON format.

    Expected format:
    [
      {"frame": 0, "faces": [{"id": "alice", "bbox": [x1, y1, x2, y2]}, ...]},
      ...
    ]
    """
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    frames: list[GTFrame] = []
    for entry in data:
        faces = [
            GTFace(subject_id=f["id"], bbox=tuple(f["bbox"]))  # type: ignore[arg-type]
            for f in entry["faces"]
        ]
        frames.append(GTFrame(frame_index=entry["frame"], faces=faces))
    return frames


def load_chokepoint_gt(chokepoint_dir: Path) -> tuple[list[Path], list[GTFrame]]:
    """Load ChokePoint dataset: frame paths + ground truth from directory structure.

    Expected layout:
        chokepoint_dir/
            ground_truth/
                <sequence_id>.xml or <sequence_id>.txt
            frames/
                <sequence_id>/
                    <frame_0001>.jpg
                    ...

    Returns (frame_paths_sorted, gt_frames).
    """
    frames_dir = chokepoint_dir / "frames"
    gt_dir = chokepoint_dir / "ground_truth"

    if not frames_dir.exists():
        msg = f"ChokePoint frames directory not found: {frames_dir}"
        raise FileNotFoundError(msg)

    # Collect all frames across all sequences, sorted.
    frame_paths: list[Path] = sorted(frames_dir.rglob("*.jpg")) + sorted(
        frames_dir.rglob("*.png")
    )
    if not frame_paths:
        msg = f"No image files found in {frames_dir}"
        raise FileNotFoundError(msg)

    # Parse ground truth if available.
    gt_frames: list[GTFrame] = []
    if gt_dir.exists():
        for gt_file in sorted(gt_dir.glob("*.txt")):
            gt_frames.extend(_parse_chokepoint_txt(gt_file))

    return frame_paths, gt_frames


def _parse_chokepoint_txt(gt_file: Path) -> list[GTFrame]:
    """Parse a ChokePoint text-format ground truth file.

    Expected format per line: frame_id subject_id x1 y1 x2 y2
    """
    frames_by_idx: dict[int, list[GTFace]] = {}
    for line in gt_file.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        frame_idx = int(parts[0])
        subject_id = parts[1]
        x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        face = GTFace(subject_id=subject_id, bbox=(x1, y1, x2, y2))
        frames_by_idx.setdefault(frame_idx, []).append(face)
    return [
        GTFrame(frame_index=idx, faces=faces) for idx, faces in sorted(frames_by_idx.items())
    ]


# ---------------------------------------------------------------------------
# Gallery enrollment from reference images
# ---------------------------------------------------------------------------


def enroll_gallery_from_dir(
    enroll_dir: Path,
    gallery: GalleryStore,
    pipeline: object,
) -> list[str]:
    """Enroll identities from a directory of reference images.

    Expected layout: enroll_dir/<subject_name>/<image>.jpg
    Returns list of enrolled subject names.
    """
    enrolled: list[str] = []
    for subject_dir in sorted(enroll_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        name = subject_dir.name
        uploads: list[tuple[str, bytes]] = []
        for img_path in sorted(subject_dir.glob("*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                uploads.append((img_path.name, img_path.read_bytes()))
        if uploads:
            gallery.enroll(name, uploads, pipeline)  # type: ignore[arg-type]
            enrolled.append(name)
    return enrolled


# ---------------------------------------------------------------------------
# Temporal metrics computation
# ---------------------------------------------------------------------------


@dataclass
class TrackRecord:
    """Per-frame record of what identity was assigned to a track."""

    track_id: int
    gt_subject: str | None  # matched GT subject (via IoU)
    assigned_name: str | None  # what the pipeline called it


@dataclass
class TemporalMetrics:
    id_switches: int
    temporal_consistency_ratio: float
    fragmentation: int
    ttfi_frames: dict[str, int]  # subject_id → frames to first correct ID
    ttfi_mean_frames: float
    unknown_slugs_created: int
    unique_gt_subjects: int
    unknown_fragmentation_rate: float


def _match_tracks_to_gt(
    track_boxes: list[Float32Array],
    gt_faces: list[GTFace],
    iou_threshold: float = 0.3,
) -> dict[int, str]:
    """Match pipeline track indices to GT faces via IoU (greedy)."""
    if not track_boxes or not gt_faces:
        return {}

    # Build IoU matrix.
    n_tracks = len(track_boxes)
    n_gt = len(gt_faces)
    pairs: list[tuple[float, int, int]] = []
    for t_idx in range(n_tracks):
        t_box = track_boxes[t_idx][:4]
        for g_idx in range(n_gt):
            gt_box = np.array(gt_faces[g_idx].bbox, dtype=np.float32)
            iou = box_iou(t_box, gt_box)
            if iou >= iou_threshold:
                pairs.append((iou, t_idx, g_idx))

    # Greedy assignment (sufficient for evaluation).
    pairs.sort(reverse=True)
    matched_tracks: set[int] = set()
    matched_gt: set[int] = set()
    result: dict[int, str] = {}
    for _, t_idx, g_idx in pairs:
        if t_idx in matched_tracks or g_idx in matched_gt:
            continue
        matched_tracks.add(t_idx)
        matched_gt.add(g_idx)
        result[t_idx] = gt_faces[g_idx].subject_id
    return result


def compute_temporal_metrics(
    frame_records: list[list[TrackRecord]],
    gt_frames: list[GTFrame],
    unknown_slugs: set[str],
) -> TemporalMetrics:
    """Compute IDSW, TCR, fragmentation, TTFI, UFR from per-frame track records."""
    # --- ID Switches ---
    # Track the last GT subject assigned to each track_id.
    track_last_gt: dict[int, str] = {}
    id_switches = 0
    total_tracked = 0

    for records in frame_records:
        for rec in records:
            if rec.gt_subject is None:
                continue
            total_tracked += 1
            if rec.track_id in track_last_gt and track_last_gt[rec.track_id] != rec.gt_subject:
                id_switches += 1
            track_last_gt[rec.track_id] = rec.gt_subject

    tcr = 1.0 - (id_switches / total_tracked) if total_tracked > 0 else 1.0

    # --- Fragmentation ---
    # Count times a GT subject's trajectory is interrupted.
    gt_active: dict[str, bool] = {}  # subject → was tracked last frame
    fragmentation = 0
    for frame_idx, records in enumerate(frame_records):
        current_gt_subjects = {rec.gt_subject for rec in records if rec.gt_subject is not None}
        # Get expected GT subjects for this frame.
        expected_subjects: set[str] = set()
        if frame_idx < len(gt_frames):
            expected_subjects = {f.subject_id for f in gt_frames[frame_idx].faces}

        for subject in expected_subjects:
            was_tracked = gt_active.get(subject, False)
            is_tracked = subject in current_gt_subjects
            if was_tracked and not is_tracked:
                fragmentation += 1
            gt_active[subject] = is_tracked

    # --- TTFI (Time To First Identification) ---
    # For each GT subject, count frames from first GT appearance to first correct match.
    first_gt_frame: dict[str, int] = {}
    first_correct_frame: dict[str, int] = {}

    for frame_idx, gt_frame in enumerate(gt_frames):
        for gt_face in gt_frame.faces:
            sid = gt_face.subject_id
            if sid not in first_gt_frame:
                first_gt_frame[sid] = frame_idx

    for frame_idx, records in enumerate(frame_records):
        for rec in records:
            if rec.gt_subject is not None and rec.assigned_name is not None:
                sid = rec.gt_subject
                # Check if the pipeline's assigned name matches the GT subject.
                if rec.assigned_name.lower() == sid.lower() and sid not in first_correct_frame:
                    first_correct_frame[sid] = frame_idx

    ttfi_frames: dict[str, int] = {}
    for sid, first_gt in first_gt_frame.items():
        if sid in first_correct_frame:
            ttfi_frames[sid] = first_correct_frame[sid] - first_gt
        else:
            ttfi_frames[sid] = -1  # Never correctly identified.

    valid_ttfi = [v for v in ttfi_frames.values() if v >= 0]
    ttfi_mean = statistics.fmean(valid_ttfi) if valid_ttfi else -1.0

    # --- Unknown Fragmentation Rate ---
    unique_gt = set(first_gt_frame.keys())
    n_unknown_slugs = len(unknown_slugs)
    n_unique_gt = len(unique_gt)
    ufr = n_unknown_slugs / n_unique_gt if n_unique_gt > 0 else 0.0

    return TemporalMetrics(
        id_switches=id_switches,
        temporal_consistency_ratio=round(tcr, 4),
        fragmentation=fragmentation,
        ttfi_frames=ttfi_frames,
        ttfi_mean_frames=round(ttfi_mean, 2),
        unknown_slugs_created=n_unknown_slugs,
        unique_gt_subjects=n_unique_gt,
        unknown_fragmentation_rate=round(ufr, 4),
    )


# ---------------------------------------------------------------------------
# Throughput metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThroughputMetrics:
    total_frames: int
    total_wall_seconds: float
    sustained_fps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    avg_detect_ms: float
    avg_track_ms: float
    avg_embed_ms: float


def compute_throughput(
    loop_ms: list[float],
    detect_ms: list[float],
    track_ms: list[float],
    embed_ms: list[float],
    wall_seconds: float,
) -> ThroughputMetrics:
    n = len(loop_ms)
    if n == 0:
        return ThroughputMetrics(
            total_frames=0,
            total_wall_seconds=wall_seconds,
            sustained_fps=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            avg_detect_ms=0.0,
            avg_track_ms=0.0,
            avg_embed_ms=0.0,
        )
    sorted_latency = sorted(loop_ms)
    fps = n / wall_seconds if wall_seconds > 0 else 0.0
    return ThroughputMetrics(
        total_frames=n,
        total_wall_seconds=round(wall_seconds, 3),
        sustained_fps=round(fps, 2),
        latency_p50_ms=round(sorted_latency[int(n * 0.50)], 2),
        latency_p95_ms=round(sorted_latency[min(int(n * 0.95), n - 1)], 2),
        latency_p99_ms=round(sorted_latency[min(int(n * 0.99), n - 1)], 2),
        avg_detect_ms=round(statistics.fmean(detect_ms), 2) if detect_ms else 0.0,
        avg_track_ms=round(statistics.fmean(track_ms), 2) if track_ms else 0.0,
        avg_embed_ms=round(statistics.fmean(embed_ms), 2) if embed_ms else 0.0,
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    config: dict[str, object] = field(default_factory=dict)
    video_metrics: dict[str, object] | None = None
    throughput: dict[str, object] = field(default_factory=dict)
    memory_mb: dict[str, float] = field(default_factory=dict)


def run_video_benchmark(
    config: AppConfig,
    video_path: Path | None,
    frame_paths: list[Path] | None,
    gt_frames: list[GTFrame] | None,
    enroll_dir: Path | None,
    warmup_frames: int,
    ram_cap_mb: float,
) -> BenchmarkResult:
    startup_memory = enforce_memory_cap(ram_cap_mb, "startup")

    pipeline = build_pipeline(config)
    gallery = GalleryStore(
        root_dir=Path(config.gallery.root_dir),
        embedding_dim=config.embedding.embedding_dim,
    )
    runtime = LiveRuntime(pipeline=pipeline, gallery=gallery, config=config)
    post_init_memory = enforce_memory_cap(ram_cap_mb, "pipeline initialization")

    # Enroll gallery if directory provided.
    enrolled: list[str] = []
    if enroll_dir is not None and enroll_dir.exists():
        enrolled = enroll_gallery_from_dir(enroll_dir, gallery, pipeline)
        print(f"Enrolled {len(enrolled)} identities: {enrolled}")

    # Build GT index for fast lookup.
    gt_by_frame: dict[int, GTFrame] = {}
    if gt_frames:
        for gt in gt_frames:
            gt_by_frame[gt.frame_index] = gt

    # Open video source.
    cap: cv2.VideoCapture | None = None
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            msg = f"Failed to open video: {video_path}"
            raise RuntimeError(msg)

    # --- Evaluation loop ---
    loop_ms: list[float] = []
    detect_ms: list[float] = []
    track_ms: list[float] = []
    embed_ms: list[float] = []
    all_frame_records: list[list[TrackRecord]] = []
    unknown_slugs_seen: set[str] = set()

    frame_idx = 0
    wall_t0 = time.perf_counter()

    while True:
        # Read frame.
        frame_bgr: UInt8Array | None = None
        if cap is not None:
            ret, raw_frame = cap.read()
            if not ret:
                break
            frame_bgr = np.asarray(raw_frame, dtype=np.uint8)
        elif frame_paths is not None:
            if frame_idx >= len(frame_paths):
                break
            img = cv2.imread(str(frame_paths[frame_idx]))
            if img is None:
                frame_idx += 1
                continue
            frame_bgr = np.asarray(img, dtype=np.uint8)
        else:
            break

        # Process frame.
        t0 = time.perf_counter()
        result = runtime.process_frame(frame_bgr)
        t1 = time.perf_counter()

        is_measured = frame_idx >= warmup_frames
        if is_measured:
            loop_ms.append((t1 - t0) * 1000.0)
            detect_ms.append(result.overlay.detect_ms)
            track_ms.append(result.overlay.track_ms)
            embed_ms.append(result.overlay.embed_ms_total)

            # Build track records for temporal metrics.
            if gt_frames is not None:
                gt_frame = gt_by_frame.get(frame_idx)
                gt_faces = gt_frame.faces if gt_frame is not None else []

                track_boxes = [f.track.box for f in result.overlay.faces]
                track_to_gt = _match_tracks_to_gt(track_boxes, gt_faces)

                records: list[TrackRecord] = []
                for t_idx, overlay_face in enumerate(result.overlay.faces):
                    gt_subject = track_to_gt.get(t_idx)
                    assigned_name = None
                    if overlay_face.match is not None and overlay_face.match.matched:
                        assigned_name = overlay_face.match.name
                    # Track unknown slugs.
                    if (
                        overlay_face.match is not None
                        and overlay_face.match.slug is not None
                        and overlay_face.match.slug.startswith("unknown-")
                    ):
                        unknown_slugs_seen.add(overlay_face.match.slug)
                    records.append(
                        TrackRecord(
                            track_id=overlay_face.track.track_id,
                            gt_subject=gt_subject,
                            assigned_name=assigned_name,
                        )
                    )
                all_frame_records.append(records)

        # Periodic memory check.
        if frame_idx > 0 and frame_idx % 100 == 0:
            enforce_memory_cap(ram_cap_mb, f"frame {frame_idx}")

        frame_idx += 1

    wall_seconds = time.perf_counter() - wall_t0

    if cap is not None:
        cap.release()

    final_memory = get_memory_stats()

    # --- Compute metrics ---
    throughput = compute_throughput(loop_ms, detect_ms, track_ms, embed_ms, wall_seconds)

    video_metrics_dict: dict[str, object] | None = None
    if gt_frames is not None and all_frame_records:
        temporal = compute_temporal_metrics(all_frame_records, gt_frames, unknown_slugs_seen)
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

    return BenchmarkResult(
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


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_result(result: BenchmarkResult) -> None:
    print("\n=== Video benchmark summary ===")
    print(f"config: {json.dumps(result.config, indent=2)}")

    tp = result.throughput
    print("\n--- Throughput ---")
    print(f"frames: {tp.get('total_frames', 0)}")
    print(f"wall time: {tp.get('total_wall_seconds', 0):.2f} s")
    print(f"sustained FPS: {tp.get('sustained_fps', 0):.2f}")
    print(f"latency p50/p95/p99: {tp.get('latency_p50_ms', 0):.1f} / "
          f"{tp.get('latency_p95_ms', 0):.1f} / {tp.get('latency_p99_ms', 0):.1f} ms")
    print(f"avg detect: {tp.get('avg_detect_ms', 0):.1f} ms, "
          f"track: {tp.get('avg_track_ms', 0):.1f} ms, "
          f"embed: {tp.get('avg_embed_ms', 0):.1f} ms")

    mem = result.memory_mb
    print("\n--- Memory ---")
    print(f"RSS (startup → post-init → final): "
          f"{mem.get('startup_rss', 0):.1f} → {mem.get('post_init_rss', 0):.1f} "
          f"→ {mem.get('final_rss', 0):.1f} MiB")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video benchmark for face recognition pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Pipeline config YAML path")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override (repeatable), e.g. live.det_every=5",
    )

    # Input sources (mutually exclusive).
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to video file (.mp4, .avi, etc.)")
    group.add_argument("--chokepoint-dir", help="Path to ChokePoint dataset root")

    # Ground truth and enrollment.
    parser.add_argument(
        "--ground-truth", help="JSON ground truth file (for --video mode)"
    )
    parser.add_argument(
        "--enroll-dir",
        help="Directory with enrollment images: <enroll-dir>/<subject>/<img>.jpg",
    )

    # Benchmark parameters.
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--ram-cap-mb", type=float, default=4096.0)
    parser.add_argument("--output-json", default="", help="JSON output path")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        config = _load_config_with_overrides(args.config, args.override)

        video_path: Path | None = None
        frame_paths: list[Path] | None = None
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
        elif args.chokepoint_dir:
            cp_dir = (
                ROOT / args.chokepoint_dir
                if not Path(args.chokepoint_dir).is_absolute()
                else Path(args.chokepoint_dir)
            )
            frame_paths, gt_frames = load_chokepoint_gt(cp_dir)
            if not gt_frames:
                gt_frames = None
            # Default enrollment from chokepoint enrollment dir if available.
            if enroll_dir is None:
                candidate = cp_dir / "enrollment"
                if candidate.exists():
                    enroll_dir = candidate

        result = run_video_benchmark(
            config=config,
            video_path=video_path,
            frame_paths=frame_paths,
            gt_frames=gt_frames,
            enroll_dir=enroll_dir,
            warmup_frames=max(0, args.warmup_frames),
            ram_cap_mb=args.ram_cap_mb,
        )

        print_result(result)

        if args.output_json:
            out_path = (
                ROOT / args.output_json
                if not Path(args.output_json).is_absolute()
                else Path(args.output_json)
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "config": result.config,
                        "video_metrics": result.video_metrics,
                        "throughput": result.throughput,
                        "memory_mb": result.memory_mb,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nSaved JSON report: {out_path}")

        return 0
    except MemoryError as exc:
        print(exc)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
