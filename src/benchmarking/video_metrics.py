"""Temporal and throughput metrics for video face-recognition benchmarks."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from src.benchmarking.ground_truth import GTFace, GTFrame
from src.contracts import Float32Array
from src.tracking import box_iou

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TrackRecord:
    """Per-frame record of what identity was assigned to a tracked face."""

    track_id: int
    gt_subject: str | None  # matched GT subject (via IoU)
    assigned_name: str | None  # pipeline's gallery match name


@dataclass(frozen=True)
class TemporalMetrics:
    id_switches: int
    temporal_consistency_ratio: float
    fragmentation: int
    ttfi_frames: dict[str, int]  # subject_id -> frames to first correct ID
    ttfi_mean_frames: float
    unknown_slugs_created: int
    unique_gt_subjects: int
    unknown_fragmentation_rate: float


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


# ---------------------------------------------------------------------------
# Track-to-GT matching
# ---------------------------------------------------------------------------


def match_tracks_to_gt(
    track_boxes: list[Float32Array],
    gt_faces: list[GTFace],
    iou_threshold: float = 0.3,
) -> dict[int, str]:
    """Match pipeline track indices to GT faces via IoU (greedy)."""
    if not track_boxes or not gt_faces:
        return {}

    pairs: list[tuple[float, int, int]] = []
    for t_idx in range(len(track_boxes)):
        t_box = track_boxes[t_idx][:4]
        for g_idx in range(len(gt_faces)):
            gt_box = np.array(gt_faces[g_idx].bbox, dtype=np.float32)
            iou = box_iou(t_box, gt_box)
            if iou >= iou_threshold:
                pairs.append((iou, t_idx, g_idx))

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


# ---------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------


def compute_temporal_metrics(
    frame_records: list[list[TrackRecord]],
    gt_frames: list[GTFrame] | None,
    unknown_slugs: set[str],
) -> TemporalMetrics:
    """Compute IDSW, TCR, fragmentation, TTFI, UFR from per-frame track records.

    *gt_frames* is a flat list aligned by position with *frame_records* — entry
    ``gt_frames[i]`` contains the GT faces for the same frame as
    ``frame_records[i]``.
    """

    # --- ID Switches ---
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
    gt_active: dict[str, bool] = {}
    fragmentation = 0
    for frame_idx, records in enumerate(frame_records):
        current_gt_subjects = {rec.gt_subject for rec in records if rec.gt_subject is not None}
        expected_subjects: set[str] = set()
        if gt_frames is not None and frame_idx < len(gt_frames):
            expected_subjects = {f.subject_id for f in gt_frames[frame_idx].faces}

        for subject in expected_subjects:
            was_tracked = gt_active.get(subject, False)
            is_tracked = subject in current_gt_subjects
            if was_tracked and not is_tracked:
                fragmentation += 1
            gt_active[subject] = is_tracked

    # --- TTFI (Time To First Identification) ---
    first_gt_frame: dict[str, int] = {}
    first_correct_frame: dict[str, int] = {}

    if gt_frames is not None:
        for frame_idx, gt_entry in enumerate(gt_frames):
            for gt_face in gt_entry.faces:
                sid = gt_face.subject_id
                if sid not in first_gt_frame:
                    first_gt_frame[sid] = frame_idx

    for frame_idx, records in enumerate(frame_records):
        for rec in records:
            if rec.gt_subject is not None and rec.assigned_name is not None:
                sid = rec.gt_subject
                if rec.assigned_name.lower() == sid.lower() and sid not in first_correct_frame:
                    first_correct_frame[sid] = frame_idx

    ttfi_frames: dict[str, int] = {}
    for sid, first_gt in first_gt_frame.items():
        if sid in first_correct_frame:
            ttfi_frames[sid] = first_correct_frame[sid] - first_gt
        else:
            ttfi_frames[sid] = -1

    valid_ttfi = [v for v in ttfi_frames.values() if v >= 0]
    ttfi_mean = statistics.fmean(valid_ttfi) if valid_ttfi else -1.0

    # --- Unknown Fragmentation Rate ---
    n_unknown_slugs = len(unknown_slugs)
    n_unique_gt = len(first_gt_frame)
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
