"""Simple cross-frame face tracking with greedy IoU matching and EMA smoothing.

Origin: Valenia src/tracking.py
box_iou is public here (deduplicated — was duplicated in live_runtime.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.contracts import Float32Array


@dataclass
class Track:
    track_id: int
    box: Float32Array  # [x1, y1, x2, y2, confidence]
    kps: Float32Array | None  # [5, 2] or None
    age: int
    hits: int
    missed: int
    matched: bool  # True if updated by a detection this frame


def box_iou(box_a: Float32Array, box_b: Float32Array) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2, ...]."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class SimpleFaceTracker:
    """Keep short-lived face tracks using greedy IoU matching."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_missed: int = 3,
        smoothing: float = 0.65,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed = max(0, int(max_missed))
        self.smoothing = min(1.0, max(0.0, float(smoothing)))
        self._next_track_id = 1
        self._tracks: list[Track] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        boxes: Float32Array,
        kps: Float32Array | None,
        *,
        max_tracks: int | None = None,
    ) -> list[Track]:
        if max_tracks is not None and max_tracks >= 0:
            boxes = np.asarray(boxes[:max_tracks], dtype=np.float32)
            if kps is not None:
                kps = np.asarray(kps[:max_tracks], dtype=np.float32)

        matched_track_indices: set[int] = set()
        matched_detection_indices: set[int] = set()
        matches = self._match_detections(boxes)
        for track_idx, det_idx in matches:
            matched_track_indices.add(track_idx)
            matched_detection_indices.add(det_idx)

        next_tracks: list[Track] = []
        for track_idx, track in enumerate(self._tracks):
            if track_idx in matched_track_indices:
                det_idx = next(det for current, det in matches if current == track_idx)
                next_tracks.append(
                    self._update_matched_track(
                        track=track,
                        det_box=np.asarray(boxes[det_idx], dtype=np.float32),
                        det_kps=(
                            None if kps is None else np.asarray(kps[det_idx], dtype=np.float32)
                        ),
                    )
                )
                continue

            missed = track.missed + 1
            if missed > self.max_missed:
                continue
            next_tracks.append(
                Track(
                    track_id=track.track_id,
                    box=np.asarray(track.box, dtype=np.float32),
                    kps=(
                        None if track.kps is None else np.asarray(track.kps, dtype=np.float32)
                    ),
                    age=track.age + 1,
                    hits=track.hits,
                    missed=missed,
                    matched=False,
                )
            )

        for det_idx in range(len(boxes)):
            if det_idx in matched_detection_indices:
                continue
            next_tracks.append(
                self._create_track(
                    box=np.asarray(boxes[det_idx], dtype=np.float32),
                    kps=None if kps is None else np.asarray(kps[det_idx], dtype=np.float32),
                )
            )

        ranked = sorted(
            next_tracks,
            key=lambda t: (int(t.matched), -t.missed, float(t.box[4]), t.hits),
            reverse=True,
        )
        if max_tracks is not None and max_tracks >= 0:
            ranked = ranked[:max_tracks]
        ranked.sort(key=lambda t: t.track_id)
        self._tracks = ranked
        return list(self._tracks)

    def _match_detections(self, boxes: Float32Array) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self._tracks):
            for det_idx, det_box in enumerate(boxes):
                iou = box_iou(track.box[:4], np.asarray(det_box[:4], dtype=np.float32))
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_idx, det_idx))

        candidates.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        result: list[tuple[int, int]] = []
        for _, track_idx, det_idx in candidates:
            if track_idx in matched_tracks or det_idx in matched_detections:
                continue
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)
            result.append((track_idx, det_idx))
        return result

    def _create_track(self, box: Float32Array, kps: Float32Array | None) -> Track:
        track = Track(
            track_id=self._next_track_id,
            box=np.asarray(box, dtype=np.float32),
            kps=None if kps is None else np.asarray(kps, dtype=np.float32),
            age=1,
            hits=1,
            missed=0,
            matched=True,
        )
        self._next_track_id += 1
        return track

    def _update_matched_track(
        self,
        *,
        track: Track,
        det_box: Float32Array,
        det_kps: Float32Array | None,
    ) -> Track:
        box = _smooth_box(track.box, det_box, self.smoothing)
        if track.kps is None or det_kps is None:
            kps = None if det_kps is None else np.asarray(det_kps, dtype=np.float32)
        else:
            kps = _smooth_points(track.kps, det_kps, self.smoothing)
        return Track(
            track_id=track.track_id,
            box=box,
            kps=kps,
            age=track.age + 1,
            hits=track.hits + 1,
            missed=0,
            matched=True,
        )


def _smooth_box(previous: Float32Array, current: Float32Array, alpha: float) -> Float32Array:
    smoothed = np.asarray(current, dtype=np.float32).copy()
    smoothed[:4] = (alpha * current[:4]) + ((1.0 - alpha) * previous[:4])
    smoothed[4] = current[4]
    return np.asarray(smoothed, dtype=np.float32)


def _smooth_points(previous: Float32Array, current: Float32Array, alpha: float) -> Float32Array:
    smoothed = (alpha * current) + ((1.0 - alpha) * previous)
    return np.asarray(smoothed, dtype=np.float32)
