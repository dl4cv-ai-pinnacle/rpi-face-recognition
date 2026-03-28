"""Tests for SimpleFaceTracker — identity persistence, missed drops, box_iou."""

from __future__ import annotations

import numpy as np
from src.tracking import SimpleFaceTracker, box_iou


def _box(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> np.ndarray:
    return np.array([x1, y1, x2, y2, conf], dtype=np.float32)


class TestBoxIou:
    def test_identical_boxes(self) -> None:
        b = _box(0, 0, 100, 100)
        assert box_iou(b, b) > 0.99

    def test_non_overlapping(self) -> None:
        a = _box(0, 0, 50, 50)
        b = _box(100, 100, 200, 200)
        assert box_iou(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = _box(0, 0, 100, 100)
        b = _box(50, 50, 150, 150)
        iou = box_iou(a, b)
        assert 0.1 < iou < 0.3


class TestTrackerIdentity:
    def test_assigns_unique_track_ids(self) -> None:
        tracker = SimpleFaceTracker()
        boxes = np.array([_box(10, 10, 60, 60), _box(200, 200, 300, 300)])
        tracks = tracker.update(boxes, kps=None)
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2

    def test_maintains_id_across_frames(self) -> None:
        tracker = SimpleFaceTracker()
        box = np.array([_box(100, 100, 200, 200)])
        t1 = tracker.update(box, kps=None)
        t2 = tracker.update(box, kps=None)
        assert t1[0].track_id == t2[0].track_id

    def test_drops_after_max_missed(self) -> None:
        tracker = SimpleFaceTracker(max_missed=2)
        box = np.array([_box(100, 100, 200, 200)])
        tracker.update(box, kps=None)

        empty = np.zeros((0, 5), dtype=np.float32)
        tracker.update(empty, kps=None)  # missed=1
        tracker.update(empty, kps=None)  # missed=2
        tracks = tracker.update(empty, kps=None)  # missed=3 → dropped
        assert len(tracks) == 0

    def test_reset_clears_all_tracks(self) -> None:
        tracker = SimpleFaceTracker()
        box = np.array([_box(100, 100, 200, 200)])
        tracker.update(box, kps=None)
        tracker.reset()
        tracks = tracker.update(box, kps=None)
        assert tracks[0].track_id == 1  # IDs restart after reset
