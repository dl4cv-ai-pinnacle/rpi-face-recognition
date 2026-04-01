"""Tests for face trackers — SimpleFaceTracker, KalmanFaceTracker, box_iou, factory."""

from __future__ import annotations

import numpy as np
from src.tracking import KalmanFaceTracker, SimpleFaceTracker, box_iou, create_tracker


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


# ---------------------------------------------------------------------------
# SimpleFaceTracker
# ---------------------------------------------------------------------------


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


class TestSimplePredict:
    def test_predict_returns_static_boxes(self) -> None:
        tracker = SimpleFaceTracker()
        box = np.array([_box(100, 100, 200, 200)])
        tracker.update(box, kps=None)

        predicted = tracker.predict()
        assert len(predicted) == 1
        assert predicted[0].matched is False
        # Box should be (approximately) unchanged since no motion model.
        np.testing.assert_allclose(predicted[0].box[:4], [100, 100, 200, 200], atol=1.0)

    def test_predict_increments_age(self) -> None:
        tracker = SimpleFaceTracker()
        box = np.array([_box(100, 100, 200, 200)])
        tracks = tracker.update(box, kps=None)
        predicted = tracker.predict()
        assert predicted[0].age == tracks[0].age + 1


# ---------------------------------------------------------------------------
# KalmanFaceTracker
# ---------------------------------------------------------------------------


class TestKalmanTrackerIdentity:
    def test_assigns_unique_track_ids(self) -> None:
        tracker = KalmanFaceTracker()
        boxes = np.array([_box(10, 10, 60, 60), _box(200, 200, 300, 300)])
        tracks = tracker.update(boxes, kps=None)
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2

    def test_maintains_id_across_frames(self) -> None:
        tracker = KalmanFaceTracker()
        box = np.array([_box(100, 100, 200, 200)])
        t1 = tracker.update(box, kps=None)
        t2 = tracker.update(box, kps=None)
        assert t1[0].track_id == t2[0].track_id

    def test_drops_after_max_missed(self) -> None:
        tracker = KalmanFaceTracker(max_missed=2)
        box = np.array([_box(100, 100, 200, 200)])
        tracker.update(box, kps=None)

        empty = np.zeros((0, 5), dtype=np.float32)
        tracker.update(empty, kps=None)  # missed=1
        tracker.update(empty, kps=None)  # missed=2
        tracks = tracker.update(empty, kps=None)  # missed=3 → dropped
        assert len(tracks) == 0

    def test_reset_clears_all_tracks(self) -> None:
        tracker = KalmanFaceTracker()
        box = np.array([_box(100, 100, 200, 200)])
        tracker.update(box, kps=None)
        tracker.reset()
        tracks = tracker.update(box, kps=None)
        assert tracks[0].track_id == 1


class TestKalmanPrediction:
    def test_predict_extrapolates_velocity(self) -> None:
        """Feed a box moving rightward; verify prediction continues the motion."""
        tracker = KalmanFaceTracker()
        # Frame 1: box at x=[100, 200]
        tracker.update(np.array([_box(100, 100, 200, 200)]), kps=None)
        # Frame 2: box shifted 20px right
        tracker.update(np.array([_box(120, 100, 220, 200)]), kps=None)
        # Frame 3: another 20px right
        tracker.update(np.array([_box(140, 100, 240, 200)]), kps=None)

        # Predict next frame without detection.
        predicted = tracker.predict()
        assert len(predicted) == 1
        # The predicted box should move further right (x1 > 140).
        assert float(predicted[0].box[0]) > 140.0
        assert predicted[0].matched is False

    def test_predict_returns_valid_box_dimensions(self) -> None:
        tracker = KalmanFaceTracker()
        tracker.update(np.array([_box(50, 50, 150, 150)]), kps=None)
        predicted = tracker.predict()
        box = predicted[0].box
        # Width and height should be positive.
        assert float(box[2]) > float(box[0])
        assert float(box[3]) > float(box[1])


class TestHungarianMatching:
    def test_optimal_assignment_differs_from_greedy(self) -> None:
        """Scenario where greedy would make a suboptimal assignment.

        Track A is at [0, 0, 50, 50], Track B at [40, 0, 90, 50].
        Detection 1 at [35, 0, 85, 50] (overlaps both, higher IoU with B).
        Detection 2 at [5, 0, 55, 50] (overlaps both, higher IoU with A).

        Greedy picks highest single IoU first and may steal the better match.
        Hungarian finds the globally optimal pairing.
        """
        tracker = KalmanFaceTracker(iou_threshold=0.1)
        # Establish two tracks.
        boxes_init = np.array([_box(0, 0, 50, 50), _box(40, 0, 90, 50)])
        tracks_init = tracker.update(boxes_init, kps=None)
        id_a = tracks_init[0].track_id
        id_b = tracks_init[1].track_id

        # Present detections that are swapped relative to original positions.
        boxes_new = np.array([_box(35, 0, 85, 50), _box(5, 0, 55, 50)])
        tracks_new = tracker.update(boxes_new, kps=None)

        ids_new = {t.track_id for t in tracks_new}
        # Both original tracks should survive (optimal matching).
        assert id_a in ids_new
        assert id_b in ids_new

    def test_handles_more_detections_than_tracks(self) -> None:
        tracker = KalmanFaceTracker()
        tracker.update(np.array([_box(100, 100, 200, 200)]), kps=None)
        # Two detections, one track → one match + one new track.
        tracks = tracker.update(
            np.array([_box(105, 105, 205, 205), _box(400, 400, 500, 500)]),
            kps=None,
        )
        assert len(tracks) == 2

    def test_handles_more_tracks_than_detections(self) -> None:
        tracker = KalmanFaceTracker(max_missed=5)
        tracker.update(
            np.array([_box(100, 100, 200, 200), _box(400, 400, 500, 500)]),
            kps=None,
        )
        # One detection, two tracks → one match + one missed.
        tracks = tracker.update(np.array([_box(105, 105, 205, 205)]), kps=None)
        assert len(tracks) == 2
        matched = [t for t in tracks if t.matched]
        assert len(matched) == 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestTrackerFactory:
    def test_creates_simple_tracker(self) -> None:
        tracker = create_tracker(method="simple")
        assert isinstance(tracker, SimpleFaceTracker)

    def test_creates_kalman_tracker(self) -> None:
        tracker = create_tracker(method="kalman")
        assert isinstance(tracker, KalmanFaceTracker)

    def test_unknown_method_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown tracking method"):
            create_tracker(method="nonexistent")

    def test_passes_params_to_simple(self) -> None:
        tracker = create_tracker(method="simple", iou_threshold=0.5, max_missed=10, smoothing=0.8)
        assert isinstance(tracker, SimpleFaceTracker)
        assert tracker.iou_threshold == 0.5
        assert tracker.max_missed == 10

    def test_passes_params_to_kalman(self) -> None:
        tracker = create_tracker(method="kalman", iou_threshold=0.5, max_missed=10)
        assert isinstance(tracker, KalmanFaceTracker)
        assert tracker.iou_threshold == 0.5
        assert tracker.max_missed == 10
