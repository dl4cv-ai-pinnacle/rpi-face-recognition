from __future__ import annotations

from src.tracking.kalman import KalmanFaceTracker
from src.tracking.simple import SimpleFaceTracker

from ._base import Track, box_iou


def create_tracker(
    *,
    method: str,
    iou_threshold: float = 0.3,
    max_missed: int = 3,
    smoothing: float = 0.65,
) -> SimpleFaceTracker | KalmanFaceTracker:
    """Build a tracker from the config method string."""
    match method:
        case "simple":
            return SimpleFaceTracker(
                iou_threshold=iou_threshold,
                max_missed=max_missed,
                smoothing=smoothing,
            )
        case "kalman":
            return KalmanFaceTracker(
                iou_threshold=iou_threshold,
                max_missed=max_missed,
                smoothing=smoothing,
            )
        case _:
            msg = f"Unknown tracking method: {method!r} (expected 'simple' or 'kalman')"
            raise ValueError(msg)


__all__ = ["create_tracker", "box_iou", "Track", "KalmanFaceTracker", "SimpleFaceTracker"]
