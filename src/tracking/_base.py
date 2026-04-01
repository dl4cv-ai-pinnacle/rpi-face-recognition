from dataclasses import dataclass

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
