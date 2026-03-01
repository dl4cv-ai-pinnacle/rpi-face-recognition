"""Face quality scoring — pure numpy, no cv2 dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from runtime_utils import Float32Array


@dataclass(frozen=True)
class FaceQuality:
    score: float  # combined 0..1
    det_confidence: float
    face_area: float


def compute_face_quality(
    box: Float32Array,
    *,
    min_area: float = 2500.0,
    max_area: float = 90000.0,
) -> FaceQuality:
    """Compute a quality score from a detection box [x1, y1, x2, y2, confidence].

    The score is the geometric mean of detection confidence and normalized face area.
    """
    arr = np.asarray(box, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 5:
        return FaceQuality(score=0.0, det_confidence=0.0, face_area=0.0)

    det_confidence = float(np.clip(arr[4], 0.0, 1.0))
    width = max(0.0, float(arr[2] - arr[0]))
    height = max(0.0, float(arr[3] - arr[1]))
    face_area = width * height

    if face_area <= 0.0 or det_confidence <= 0.0:
        return FaceQuality(score=0.0, det_confidence=det_confidence, face_area=face_area)

    norm_area = float(np.clip((face_area - min_area) / (max_area - min_area), 0.0, 1.0))
    score = math.sqrt(det_confidence * norm_area)
    return FaceQuality(score=score, det_confidence=det_confidence, face_area=face_area)
