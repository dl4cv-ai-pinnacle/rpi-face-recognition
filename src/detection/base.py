from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray  # shape (4,) — x1, y1, x2, y2
    score: float
    landmarks: np.ndarray  # shape (5, 2) — x, y for each keypoint


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in an RGB image. Returns list of detections."""
        ...
