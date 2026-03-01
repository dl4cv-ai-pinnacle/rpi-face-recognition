"""Face detector protocol and shared data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Landmark:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: tuple[Landmark, ...] | None = None  # 5 points when available


@runtime_checkable
class FaceDetector(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]: ...
