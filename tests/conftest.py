"""Shared test fixtures and stubs.

Provides lightweight fakes for detector, aligner, and embedder so that
pipeline and integration tests run without ONNX models or insightface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from src.contracts import Detection, Float32Array, UInt8Array


def make_frame(h: int = 480, w: int = 640) -> UInt8Array:
    """Create a synthetic BGR frame."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_landmarks() -> Float32Array:
    """Create plausible 5-point landmarks within a 640×480 frame."""
    return np.array(
        [[200, 180], [280, 180], [240, 220], [210, 260], [270, 260]],
        dtype=np.float32,
    )


def make_detection(with_landmarks: bool = True) -> Detection:
    """Create a plausible Detection."""
    return Detection(
        x1=150.0,
        y1=130.0,
        x2=330.0,
        y2=310.0,
        confidence=0.92,
        landmarks=make_landmarks() if with_landmarks else None,
    )


@dataclass
class StubDetector:
    """Returns canned detections. Tracks call count."""

    detections: list[Detection] = field(default_factory=list)
    call_count: int = 0

    @property
    def name(self) -> str:
        return "StubDetector"

    @property
    def provider_name(self) -> str:
        return "CPUExecutionProvider"

    def detect(self, frame_bgr: UInt8Array, /) -> list[Detection]:
        self.call_count += 1
        return self.detections


@dataclass
class StubAligner:
    """Returns a fixed 112×112 crop. Tracks call count."""

    call_count: int = 0

    @property
    def name(self) -> str:
        return "StubAligner"

    def align(self, frame_bgr: UInt8Array, landmarks: Float32Array, /) -> UInt8Array | None:
        self.call_count += 1
        return np.zeros((112, 112, 3), dtype=np.uint8)


@dataclass
class StubEmbedder:
    """Returns a fixed normalized embedding. Tracks call count."""

    embedding: Float32Array = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    call_count: int = 0

    @property
    def name(self) -> str:
        return "StubEmbedder"

    @property
    def provider_name(self) -> str:
        return "CPUExecutionProvider"

    def get_embedding(self, crop_bgr: UInt8Array, /) -> Float32Array:
        self.call_count += 1
        return self.embedding
