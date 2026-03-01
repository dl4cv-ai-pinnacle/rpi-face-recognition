"""Face detector factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from butler import settings

if TYPE_CHECKING:
    from .base import FaceDetector


def create_detector() -> FaceDetector:
    """Create a face detector based on settings.FACE_DETECTOR."""
    name = settings.FACE_DETECTOR.lower()
    if name == "scrfd":
        from .scrfd import SCRFDDetector

        return SCRFDDetector()
    if name == "ultraface":
        from .ultraface import UltraFaceDetector

        return UltraFaceDetector()
    raise ValueError(f"Unknown face detector: {name!r} (expected 'scrfd' or 'ultraface')")
