"""Face detection backends with factory dispatch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import DetectionConfig
    from src.contracts import DetectorLike

logger = logging.getLogger(__name__)


def create_detector(config: DetectionConfig) -> DetectorLike:
    """Create a detector backend based on config.backend."""
    match config.backend:
        case "insightface":
            from src.detection.insightface import InsightFaceSCRFD

            return InsightFaceSCRFD(config)
        case "ultraface":
            from src.detection.ultraface import UltraFaceDetector

            return UltraFaceDetector(config)
        case _:
            msg = f"Unknown detection backend: {config.backend!r}"
            raise ValueError(msg)


def create_detector_safe(config: DetectionConfig) -> DetectorLike | None:
    """Create a detector, returning None on failure instead of raising."""
    try:
        return create_detector(config)
    except Exception:
        logger.warning("Failed to load detector %s", config.backend, exc_info=True)
        return None
