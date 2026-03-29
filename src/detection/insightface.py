"""SCRFD face detector via insightface library.

Origin: Shalaiev butler/detectors/scrfd.py — adapted to unified Detection dataclass
and config-driven initialization (no module-level settings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.contracts import Detection

if TYPE_CHECKING:
    from src.config import DetectionConfig
    from src.contracts import UInt8Array


class InsightFaceSCRFD:
    """SCRFD-500M using the insightface buffalo_sc model pack."""

    def __init__(self, config: DetectionConfig) -> None:
        from insightface.app import FaceAnalysis

        self._confidence_threshold = config.confidence_threshold
        self._app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection"],
        )
        self._app.prepare(ctx_id=-1)

    @property
    def name(self) -> str:
        return "SCRFD (insightface)"

    @property
    def provider_name(self) -> str:
        return "CPUExecutionProvider"

    def detect(self, frame_bgr: UInt8Array, /) -> list[Detection]:
        faces = self._app.get(frame_bgr)

        detections: list[Detection] = []
        for face in faces:
            if face.det_score < self._confidence_threshold:
                continue
            x1, y1, x2, y2 = face.bbox.astype(float)
            landmarks = None
            if face.kps is not None:
                landmarks = np.asarray(face.kps, dtype=np.float32)
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(face.det_score),
                    landmarks=landmarks,
                )
            )
        return detections
