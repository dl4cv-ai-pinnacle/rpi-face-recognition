"""SCRFD-500M face detector via insightface."""

from __future__ import annotations

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from butler import settings
from .base import Detection, Landmark


class SCRFDDetector:
    """SCRFD-500M using the insightface buffalo_sc model pack."""

    def __init__(self) -> None:
        self._app: FaceAnalysis | None = None

    @property
    def name(self) -> str:
        return "SCRFD-500M"

    def _ensure_loaded(self) -> FaceAnalysis:
        if self._app is None:
            self._app = FaceAnalysis(
                name="buffalo_sc",
                root=str(settings.MODELS_DIR),
                allowed_modules=["detection"],
            )
            det_w, det_h = settings.DETECTION_RESOLUTION
            self._app.prepare(ctx_id=-1, det_size=(det_w, det_h))
        return self._app

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        app = self._ensure_loaded()
        # insightface expects BGR input.
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)

        detections: list[Detection] = []
        for face in faces:
            if face.det_score < settings.DETECTION_CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = face.bbox.astype(float)
            landmarks: tuple[Landmark, ...] | None = None
            if face.kps is not None:
                landmarks = tuple(
                    Landmark(float(pt[0]), float(pt[1])) for pt in face.kps
                )
            detections.append(
                Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(face.det_score),
                    landmarks=landmarks,
                )
            )
        return detections
