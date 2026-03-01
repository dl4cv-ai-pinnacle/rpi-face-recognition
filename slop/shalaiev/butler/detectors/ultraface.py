"""UltraFace-slim face detector via ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from butler import settings
from .base import Detection


_MODEL_FILENAME = "version-slim-320.onnx"


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> list[int]:
    """Non-maximum suppression. Returns indices of kept boxes."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


class UltraFaceDetector:
    """UltraFace-slim-320 via ONNX Runtime."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None

    @property
    def name(self) -> str:
        return "UltraFace-slim"

    def _ensure_loaded(self) -> ort.InferenceSession:
        if self._session is None:
            model_path = settings.MODELS_DIR / _MODEL_FILENAME
            if not model_path.exists():
                raise FileNotFoundError(
                    f"UltraFace model not found at {model_path}. "
                    "Download version-slim-320.onnx and place it in the models/ directory."
                )
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        session = self._ensure_loaded()
        orig_h, orig_w = frame_rgb.shape[:2]

        # Preprocess: resize to 320x240, normalize, HWC → NCHW.
        img = cv2.resize(frame_rgb, (320, 240))
        img = (img.astype(np.float32) - 127.0) / 128.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # add batch dim

        input_name = session.get_inputs()[0].name
        confidences, boxes = session.run(None, {input_name: img})
        # confidences: (1, N, 2)  boxes: (1, N, 4) in [0, 1] normalized coords
        confidences = confidences[0]  # (N, 2)
        boxes = boxes[0]  # (N, 4)

        # Filter by face confidence (column 1 is face class).
        scores = confidences[:, 1]
        mask = scores > settings.DETECTION_CONFIDENCE_THRESHOLD
        scores = scores[mask]
        boxes = boxes[mask]

        if len(scores) == 0:
            return []

        # Denormalize to original frame coordinates.
        boxes[:, 0] *= orig_w  # x1
        boxes[:, 1] *= orig_h  # y1
        boxes[:, 2] *= orig_w  # x2
        boxes[:, 3] *= orig_h  # y2

        # NMS
        keep = _nms(boxes, scores)

        return [
            Detection(
                x1=float(boxes[i, 0]),
                y1=float(boxes[i, 1]),
                x2=float(boxes[i, 2]),
                y2=float(boxes[i, 3]),
                confidence=float(scores[i]),
            )
            for i in keep
        ]
