"""UltraFace-slim-320 face detector via ONNX Runtime.

Origin: Shalaiev butler/detectors/ultraface.py — adapted to unified Detection dataclass,
config-driven initialization, and NMS from Shalaiev's hand-written implementation.

UltraFace does NOT produce landmarks. Detected faces will have landmarks=None,
which triggers center-crop fallback in the alignment stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from src.contracts import Detection
from src.onnx_session import suppress_stderr_fd

if TYPE_CHECKING:
    from src.config import DetectionConfig
    from src.contracts import UInt8Array


_MODEL_FILENAME = "version-slim-320.onnx"


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> list[int]:
    """Greedy non-maximum suppression. Returns indices of kept boxes."""
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
    """UltraFace-slim-320 via ONNX Runtime. Fast fallback — no landmarks."""

    def __init__(self, config: DetectionConfig, model_dir: str = "models") -> None:
        self._confidence_threshold = config.confidence_threshold
        self._nms_threshold = config.nms_threshold
        model_path = Path(model_dir) / _MODEL_FILENAME
        if not model_path.exists():
            msg = f"UltraFace model not found at {model_path}. Run: bash scripts/download_models.sh"
            raise FileNotFoundError(msg)

        so = ort.SessionOptions()
        so.log_severity_level = 3
        with suppress_stderr_fd():
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def name(self) -> str:
        return "UltraFace-slim"

    @property
    def provider_name(self) -> str:
        providers = self._session.get_providers()
        return providers[0] if providers else "CPUExecutionProvider"

    def detect(self, frame_bgr: UInt8Array, /) -> list[Detection]:
        import cv2

        orig_h, orig_w = frame_bgr.shape[:2]

        # Preprocess: resize to 320x240, normalize to [-1, 1], HWC → NCHW.
        img = cv2.resize(frame_bgr, (320, 240))
        img = (img.astype(np.float32) - 127.0) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        raw_outputs = self._session.run(None, {self._input_name: img})
        confidences = np.asarray(raw_outputs[0])[0]  # (N, 2)
        boxes = np.asarray(raw_outputs[1])[0]  # (N, 4) normalized [0, 1]

        # Filter by face confidence (column 1 = face class).
        scores = confidences[:, 1]
        mask = scores > self._confidence_threshold
        scores = scores[mask]
        boxes = boxes[mask]

        if len(scores) == 0:
            return []

        # Denormalize to original frame coordinates.
        boxes[:, 0] *= orig_w
        boxes[:, 1] *= orig_h
        boxes[:, 2] *= orig_w
        boxes[:, 3] *= orig_h

        keep = _nms(boxes, scores, self._nms_threshold)

        return [
            Detection(
                x1=float(boxes[i, 0]),
                y1=float(boxes[i, 1]),
                x2=float(boxes[i, 2]),
                y2=float(boxes[i, 3]),
                confidence=float(scores[i]),
                landmarks=None,  # UltraFace does not produce landmarks
            )
            for i in keep
        ]
