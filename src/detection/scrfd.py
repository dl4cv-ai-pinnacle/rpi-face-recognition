"""Standalone SCRFD face detector via ONNX Runtime.

Loads any SCRFD KPS model (500M, 2.5G, 10G) directly without the insightface
library. This gives full control over the ONNX Runtime session for quantization,
precision, and provider selection.

Preprocessing and postprocessing replicate the insightface reference:
  insightface/model_zoo/scrfd.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from src.contracts import Detection
from src.onnx_session import suppress_stderr_fd

from ._base import non_maximum_suppression

if TYPE_CHECKING:
    from src.config import DetectionConfig
    from src.contracts import Float32Array, UInt8Array

# SCRFD FPN strides — same for all variants (500M, 2.5G, 10G).
_FPN_STRIDES: tuple[int, ...] = (8, 16, 32)
# All distributed SCRFD KPS models use 2 anchors per FPN cell.
_NUM_ANCHORS: int = 2


def _build_anchor_centers(
    input_h: int,
    input_w: int,
    stride: int,
) -> npt.NDArray[np.float32]:
    """Generate anchor center coordinates for one FPN level."""
    h = input_h // stride
    w = input_w // stride
    # mgrid[:h, :w] produces (row, col); reverse to (x, y).
    grid = list(np.mgrid[:h, :w][::-1])
    centers = np.stack(grid, axis=-1).astype(np.float32)
    centers = (centers * stride).reshape(-1, 2)
    # Duplicate for num_anchors per cell.
    centers = np.stack([centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)
    return centers


def _distance2bbox(
    points: npt.NDArray[np.float32], distance: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Decode anchor-relative distances to absolute bounding boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(
    points: npt.NDArray[np.float32], distance: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Decode anchor-relative distances to absolute keypoint coordinates."""
    kps = distance.copy()
    for i in range(0, 10, 2):
        kps[:, i] += points[:, 0]
        kps[:, i + 1] += points[:, 1]
    return kps


class SCRFDDetector:
    """Standalone SCRFD detector — loads any SCRFD KPS ONNX model directly."""

    def __init__(
        self,
        config: DetectionConfig,
        model_dir: str = "models",
    ) -> None:
        self._confidence_threshold = config.confidence_threshold
        self._nms_threshold = config.nms_threshold

        model_path = Path(config.model_path) if config.model_path else None
        if model_path is None:
            model_path = Path(model_dir) / "buffalo_sc" / "det_500m.onnx"
        if not model_path.exists():
            msg = f"SCRFD model not found at {model_path}. Run: bash scripts/download_models.sh"
            raise FileNotFoundError(msg)

        so = ort.SessionOptions()
        so.log_severity_level = 3
        with suppress_stderr_fd():
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )

        self._input_name: str = self._session.get_inputs()[0].name
        self._output_names: list[str] = [o.name for o in self._session.get_outputs()]
        self._model_name = model_path.stem

        # Default input size — SCRFD accepts dynamic shapes,
        # 640x640 is the standard evaluation resolution.
        self._input_h = 640
        self._input_w = 640

        # Pre-compute anchor centers for each FPN level (reused every frame).
        self._anchor_centers: dict[int, npt.NDArray[np.float32]] = {
            stride: _build_anchor_centers(self._input_h, self._input_w, stride)
            for stride in _FPN_STRIDES
        }

    @property
    def name(self) -> str:
        return f"SCRFD ({self._model_name})"

    @property
    def provider_name(self) -> str:
        providers = self._session.get_providers()
        return providers[0] if providers else "CPUExecutionProvider"

    def _preprocess(
        self,
        frame_bgr: UInt8Array,
    ) -> tuple[npt.NDArray[np.float32], float]:
        """Aspect-ratio resize + zero-pad + normalize. Returns blob and scale."""
        orig_h, orig_w = frame_bgr.shape[:2]
        im_ratio = orig_h / orig_w
        model_ratio = self._input_h / self._input_w

        if im_ratio > model_ratio:
            new_h = self._input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = self._input_w
            new_h = int(new_w * im_ratio)

        det_scale = orig_h / new_h
        resized = cv2.resize(frame_bgr, (new_w, new_h))

        det_img = np.zeros((self._input_h, self._input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        blob: npt.NDArray[np.float32] = cv2.dnn.blobFromImage(
            det_img,
            scalefactor=1.0 / 128.0,
            size=(self._input_w, self._input_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob, det_scale

    def detect(self, frame_bgr: UInt8Array, /) -> list[Detection]:
        blob, det_scale = self._preprocess(frame_bgr)
        raw = self._session.run(self._output_names, {self._input_name: blob})

        # SCRFD KPS models output 9 tensors: 3 levels × (scores, bboxes, kps).
        num_levels = len(_FPN_STRIDES)
        all_scores: list[npt.NDArray[np.float32]] = []
        all_boxes: list[npt.NDArray[np.float32]] = []
        all_kps: list[npt.NDArray[np.float32]] = []

        for idx, stride in enumerate(_FPN_STRIDES):
            scores_raw = np.asarray(raw[idx]).ravel()
            bbox_preds = np.asarray(raw[idx + num_levels]) * stride
            kps_preds = np.asarray(raw[idx + 2 * num_levels]) * stride

            mask = scores_raw >= self._confidence_threshold
            if not np.any(mask):
                continue

            anchors = self._anchor_centers[stride][mask]
            scores_filtered = scores_raw[mask]
            bbox_preds = bbox_preds[mask]
            kps_preds = kps_preds[mask]

            boxes = _distance2bbox(anchors, bbox_preds)
            kps = _distance2kps(anchors, kps_preds)

            all_scores.append(scores_filtered)
            all_boxes.append(boxes)
            all_kps.append(kps)

        if not all_scores:
            return []

        scores_cat = np.concatenate(all_scores)
        boxes_cat = np.concatenate(all_boxes)
        kps_cat = np.concatenate(all_kps)

        keep = non_maximum_suppression(boxes_cat, scores_cat, self._nms_threshold)

        detections: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = boxes_cat[i] * det_scale
            landmarks: Float32Array = (kps_cat[i].reshape(5, 2) * det_scale).astype(np.float32)
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(scores_cat[i]),
                    landmarks=landmarks,
                ),
            )
        return detections
