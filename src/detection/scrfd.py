from __future__ import annotations

import logging

import cv2
import MNN
import numpy as np

from src.config import DetectionConfig
from src.detection.base import Detection

logger = logging.getLogger(__name__)

# SCRFD FPN strides and anchors-per-location
_STRIDES = [8, 16, 32]
_NUM_ANCHORS = 2


class SCRFDDetector:
    """SCRFD-500M face detector using MNN inference."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.input_h, self.input_w = config.input_size

        self._interpreter = MNN.Interpreter(config.model_path)
        self._session = self._interpreter.createSession()
        self._input_tensor = self._interpreter.getSessionInput(self._session)

        logger.info(
            "SCRFD loaded: %s (input %dx%d)",
            config.model_path,
            self.input_w,
            self.input_h,
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in an RGB image.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            List of Detection objects with bbox, score, and 5 landmarks.
        """
        img, scale, pad_h, pad_w = self._preprocess(image)
        self._run_session(img)
        raw_dets = self._decode_outputs()
        detections = self._postprocess(raw_dets, scale, pad_h, pad_w, image.shape)
        return detections

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize and normalize.

        Returns:
            (preprocessed CHW float32 array, scale factor, pad_h, pad_w)
        """
        h, w = image.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to input size
        pad_h = (self.input_h - new_h) // 2
        pad_w = (self.input_w - new_w) // 2

        padded = np.full(
            (self.input_h, self.input_w, 3), 0, dtype=np.uint8
        )
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # Normalize: (pixel - 127.5) / 128.0
        img = (padded.astype(np.float32) - 127.5) / 128.0

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        return img, scale, pad_h, pad_w

    def _run_session(self, img: np.ndarray) -> None:
        """Feed input tensor and run MNN session."""
        tmp_input = MNN.Tensor(
            (1, 3, self.input_h, self.input_w),
            MNN.Halide_Type_Float,
            img.flatten().tolist(),
            MNN.Tensor_DimensionType_Caffe,
        )
        self._input_tensor.copyFrom(tmp_input)
        self._interpreter.runSession(self._session)

    def _decode_outputs(self) -> list[np.ndarray]:
        """Decode raw FPN outputs into [x1, y1, x2, y2, score, lm0..lm9].

        Returns:
            List of arrays, one per stride. Each has shape (N, 15).
        """
        all_dets = []

        for idx, stride in enumerate(_STRIDES):
            # Output tensor names follow SCRFD convention:
            #   score: score_8, score_16, score_32
            #   bbox:  bbox_8, bbox_16, bbox_32
            #   kps:   kps_8, kps_16, kps_32
            cls_name = f"score_{stride}"
            bbox_name = f"bbox_{stride}"
            kps_name = f"kps_{stride}"

            cls_tensor = self._interpreter.getSessionOutput(
                self._session, cls_name
            )
            bbox_tensor = self._interpreter.getSessionOutput(
                self._session, bbox_name
            )
            kps_tensor = self._interpreter.getSessionOutput(
                self._session, kps_name
            )

            cls_data = np.array(cls_tensor.getData()).reshape(-1, 1)
            bbox_data = np.array(bbox_tensor.getData()).reshape(-1, 4)
            kps_data = np.array(kps_tensor.getData()).reshape(-1, 10)

            feat_h = self.input_h // stride
            feat_w = self.input_w // stride

            # Generate anchor centers
            anchors = []
            for i in range(feat_h):
                for j in range(feat_w):
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    for _ in range(_NUM_ANCHORS):
                        anchors.append([cx, cy])
            anchors = np.array(anchors, dtype=np.float32)

            # Filter by confidence
            scores = cls_data.flatten()
            mask = scores > self.config.confidence_threshold
            if not np.any(mask):
                continue

            scores = scores[mask]
            bbox_data = bbox_data[mask]
            kps_data = kps_data[mask]
            anchors_sel = anchors[mask]

            # Decode bboxes: distance from anchor
            x1 = anchors_sel[:, 0] - bbox_data[:, 0] * stride
            y1 = anchors_sel[:, 1] - bbox_data[:, 1] * stride
            x2 = anchors_sel[:, 0] + bbox_data[:, 2] * stride
            y2 = anchors_sel[:, 1] + bbox_data[:, 3] * stride

            # Decode landmarks
            lms = np.zeros_like(kps_data)
            for k in range(5):
                lms[:, k * 2] = anchors_sel[:, 0] + kps_data[:, k * 2] * stride
                lms[:, k * 2 + 1] = (
                    anchors_sel[:, 1] + kps_data[:, k * 2 + 1] * stride
                )

            dets = np.column_stack([x1, y1, x2, y2, scores, lms])
            all_dets.append(dets)

        return all_dets

    def _postprocess(
        self,
        raw_dets: list[np.ndarray],
        scale: float,
        pad_h: int,
        pad_w: int,
        orig_shape: tuple[int, ...],
    ) -> list[Detection]:
        """Apply NMS and map coordinates back to original image space."""
        if not raw_dets:
            return []

        dets = np.concatenate(raw_dets, axis=0)

        # NMS
        boxes = dets[:, :4]
        scores = dets[:, 4]

        # Convert to xywh for cv2.dnn.NMSBoxes
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        nms_boxes = list(zip(x1.tolist(), y1.tolist(), w.tolist(), h.tolist()))
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores.tolist(),
            self.config.confidence_threshold,
            self.config.nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        dets = dets[indices]

        # Map back to original image coordinates
        orig_h, orig_w = orig_shape[:2]
        results = []
        for det in dets:
            bbox = det[:4].copy()
            score = float(det[4])
            lms = det[5:].reshape(5, 2).copy()

            # Remove padding offset
            bbox[[0, 2]] -= pad_w
            bbox[[1, 3]] -= pad_h
            lms[:, 0] -= pad_w
            lms[:, 1] -= pad_h

            # Remove scale
            bbox /= scale
            lms /= scale

            # Clip to image bounds
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, orig_w)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, orig_h)

            results.append(
                Detection(
                    bbox=bbox.astype(np.float32),
                    score=score,
                    landmarks=lms.astype(np.float32),
                )
            )

        return results
