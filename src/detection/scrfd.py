from __future__ import annotations

import logging

import cv2
import numpy as np
import onnxruntime as ort

from src.config import DetectionConfig
from src.detection.base import Detection

logger = logging.getLogger(__name__)

# SCRFD FPN strides
_STRIDES = [8, 16, 32]
_FMC = 3  # feature map count (number of FPN levels)
_NUM_ANCHORS = 2


class SCRFDDetector:
    """SCRFD-500M face detector using ONNX Runtime inference."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.input_h, self.input_w = config.input_size

        self._session = ort.InferenceSession(
            config.model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Detect model variant: 6 outputs = no keypoints, 9 = with keypoints
        self._use_kps = len(self._output_names) == 9

        logger.info(
            "SCRFD loaded: %s (input %dx%d, %d outputs, kps=%s)",
            config.model_path,
            self.input_w,
            self.input_h,
            len(self._output_names),
            self._use_kps,
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in an RGB image.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            List of Detection objects with bbox, score, and 5 landmarks.
        """
        img, scale, pad_h, pad_w = self._preprocess(image)
        outputs = self._run_session(img)
        raw_dets = self._decode_outputs(outputs)
        detections = self._postprocess(raw_dets, scale, pad_h, pad_w, image.shape)
        return detections

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize and normalize."""
        h, w = image.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        pad_h = (self.input_h - new_h) // 2
        pad_w = (self.input_w - new_w) // 2

        padded = np.full(
            (self.input_h, self.input_w, 3), 0, dtype=np.uint8
        )
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        img = (padded.astype(np.float32) - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]

        return img, scale, pad_h, pad_w

    def _run_session(self, img: np.ndarray) -> list[np.ndarray]:
        """Run ONNX Runtime inference. Returns list of output arrays."""
        return self._session.run(
            self._output_names,
            {self._input_name: img.astype(np.float32)},
        )

    def _decode_outputs(self, net_outs: list[np.ndarray]) -> list[np.ndarray]:
        """Decode raw FPN outputs into [x1, y1, x2, y2, score, lm0..lm9].

        Output ordering (InsightFace convention):
          6 outputs: [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32]
          9 outputs: [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
        """
        all_dets = []

        for idx, stride in enumerate(_STRIDES):
            scores = net_outs[idx].reshape(-1)
            bbox_raw = net_outs[idx + _FMC].reshape(-1, 4)

            # Pre-multiply by stride (official InsightFace convention)
            bbox_preds = bbox_raw * stride

            if self._use_kps:
                kps_raw = net_outs[idx + _FMC * 2].reshape(-1, 10)
                kps_preds = kps_raw * stride
            else:
                kps_preds = None

            feat_h = self.input_h // stride
            feat_w = self.input_w // stride

            # Generate anchor centers (no +0.5 offset, per official code)
            anchor_centers = np.stack(
                np.mgrid[:feat_h, :feat_w][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)
            if _NUM_ANCHORS > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * _NUM_ANCHORS, axis=1
                ).reshape(-1, 2)

            # Filter by confidence
            mask = scores > self.config.confidence_threshold
            if not np.any(mask):
                continue

            scores_sel = scores[mask]
            bbox_sel = bbox_preds[mask]
            anchors_sel = anchor_centers[mask]

            # Decode bboxes: distance from anchor (already scaled by stride)
            x1 = anchors_sel[:, 0] - bbox_sel[:, 0]
            y1 = anchors_sel[:, 1] - bbox_sel[:, 1]
            x2 = anchors_sel[:, 0] + bbox_sel[:, 2]
            y2 = anchors_sel[:, 1] + bbox_sel[:, 3]

            # Decode landmarks
            if kps_preds is not None:
                kps_sel = kps_preds[mask]
                lms = np.zeros_like(kps_sel)
                for k in range(5):
                    lms[:, k * 2] = anchors_sel[:, 0] + kps_sel[:, k * 2]
                    lms[:, k * 2 + 1] = anchors_sel[:, 1] + kps_sel[:, k * 2 + 1]
            else:
                # No keypoints: estimate from bbox center
                n = len(scores_sel)
                lms = self._estimate_landmarks(x1, y1, x2, y2, n)

            dets = np.column_stack([x1, y1, x2, y2, scores_sel, lms])
            all_dets.append(dets)

        return all_dets

    @staticmethod
    def _estimate_landmarks(
        x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, n: int
    ) -> np.ndarray:
        """Estimate 5-point landmarks from bounding box when model has no kps output.

        Uses standard face proportions relative to the bounding box.
        """
        w = x2 - x1
        h = y2 - y1
        lms = np.zeros((n, 10), dtype=np.float32)
        # left eye
        lms[:, 0] = x1 + w * 0.3
        lms[:, 1] = y1 + h * 0.35
        # right eye
        lms[:, 2] = x1 + w * 0.7
        lms[:, 3] = y1 + h * 0.35
        # nose
        lms[:, 4] = x1 + w * 0.5
        lms[:, 5] = y1 + h * 0.55
        # left mouth
        lms[:, 6] = x1 + w * 0.35
        lms[:, 7] = y1 + h * 0.75
        # right mouth
        lms[:, 8] = x1 + w * 0.65
        lms[:, 9] = y1 + h * 0.75
        return lms

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

        boxes = dets[:, :4]
        scores = dets[:, 4]

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

        orig_h, orig_w = orig_shape[:2]
        results = []
        for det in dets:
            bbox = det[:4].copy()
            score = float(det[4])
            lms = det[5:].reshape(5, 2).copy()

            bbox[[0, 2]] -= pad_w
            bbox[[1, 3]] -= pad_h
            lms[:, 0] -= pad_w
            lms[:, 1] -= pad_h

            bbox /= scale
            lms /= scale

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
