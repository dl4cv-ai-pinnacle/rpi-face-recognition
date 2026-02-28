"""Minimal SCRFD ONNX detector for CPU inference on Raspberry Pi."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from runtime_utils import Float32Array, Int32Array, UInt8Array, suppress_stderr_fd


def _distance2bbox(points: Float32Array, distance: Float32Array) -> Float32Array:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.asarray(np.stack([x1, y1, x2, y2], axis=-1), dtype=np.float32)


def _distance2kps(points: Float32Array, distance: Float32Array) -> Float32Array:
    preds: list[Float32Array] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.asarray(np.stack(preds, axis=-1), dtype=np.float32)


@dataclass
class DetectionResult:
    boxes: Float32Array
    kps: Float32Array | None


class SCRFDDetector:
    """SCRFD ONNX wrapper adapted from InsightFace model-zoo implementation."""

    def __init__(
        self,
        model_path: str,
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        providers: list[str] | None = None,
        quiet: bool = True,
    ) -> None:
        so = ort.SessionOptions()
        so.log_severity_level = 3  # suppress INFO/WARN noise from ORT
        with suppress_stderr_fd(enabled=quiet):
            self.session = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=providers or ["CPUExecutionProvider"],
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = None if isinstance(input_shape[2], str) else tuple(input_shape[2:4][::-1])
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.input_mean = 127.5
        self.input_std = 128.0
        self.center_cache: dict[tuple[int, int, int], Float32Array] = {}
        self._init_heads()

    def _init_heads(self) -> None:
        outputs = self.session.get_outputs()
        self.use_kps = False
        self.num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self.feat_stride_fpn = [8, 16, 32]
            self.num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self.feat_stride_fpn = [8, 16, 32]
            self.num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
            self.use_kps = True
        else:
            raise RuntimeError(f"Unexpected SCRFD output count: {len(outputs)}")

    def _nms(self, dets: Float32Array) -> Int32Array:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return np.asarray(keep, dtype=np.int32)

    def _forward(
        self, image: UInt8Array, score_thresh: float
    ) -> tuple[list[Float32Array], list[Float32Array], list[Float32Array]]:
        scores_list: list[Float32Array] = []
        bboxes_list: list[Float32Array] = []
        kpss_list: list[Float32Array] = []

        input_size = (int(image.shape[1]), int(image.shape[0]))
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(self.feat_stride_fpn):
            scores = np.asarray(net_outs[idx].reshape(-1), dtype=np.float32)
            bbox_preds = (
                np.asarray(net_outs[idx + self.fmc].reshape(-1, 4), dtype=np.float32) * stride
            )
            kps_preds: Float32Array | None = None
            if self.use_kps:
                kps_preds = (
                    np.asarray(net_outs[idx + self.fmc * 2].reshape(-1, 10), dtype=np.float32)
                    * stride
                )

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                grid_y, grid_x = np.mgrid[:height, :width]
                anchor_centers = np.asarray(np.stack([grid_x, grid_y], axis=-1), dtype=np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self.num_anchors > 1:
                    anchor_centers = np.asarray(
                        np.repeat(anchor_centers, self.num_anchors, axis=0),
                        dtype=np.float32,
                    )
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= score_thresh)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(np.asarray(scores[pos_inds], dtype=np.float32))
            bboxes_list.append(np.asarray(bboxes[pos_inds], dtype=np.float32))
            if self.use_kps and kps_preds is not None:
                kpss = np.asarray(_distance2kps(anchor_centers, kps_preds).reshape((-1, 5, 2)))
                kpss_list.append(np.asarray(kpss[pos_inds], dtype=np.float32))

        return scores_list, bboxes_list, kpss_list

    def detect(self, image: UInt8Array, input_size: tuple[int, int]) -> DetectionResult:
        image_h, image_w = image.shape[:2]
        model_w, model_h = input_size

        image_ratio = float(image_h) / image_w
        model_ratio = float(model_h) / model_w
        if image_ratio > model_ratio:
            new_h = model_h
            new_w = int(new_h / image_ratio)
        else:
            new_w = model_w
            new_h = int(new_w * image_ratio)

        det_scale = float(new_h) / image_h
        resized = cv2.resize(image, (new_w, new_h))
        det_img = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        scores_list, bboxes_list, kpss_list = self._forward(det_img, self.det_thresh)
        scores = np.asarray(np.hstack(scores_list), dtype=np.float32)
        order = scores.argsort()[::-1]
        bboxes = np.asarray(np.vstack(bboxes_list) / det_scale, dtype=np.float32)
        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det)
        boxes = np.asarray(pre_det[keep, :], dtype=np.float32)

        kps: Float32Array | None = None
        if self.use_kps and kpss_list:
            all_kps = np.asarray(np.vstack(kpss_list) / det_scale, dtype=np.float32)
            all_kps = all_kps[order, :, :]
            kps = np.asarray(all_kps[keep, :, :], dtype=np.float32)

        return DetectionResult(boxes=boxes, kps=kps)
