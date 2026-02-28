"""MobileFaceNet/ArcFace ONNX embedding helper."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from runtime_utils import Float32Array, UInt8Array, suppress_stderr_fd


class ArcFaceEmbedder:
    def __init__(
        self,
        model_path: str,
        providers: list[str] | None = None,
        input_mean: float = 127.5,
        input_std: float = 127.5,
        quiet: bool = True,
    ) -> None:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        with suppress_stderr_fd(enabled=quiet):
            self.session = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=providers or ["CPUExecutionProvider"],
            )
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = (int(input_cfg.shape[3]), int(input_cfg.shape[2]))
        self.output_name = self.session.get_outputs()[0].name
        self.input_mean = input_mean
        self.input_std = input_std

    def get_embedding(self, face_bgr: UInt8Array) -> Float32Array:
        blob = cv2.dnn.blobFromImage(
            face_bgr,
            scalefactor=1.0 / self.input_std,
            size=self.input_size,
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        embedding = np.asarray(
            self.session.run([self.output_name], {self.input_name: blob})[0][0],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return np.asarray(embedding, dtype=np.float32)
