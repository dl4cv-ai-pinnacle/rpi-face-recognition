"""ArcFace / MobileFaceNet ONNX embedding extractor.

Origin: Valenia src/arcface.py — adapted to accept EmbeddingConfig and import
from unified module paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import onnxruntime as ort

from src.contracts import Float32Array, UInt8Array
from src.onnx_session import suppress_stderr_fd

if TYPE_CHECKING:
    from src.config import EmbeddingConfig


class ArcFaceEmbedder:
    """MobileFaceNet with ArcFace loss — 512-dim L2-normalized embeddings."""

    def __init__(
        self,
        config: EmbeddingConfig,
        providers: list[str] | None = None,
        input_mean: float = 127.5,
        input_std: float = 127.5,
        quiet: bool = True,
    ) -> None:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        with suppress_stderr_fd(enabled=quiet):
            self.session = ort.InferenceSession(
                config.model_path,
                sess_options=so,
                providers=providers or ["CPUExecutionProvider"],
            )
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = (int(input_cfg.shape[3]), int(input_cfg.shape[2]))
        self.output_name = self.session.get_outputs()[0].name
        self._input_mean = input_mean
        self._input_std = input_std
        self._embedding_dim = config.embedding_dim

    @property
    def name(self) -> str:
        return "MobileFaceNet"

    def get_embedding(self, crop_bgr: UInt8Array, /) -> Float32Array:
        """Extract a 512-dim L2-normalized embedding from an aligned 112×112 crop."""
        blob = cv2.dnn.blobFromImage(
            crop_bgr,
            scalefactor=1.0 / self._input_std,
            size=self.input_size,
            mean=(self._input_mean, self._input_mean, self._input_mean),
            swapRB=True,
        )
        raw_output = self.session.run([self.output_name], {self.input_name: blob})
        embedding = np.asarray(raw_output[0], dtype=np.float32).flatten()[: self._embedding_dim]
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return np.asarray(embedding, dtype=np.float32)
