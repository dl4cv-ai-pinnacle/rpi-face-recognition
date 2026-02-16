from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

from src.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class MobileFaceNetExtractor:
    """MobileFaceNet embedding extractor using ONNX Runtime inference."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.input_size = config.input_size
        self.embedding_dim = config.embedding_dim

        self._session = ort.InferenceSession(
            config.model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        logger.info(
            "MobileFaceNet loaded: %s (dim=%d)",
            config.model_path,
            config.embedding_dim,
        )

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embedding from a 112x112 aligned face.

        Args:
            aligned_face: RGB image (112, 112, 3) uint8.

        Returns:
            L2-normalized embedding, shape (embedding_dim,).
        """
        img = self._preprocess(aligned_face)
        embedding = self._run_session(img)
        return embedding

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] and convert HWC to NCHW."""
        img = face.astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img[np.newaxis, ...]  # add batch dim -> NCHW
        return img

    def _run_session(self, img: np.ndarray) -> np.ndarray:
        """Run inference and return L2-normalized embedding."""
        output = self._session.run(
            [self._output_name],
            {self._input_name: img.astype(np.float32)},
        )[0]

        embedding = output.flatten()[: self.embedding_dim]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding
