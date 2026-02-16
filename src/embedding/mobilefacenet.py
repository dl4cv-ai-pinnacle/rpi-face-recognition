from __future__ import annotations

import logging

import MNN
import numpy as np

from src.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class MobileFaceNetExtractor:
    """MobileFaceNet embedding extractor using MNN inference."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.input_size = config.input_size
        self.embedding_dim = config.embedding_dim

        self._interpreter = MNN.Interpreter(config.model_path)
        self._session = self._interpreter.createSession()
        self._input_tensor = self._interpreter.getSessionInput(self._session)

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
        self._run_session(img)
        embedding = self._get_output()
        return embedding

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] and convert HWC to CHW."""
        img = face.astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return img

    def _run_session(self, img: np.ndarray) -> None:
        """Feed input and run MNN session."""
        tmp_input = MNN.Tensor(
            (1, 3, self.input_size, self.input_size),
            MNN.Halide_Type_Float,
            img.flatten().tolist(),
            MNN.Tensor_DimensionType_Caffe,
        )
        self._input_tensor.copyFrom(tmp_input)
        self._interpreter.runSession(self._session)

    def _get_output(self) -> np.ndarray:
        """Read output tensor and L2-normalize."""
        output_tensor = self._interpreter.getSessionOutput(self._session)
        embedding = np.array(output_tensor.getData(), dtype=np.float32)
        embedding = embedding.flatten()[: self.embedding_dim]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding
