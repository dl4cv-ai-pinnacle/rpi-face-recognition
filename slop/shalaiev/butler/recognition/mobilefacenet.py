"""MobileFaceNet face embedding extractor using ONNX Runtime."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from butler import settings
from butler.detectors.base import Detection
from .align import align_face


class MobileFaceNetEmbedder:
    """MobileFaceNet (w600k_mbf) embedder loaded from the buffalo_sc model pack."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None

    def _ensure_session(self) -> ort.InferenceSession:
        if self._session is None:
            model_path = settings.MODELS_DIR / "models" / "buffalo_sc" / "w600k_mbf.onnx"
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        return self._session

    @property
    def name(self) -> str:
        return "MobileFaceNet"

    @property
    def embedding_dim(self) -> int:
        return 512

    def extract(self, frame_rgb: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Align face, run inference, return L2-normalized 512-dim embedding."""
        if detection.landmarks is None:
            return None

        aligned = align_face(frame_rgb, detection.landmarks)
        if aligned is None:
            return None

        # Preprocess: RGB → BGR, normalize to [-1, 1], CHW, add batch dim.
        bgr = aligned[:, :, ::-1].copy()
        blob = (bgr.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 112, 112)

        session = self._ensure_session()
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: blob})[0]  # (1, 512)

        embedding = output.flatten()
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return None
        return embedding / norm
