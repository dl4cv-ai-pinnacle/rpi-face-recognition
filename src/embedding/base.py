from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingExtractor(Protocol):
    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embedding from a 112x112 aligned face crop.

        Returns embedding vector of shape (embedding_dim,).
        """
        ...
