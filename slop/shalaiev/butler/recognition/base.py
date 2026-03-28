"""Face embedder protocol and shared recognition types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from butler.detectors.base import Detection


@dataclass(frozen=True, slots=True)
class RecognizedFace:
    detection: Detection
    identity: str       # name or "Unknown"
    similarity: float   # cosine similarity, 0.0 if unmatched/no landmarks


@dataclass(frozen=True, slots=True)
class MatchResult:
    identity: str
    similarity: float


@runtime_checkable
class FaceEmbedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    def extract(self, frame_rgb: np.ndarray, detection: Detection) -> np.ndarray | None: ...
