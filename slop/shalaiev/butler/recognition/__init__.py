"""Face recognition: embedding extraction and identity matching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from butler import settings

from .base import FaceEmbedder, MatchResult, RecognizedFace
from .database import FaceDatabase

if TYPE_CHECKING:
    pass

__all__ = [
    "FaceEmbedder",
    "FaceDatabase",
    "MatchResult",
    "RecognizedFace",
    "create_embedder",
]


def create_embedder() -> FaceEmbedder:
    """Create a face embedder based on settings.FACE_EMBEDDER."""
    name = settings.FACE_EMBEDDER.lower()
    if name == "mobilefacenet":
        from .mobilefacenet import MobileFaceNetEmbedder

        return MobileFaceNetEmbedder()
    raise ValueError(f"Unknown face embedder: {name!r} (expected 'mobilefacenet')")
