"""Face embedding backends with factory dispatch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import EmbeddingConfig
    from src.contracts import EmbedderLike

logger = logging.getLogger(__name__)


def create_embedder(config: EmbeddingConfig) -> EmbedderLike:
    """Create an embedder backend based on config."""
    from src.embedding.arcface import ArcFaceEmbedder

    return ArcFaceEmbedder(config)


def create_embedder_safe(config: EmbeddingConfig) -> EmbedderLike | None:
    """Create an embedder, returning None on failure instead of raising."""
    try:
        return create_embedder(config)
    except Exception:
        logger.warning("Failed to load embedder", exc_info=True)
        return None
