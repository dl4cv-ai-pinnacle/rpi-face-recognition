"""Protocols for dependency injection across the pipeline.

Every swappable stage is defined as a @runtime_checkable Protocol so that:
- Concrete implementations are structurally typed (no base classes needed).
- isinstance() checks work for graceful degradation.
- Tests can use lightweight stubs without inheritance.

Protocol origins:
- DetectionLike, PipelineLike, GalleryLike, factories: Valenia contracts.py
- @runtime_checkable decorator pattern: Shalaiev detectors/base.py
- FrameCapture, name properties: Avdieienko capture/base.py, Shalaiev base.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# Shared numpy type aliases — the vocabulary for array shapes across the pipeline.
type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]
type Int32Array = npt.NDArray[np.int32]
type UInt8Array = npt.NDArray[np.uint8]

if TYPE_CHECKING:
    from pathlib import Path

    from src.tracking import Track


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected face with bounding box, confidence, and optional landmarks."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: Float32Array | None = None  # shape (5, 2) or None


@dataclass(frozen=True)
class MatchResult:
    """Result of a gallery match query."""

    name: str | None
    slug: str | None
    score: float
    matched: bool


# ---------------------------------------------------------------------------
# Pipeline stage Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class FrameCapture(Protocol):
    """Camera or frame source."""

    def read(self) -> UInt8Array | None: ...
    def release(self) -> None: ...


@runtime_checkable
class DetectorLike(Protocol):
    """Face detector backend (SCRFD, UltraFace, etc.)."""

    @property
    def name(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def detect(self, frame_bgr: UInt8Array, /) -> list[Detection]: ...


@runtime_checkable
class AlignerLike(Protocol):
    """Face alignment backend (cv2, skimage)."""

    @property
    def name(self) -> str: ...

    def align(self, frame_bgr: UInt8Array, landmarks: Float32Array, /) -> UInt8Array | None: ...


@runtime_checkable
class EmbedderLike(Protocol):
    """Face embedding extractor (ArcFace/MobileFaceNet)."""

    @property
    def name(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def get_embedding(self, crop_bgr: UInt8Array, /) -> Float32Array: ...


@runtime_checkable
class TrackerLike(Protocol):
    """Face tracker: associate detections across frames, predict between frames."""

    def update(
        self,
        boxes: Float32Array,
        kps: Float32Array | None,
        *,
        max_tracks: int | None = None,
    ) -> list[Track]: ...

    def predict(self) -> list[Track]: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Composite pipeline Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class DetectionResultLike(Protocol):
    """Minimal detection result shape shared by runtime components."""

    @property
    def boxes(self) -> Float32Array: ...

    @property
    def kps(self) -> Float32Array | None: ...


@runtime_checkable
class FrameResultLike(Protocol):
    """Minimal frame result shape returned by frame processing."""

    @property
    def boxes(self) -> Float32Array: ...

    @property
    def kps(self) -> Float32Array | None: ...

    @property
    def detect_ms(self) -> float: ...

    @property
    def embed_ms_total(self) -> float: ...


@runtime_checkable
class PipelineLike(Protocol):
    """Composite pipeline: detect + align + embed."""

    @property
    def detector(self) -> DetectorLike: ...

    @property
    def embedder(self) -> EmbedderLike: ...

    def detect(self, frame_bgr: UInt8Array, /) -> tuple[DetectionResultLike, float]: ...

    def embed_from_kps(
        self, frame_bgr: UInt8Array, landmarks: Float32Array, /
    ) -> tuple[Float32Array, float]: ...

    def process_frame(
        self, frame_bgr: UInt8Array, /, max_faces: int | None = None
    ) -> FrameResultLike: ...

    def step(self, frame_bgr: UInt8Array, /, max_faces: int | None = None) -> FrameResultLike: ...


# ---------------------------------------------------------------------------
# Gallery Protocol
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from src.gallery import EnrollmentResult, GalleryMatch, IdentityRecord, UnknownRecord


class GalleryLike(Protocol):
    """Full identity lifecycle: enroll, match, capture unknowns, promote, enrich."""

    def enroll(
        self, name: str, uploads: list[tuple[str, bytes]], pipeline: PipelineLike, /
    ) -> EnrollmentResult: ...

    def enroll_captured(
        self,
        name: str,
        embeddings: list[Float32Array],
        uploads: list[tuple[str, bytes]],
        /,
    ) -> EnrollmentResult: ...

    def match(self, embedding: Float32Array, threshold: float, /) -> GalleryMatch: ...

    def count(self) -> int: ...

    def capture_unknown(self, embedding: Float32Array, crop_bgr: UInt8Array, /) -> GalleryMatch: ...

    def identities(self) -> list[IdentityRecord]: ...

    def unknowns(self) -> list[UnknownRecord]: ...

    def upload_to_identity(
        self, slug: str, uploads: list[tuple[str, bytes]], pipeline: PipelineLike, /
    ) -> EnrollmentResult: ...

    def upload_captured_to_identity(
        self,
        slug: str,
        embeddings: list[Float32Array],
        uploads: list[tuple[str, bytes]],
        /,
    ) -> EnrollmentResult: ...

    def rename_identity(self, slug: str, new_name: str, /) -> IdentityRecord: ...

    def promote_unknown(self, unknown_slug: str, name: str, /) -> EnrollmentResult: ...

    def merge_unknowns(self, target_slug: str, source_slug: str, /) -> UnknownRecord: ...

    def delete_unknown(self, unknown_slug: str, /) -> None: ...

    def read_image(self, kind: str, slug: str, filename: str, /) -> tuple[bytes, str]: ...

    def list_identity_images(self, slug: str, /) -> list[str]: ...

    def delete_identity(self, slug: str, /) -> None: ...

    def delete_identity_sample(self, slug: str, filename: str, /) -> IdentityRecord: ...

    def enrich_identity(
        self,
        slug: str,
        embedding: Float32Array,
        quality: float,
        /,
        *,
        max_samples: int = 48,
        diversity_threshold: float = 0.95,
        crop_bgr: UInt8Array | None = None,
    ) -> bool: ...


# ---------------------------------------------------------------------------
# Factory Protocols (for DI in pipeline construction)
# ---------------------------------------------------------------------------


class DetectorFactory(Protocol):
    """Build a detector from a model path."""

    def __call__(self, model_path: Path, *, det_thresh: float) -> DetectorLike: ...


class EmbedderFactory(Protocol):
    """Build an embedder from a model path."""

    def __call__(self, model_path: Path) -> EmbedderLike: ...


class PipelineFactory(Protocol):
    """Compose detector + aligner + embedder into a pipeline."""

    def __call__(
        self,
        *,
        detector: DetectorLike,
        embedder: EmbedderLike,
        det_size: tuple[int, int],
        max_faces: int,
    ) -> PipelineLike: ...
