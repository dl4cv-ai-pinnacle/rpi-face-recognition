"""Lightweight protocols for dependency injection across the runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from runtime_utils import Float32Array, UInt8Array

if TYPE_CHECKING:
    from gallery import EnrollmentResult, GalleryMatch


class DetectionLike(Protocol):
    """Minimal detection result shape shared by runtime components."""

    @property
    def boxes(self) -> Float32Array:
        """Detection boxes with confidence scores."""
        ...

    @property
    def kps(self) -> Float32Array | None:
        """Optional facial landmarks."""
        ...


class FrameResultLike(Protocol):
    """Minimal frame result shape returned by frame processing."""

    @property
    def boxes(self) -> Float32Array:
        """Detection boxes with confidence scores."""
        ...

    @property
    def kps(self) -> Float32Array | None:
        """Optional facial landmarks."""
        ...

    @property
    def detect_ms(self) -> float:
        """Detector latency in milliseconds."""
        ...

    @property
    def embed_ms_total(self) -> float:
        """Total embedding latency for the frame."""
        ...


class PipelineLike(Protocol):
    """Minimal pipeline contract used by runtime and tests."""

    def detect(self, frame_bgr: UInt8Array, /) -> tuple[DetectionLike, float]:
        """Return face detections and detector latency."""
        ...

    def embed_from_kps(
        self,
        frame_bgr: UInt8Array,
        landmarks: Float32Array,
        /,
    ) -> tuple[Float32Array, float]:
        """Return one embedding and embedder latency."""
        ...

    def process_frame(
        self,
        frame_bgr: UInt8Array,
        /,
        max_faces: int | None = None,
    ) -> FrameResultLike:
        """Return the full frame pipeline output."""
        ...


class GalleryLike(Protocol):
    """Minimal gallery contract used by the live runtime."""

    def enroll(
        self,
        name: str,
        uploads: list[tuple[str, bytes]],
        pipeline: PipelineLike,
        /,
    ) -> EnrollmentResult:
        """Enroll a new identity."""
        ...

    def match(self, embedding: Float32Array, threshold: float, /) -> GalleryMatch:
        """Return the best gallery match."""
        ...

    def count(self) -> int:
        """Return the number of enrolled identities."""
        ...


class DetectorLike(Protocol):
    """Minimal detector object passed into FacePipeline."""

    def detect(
        self,
        frame_bgr: UInt8Array,
        input_size: tuple[int, int],
        /,
    ) -> DetectionLike:
        """Run the face detector."""
        ...


class EmbedderLike(Protocol):
    """Minimal embedder object passed into FacePipeline."""

    def get_embedding(self, crop_bgr: UInt8Array, /) -> Float32Array:
        """Return one normalized embedding."""
        ...


class DetectorFactory(Protocol):
    """Build a detector instance from a model path."""

    def __call__(self, model_path: Path, *, det_thresh: float) -> DetectorLike:
        """Create a detector."""
        ...


class EmbedderFactory(Protocol):
    """Build an embedder instance from a model path."""

    def __call__(self, model_path: Path) -> EmbedderLike:
        """Create an embedder."""
        ...


class PipelineFactory(Protocol):
    """Compose detector and embedder into a pipeline implementation."""

    def __call__(
        self,
        *,
        detector: DetectorLike,
        embedder: EmbedderLike,
        det_size: tuple[int, int],
        max_faces: int,
    ) -> PipelineLike:
        """Create a pipeline object."""
        ...
