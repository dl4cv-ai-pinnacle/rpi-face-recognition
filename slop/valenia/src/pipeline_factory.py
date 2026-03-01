"""Shared helpers for building swappable pipeline variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from contracts import (
    DetectorFactory,
    DetectorLike,
    EmbedderFactory,
    EmbedderLike,
    PipelineFactory,
    PipelineLike,
)


@dataclass(frozen=True)
class PipelineSpec:
    """Complete pipeline wiring for one detector + embedder variant."""

    det_model: Path
    rec_model: Path
    det_size: tuple[int, int]
    det_thresh: float
    max_faces: int


def parse_size(value: str) -> tuple[int, int]:
    """Parse WxH detector input sizes from CLI values."""
    width, height = value.lower().split("x", maxsplit=1)
    return int(width), int(height)


def resolve_project_path(root: Path, path_str: str) -> Path:
    """Resolve project-relative CLI paths against the setup root."""
    path = Path(path_str)
    return path if path.is_absolute() else root / path


def build_face_pipeline(
    spec: PipelineSpec,
    *,
    detector_factory: DetectorFactory | None = None,
    embedder_factory: EmbedderFactory | None = None,
    pipeline_factory: PipelineFactory | None = None,
) -> PipelineLike:
    """Build one pipeline variant with overrideable construction hooks."""
    active_detector_factory = detector_factory or _default_detector_factory
    active_embedder_factory = embedder_factory or _default_embedder_factory
    active_pipeline_factory = pipeline_factory or _default_pipeline_factory

    detector = active_detector_factory(spec.det_model, det_thresh=spec.det_thresh)
    embedder = active_embedder_factory(spec.rec_model)
    return active_pipeline_factory(
        detector=detector,
        embedder=embedder,
        det_size=spec.det_size,
        max_faces=spec.max_faces,
    )


def _default_detector_factory(model_path: Path, *, det_thresh: float) -> DetectorLike:
    from scrfd import SCRFDDetector

    return SCRFDDetector(str(model_path), det_thresh=det_thresh)


def _default_embedder_factory(model_path: Path) -> EmbedderLike:
    from arcface import ArcFaceEmbedder

    return ArcFaceEmbedder(str(model_path))


def _default_pipeline_factory(
    *,
    detector: DetectorLike,
    embedder: EmbedderLike,
    det_size: tuple[int, int],
    max_faces: int,
) -> PipelineLike:
    from pipeline import FacePipeline

    return FacePipeline(
        detector=detector,
        embedder=embedder,
        det_size=det_size,
        max_faces=max_faces,
    )
