from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from contracts import DetectionLike, FrameResultLike
from pipeline_factory import PipelineSpec, build_face_pipeline, parse_size, resolve_project_path
from runtime_utils import Float32Array, UInt8Array


@dataclass(frozen=True)
class FakeDetector:
    model_path: Path
    det_thresh: float

    def detect(
        self,
        frame_bgr: UInt8Array,
        input_size: tuple[int, int],
    ) -> DetectionLike:
        del frame_bgr, input_size
        return FakeDetectionResult(boxes=np.zeros((0, 5), dtype=np.float32), kps=None)


@dataclass(frozen=True)
class FakeEmbedder:
    model_path: Path

    def get_embedding(self, crop_bgr: UInt8Array) -> Float32Array:
        del crop_bgr
        return np.zeros((3,), dtype=np.float32)


@dataclass(frozen=True)
class FakeDetectionResult:
    boxes: Float32Array
    kps: Float32Array | None


@dataclass(frozen=True)
class FakeFrameResult:
    boxes: Float32Array
    kps: Float32Array | None
    detect_ms: float
    embed_ms_total: float


@dataclass(frozen=True)
class BuiltPipeline:
    detector: object
    embedder: object
    det_size: tuple[int, int]
    max_faces: int

    def detect(self, frame_bgr: UInt8Array) -> tuple[DetectionLike, float]:
        del frame_bgr
        return FakeDetectionResult(boxes=np.zeros((0, 5), dtype=np.float32), kps=None), 0.0

    def embed_from_kps(
        self,
        frame_bgr: UInt8Array,
        landmarks: Float32Array,
    ) -> tuple[Float32Array, float]:
        del frame_bgr, landmarks
        return np.zeros((3,), dtype=np.float32), 0.0

    def process_frame(
        self,
        frame_bgr: UInt8Array,
        max_faces: int | None = None,
    ) -> FrameResultLike:
        del frame_bgr, max_faces
        return FakeFrameResult(
            boxes=np.zeros((0, 5), dtype=np.float32),
            kps=None,
            detect_ms=0.0,
            embed_ms_total=0.0,
        )


def test_parse_size_parses_wxh_values() -> None:
    assert parse_size("640x480") == (640, 480)


def test_resolve_project_path_handles_relative_and_absolute_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    relative = resolve_project_path(root, "models/test.onnx")
    absolute = resolve_project_path(root, str(tmp_path / "x.onnx"))

    assert relative == root / "models/test.onnx"
    assert absolute == tmp_path / "x.onnx"


def test_build_face_pipeline_supports_injected_factories() -> None:
    spec = PipelineSpec(
        det_model=Path("det.onnx"),
        rec_model=Path("rec.onnx"),
        det_size=(320, 320),
        det_thresh=0.6,
        max_faces=2,
    )

    def detector_factory(model_path: Path, *, det_thresh: float) -> FakeDetector:
        return FakeDetector(model_path=model_path, det_thresh=det_thresh)

    def embedder_factory(model_path: Path) -> FakeEmbedder:
        return FakeEmbedder(model_path=model_path)

    def pipeline_builder(
        *,
        detector: object,
        embedder: object,
        det_size: tuple[int, int],
        max_faces: int,
    ) -> BuiltPipeline:
        return BuiltPipeline(
            detector=detector,
            embedder=embedder,
            det_size=det_size,
            max_faces=max_faces,
        )

    pipeline = cast(
        BuiltPipeline,
        build_face_pipeline(
            spec,
            detector_factory=detector_factory,
            embedder_factory=embedder_factory,
            pipeline_factory=pipeline_builder,
        ),
    )

    assert isinstance(pipeline.detector, FakeDetector)
    assert pipeline.detector.model_path == Path("det.onnx")
    assert pipeline.detector.det_thresh == 0.6
    assert isinstance(pipeline.embedder, FakeEmbedder)
    assert pipeline.embedder.model_path == Path("rec.onnx")
    assert pipeline.det_size == (320, 320)
    assert pipeline.max_faces == 2
