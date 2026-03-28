"""Tests for Protocol conformance via @runtime_checkable."""

from __future__ import annotations

from src.alignment import Cv2Aligner, SkimageAligner
from src.contracts import AlignerLike, DetectorLike, EmbedderLike, PipelineLike
from src.pipeline import FacePipeline

from tests.conftest import StubAligner, StubDetector, StubEmbedder


class TestProtocolConformance:
    def test_stub_detector_satisfies_protocol(self) -> None:
        assert isinstance(StubDetector(), DetectorLike)

    def test_stub_aligner_satisfies_protocol(self) -> None:
        assert isinstance(StubAligner(), AlignerLike)

    def test_stub_embedder_satisfies_protocol(self) -> None:
        assert isinstance(StubEmbedder(), EmbedderLike)

    def test_cv2_aligner_satisfies_protocol(self) -> None:
        assert isinstance(Cv2Aligner(), AlignerLike)

    def test_skimage_aligner_satisfies_protocol(self) -> None:
        assert isinstance(SkimageAligner(), AlignerLike)

    def test_face_pipeline_satisfies_protocol(self) -> None:
        pipeline = FacePipeline(
            detector=StubDetector(),
            aligner=StubAligner(),
            embedder=StubEmbedder(),
        )
        assert isinstance(pipeline, PipelineLike)
