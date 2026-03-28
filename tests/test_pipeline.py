"""Tests for FacePipeline with DI stubs — verifies the Protocol chain works."""

from __future__ import annotations

import pytest
from src.pipeline import FacePipeline

from tests.conftest import (
    StubAligner,
    StubDetector,
    StubEmbedder,
    make_detection,
    make_frame,
)


class TestPipelineDetection:
    def test_returns_empty_boxes_when_no_faces(self) -> None:
        pipeline = FacePipeline(
            detector=StubDetector(detections=[]),
            aligner=StubAligner(),
            embedder=StubEmbedder(),
        )

        result = pipeline.step(make_frame())

        assert len(result.boxes) == 0
        assert result.kps is None
        assert result.embeddings == []
        assert result.detect_ms >= 0

    def test_returns_boxes_for_detected_faces(self) -> None:
        det = make_detection(with_landmarks=True)
        pipeline = FacePipeline(
            detector=StubDetector(detections=[det]),
            aligner=StubAligner(),
            embedder=StubEmbedder(),
        )

        result = pipeline.step(make_frame())

        assert len(result.boxes) == 1
        assert result.boxes[0][4] == pytest.approx(0.92)

    def test_detect_ms_is_positive(self) -> None:
        pipeline = FacePipeline(
            detector=StubDetector(detections=[make_detection()]),
            aligner=StubAligner(),
            embedder=StubEmbedder(),
        )

        result = pipeline.step(make_frame())

        assert result.detect_ms > 0


class TestPipelineEmbedding:
    def test_produces_one_embedding_per_face(self) -> None:
        dets = [make_detection(), make_detection()]
        pipeline = FacePipeline(
            detector=StubDetector(detections=dets),
            aligner=StubAligner(),
            embedder=StubEmbedder(),
        )

        result = pipeline.step(make_frame())

        assert len(result.embeddings) == 2

    def test_respects_max_faces_limit(self) -> None:
        dets = [make_detection() for _ in range(5)]
        pipeline = FacePipeline(
            detector=StubDetector(detections=dets),
            aligner=StubAligner(),
            embedder=StubEmbedder(),
            max_faces=2,
        )

        result = pipeline.step(make_frame())

        assert len(result.embeddings) == 2

    def test_calls_aligner_and_embedder(self) -> None:
        aligner = StubAligner()
        embedder = StubEmbedder()
        pipeline = FacePipeline(
            detector=StubDetector(detections=[make_detection()]),
            aligner=aligner,
            embedder=embedder,
        )

        pipeline.step(make_frame())

        assert aligner.call_count == 1
        assert embedder.call_count == 1


class TestPipelineLandmarkFallback:
    def test_embeds_via_center_crop_when_no_landmarks(self) -> None:
        det = make_detection(with_landmarks=False)
        embedder = StubEmbedder()
        pipeline = FacePipeline(
            detector=StubDetector(detections=[det]),
            aligner=StubAligner(),
            embedder=embedder,
        )

        result = pipeline.step(make_frame())

        assert len(result.embeddings) == 1
        assert embedder.call_count == 1
        assert result.kps is None


class TestStepIsAliasForProcessFrame:
    def test_step_and_process_frame_return_same_structure(self) -> None:
        pipeline = FacePipeline(
            detector=StubDetector(detections=[make_detection()]),
            aligner=StubAligner(),
            embedder=StubEmbedder(),
        )
        frame = make_frame()

        step_result = pipeline.step(frame)
        process_result = pipeline.process_frame(frame)

        assert type(step_result) is type(process_result)

