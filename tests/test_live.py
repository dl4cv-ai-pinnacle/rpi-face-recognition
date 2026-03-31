"""Tests for LiveRuntime with DI stubs — strided detection, embedding reuse, swap."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from src.config import (
    AlignmentConfig,
    AppConfig,
    CaptureConfig,
    DetectionConfig,
    DisplayConfig,
    EmbeddingConfig,
    GalleryConfig,
    LiveConfig,
    MetricsConfig,
    ServerConfig,
    TrackingConfig,
)
from src.contracts import Float32Array, UInt8Array
from src.gallery import GalleryMatch
from src.live import LiveRuntime

from tests.conftest import StubAligner, StubDetector, StubEmbedder, make_detection, make_frame


def _test_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        capture=CaptureConfig(resolution=(640, 480), format="RGB888"),
        detection=DetectionConfig(backend="stub", confidence_threshold=0.5, nms_threshold=0.4),
        alignment=AlignmentConfig(method="cv2", output_size=112),
        embedding=EmbeddingConfig(
            model_path="fake.onnx", embedding_dim=3, quantize_int8=False
        ),
        tracking=TrackingConfig(iou_threshold=0.3, max_missed=3, smoothing=0.65, method="simple"),
        gallery=GalleryConfig(
            root_dir=str(tmp_path / "gallery"),
            enrich_margin=0.10,
            enrich_min_quality=0.40,
            enrich_cooldown_seconds=30.0,
            enrich_max_samples=48,
        ),
        live=LiveConfig(
            max_faces=3,
            target_fps=15.0,
            det_every=2,
            match_threshold=0.4,
            embed_refresh_frames=30,
            embed_refresh_iou=0.6,
        ),
        server=ServerConfig(host="0.0.0.0", port=8080),
        metrics=MetricsConfig(json_path=None, write_every_frames=30),
        display=DisplayConfig(show_window=False, window_name="Test"),
        log_level="WARNING",
        memory_limit_mb=0.0,
    )


@dataclass
class StubPipeline:
    """Minimal pipeline stub for LiveRuntime tests."""

    detector: StubDetector = field(default_factory=lambda: StubDetector(detections=[]))
    aligner: StubAligner = field(default_factory=StubAligner)
    embedder: StubEmbedder = field(default_factory=StubEmbedder)

    def detect(self, frame_bgr: UInt8Array, /) -> tuple[object, float]:
        from src.pipeline import DetectionResult

        detections = self.detector.detect(frame_bgr)
        if not detections:
            return DetectionResult(
                boxes=np.zeros((0, 5), dtype=np.float32), kps=None
            ), 1.0

        boxes = np.array(
            [[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in detections],
            dtype=np.float32,
        )
        kps_list = [d.landmarks for d in detections if d.landmarks is not None]
        kps = np.stack(kps_list).astype(np.float32) if len(kps_list) == len(detections) else None
        return DetectionResult(boxes=boxes, kps=kps), 1.0

    def embed_from_kps(
        self, frame_bgr: UInt8Array, landmarks: Float32Array, /
    ) -> tuple[Float32Array, float]:
        self.embedder.call_count += 1
        return self.embedder.embedding, 1.0

    def process_frame(
        self, frame_bgr: UInt8Array, /, max_faces: int | None = None
    ) -> object:
        return self.detect(frame_bgr)

    def step(
        self, frame_bgr: UInt8Array, /, max_faces: int | None = None
    ) -> object:
        return self.process_frame(frame_bgr, max_faces)


@dataclass
class StubGallery:
    """Minimal gallery stub."""

    _match_result: GalleryMatch = field(
        default_factory=lambda: GalleryMatch(name=None, slug=None, score=0.0, matched=False)
    )
    match_calls: int = 0

    def match(self, embedding: Float32Array, threshold: float, /) -> GalleryMatch:
        self.match_calls += 1
        return self._match_result

    def count(self) -> int:
        return 0

    def capture_unknown(
        self, embedding: Float32Array, crop_bgr: UInt8Array, /
    ) -> GalleryMatch:
        return GalleryMatch(name="unknown-0001", slug="unknown-0001", score=0.0, matched=False)

    def enrich_identity(self, *args: object, **kwargs: object) -> bool:
        return False

    def identities(self) -> list[object]:
        return []

    def unknowns(self) -> list[object]:
        return []


class TestLiveRuntimeStridedDetection:
    def test_detection_runs_on_first_frame(self, tmp_path: Path) -> None:
        config = _test_config(tmp_path)
        pipeline = StubPipeline(detector=StubDetector(detections=[make_detection()]))
        runtime = LiveRuntime(pipeline=pipeline, gallery=StubGallery(), config=config)

        runtime.process_frame(make_frame())

        assert pipeline.detector.call_count == 1

    def test_detection_skips_second_frame_with_det_every_2(self, tmp_path: Path) -> None:
        config = _test_config(tmp_path)
        pipeline = StubPipeline(detector=StubDetector(detections=[make_detection()]))
        runtime = LiveRuntime(pipeline=pipeline, gallery=StubGallery(), config=config)

        runtime.process_frame(make_frame())  # frame 1 → detect
        runtime.process_frame(make_frame())  # frame 2 → skip

        assert pipeline.detector.call_count == 1

    def test_detection_runs_on_third_frame_with_det_every_2(self, tmp_path: Path) -> None:
        config = _test_config(tmp_path)
        pipeline = StubPipeline(detector=StubDetector(detections=[make_detection()]))
        runtime = LiveRuntime(pipeline=pipeline, gallery=StubGallery(), config=config)

        runtime.process_frame(make_frame())  # frame 1 → detect
        runtime.process_frame(make_frame())  # frame 2 → skip
        runtime.process_frame(make_frame())  # frame 3 → detect

        assert pipeline.detector.call_count == 2


class TestLiveRuntimeSwapPipeline:
    def test_swap_returns_old_pipeline(self, tmp_path: Path) -> None:
        config = _test_config(tmp_path)
        old_pipeline = StubPipeline()
        runtime = LiveRuntime(pipeline=old_pipeline, gallery=StubGallery(), config=config)

        new_pipeline = StubPipeline()
        returned = runtime.swap_pipeline(new_pipeline)

        assert returned is old_pipeline
        assert runtime.pipeline is new_pipeline

    def test_swap_clears_tracker_state(self, tmp_path: Path) -> None:
        config = _test_config(tmp_path)
        pipeline = StubPipeline(detector=StubDetector(detections=[make_detection()]))
        runtime = LiveRuntime(pipeline=pipeline, gallery=StubGallery(), config=config)

        runtime.process_frame(make_frame())
        assert len(runtime._last_tracks) > 0

        runtime.swap_pipeline(StubPipeline())
        assert len(runtime._last_tracks) == 0


class TestLiveRuntimeKalmanMode:
    def _kalman_config(self, tmp_path: Path) -> AppConfig:
        base = _test_config(tmp_path)
        return AppConfig(
            capture=base.capture,
            detection=base.detection,
            alignment=base.alignment,
            embedding=base.embedding,
            tracking=TrackingConfig(
                iou_threshold=0.3, max_missed=3, smoothing=0.65, method="kalman"
            ),
            gallery=base.gallery,
            live=base.live,
            server=base.server,
            metrics=base.metrics,
            display=base.display,
            log_level=base.log_level,
            memory_limit_mb=base.memory_limit_mb,
        )

    def test_kalman_runtime_processes_frames(self, tmp_path: Path) -> None:
        config = self._kalman_config(tmp_path)
        pipeline = StubPipeline(detector=StubDetector(detections=[make_detection()]))
        runtime = LiveRuntime(pipeline=pipeline, gallery=StubGallery(), config=config)

        result = runtime.process_frame(make_frame())
        assert len(result.overlay.faces) == 1

    def test_kalman_held_tracks_use_prediction(self, tmp_path: Path) -> None:
        """With det_every=2, non-detection frames should use Kalman prediction."""
        config = self._kalman_config(tmp_path)
        pipeline = StubPipeline(detector=StubDetector(detections=[make_detection()]))
        runtime = LiveRuntime(pipeline=pipeline, gallery=StubGallery(), config=config)

        # Frame 1: detect (det_every=2 → detect on frame 1).
        runtime.process_frame(make_frame())
        assert pipeline.detector.call_count == 1

        # Frame 2: skip detection, use Kalman prediction.
        result2 = runtime.process_frame(make_frame())
        assert pipeline.detector.call_count == 1  # No new detection.
        # Track should still exist via prediction.
        assert len(result2.overlay.faces) == 1
