from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from contracts import DetectionLike, FrameResultLike
from gallery import EnrollmentResult, GalleryMatch, IdentityRecord, UnknownRecord
from live_runtime import LiveRuntime, LiveRuntimeConfig
from runtime_utils import Float32Array, MemoryStats, UInt8Array


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


class StubPipeline:
    def __init__(self, detections: list[FakeDetectionResult]) -> None:
        self._detections = detections
        self.detect_calls = 0
        self.embed_calls = 0

    def detect(self, frame_bgr: UInt8Array) -> tuple[DetectionLike, float]:
        del frame_bgr
        idx = min(self.detect_calls, len(self._detections) - 1)
        self.detect_calls += 1
        return self._detections[idx], 4.0

    def embed_from_kps(
        self,
        frame_bgr: UInt8Array,
        landmarks: Float32Array,
    ) -> tuple[Float32Array, float]:
        del frame_bgr
        self.embed_calls += 1
        seed = float(np.sum(landmarks))
        embedding = np.asarray([seed, 1.0, 0.0], dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        return np.asarray(embedding / norm, dtype=np.float32), 2.0

    def process_frame(
        self,
        frame_bgr: UInt8Array,
        max_faces: int | None = None,
    ) -> FrameResultLike:
        det, detect_ms = self.detect(frame_bgr)
        embed_ms_total = 0.0
        if det.kps is not None:
            face_limit = len(det.boxes) if max_faces is None else min(max_faces, len(det.boxes))
            for idx in range(face_limit):
                _, embed_latency_ms = self.embed_from_kps(frame_bgr, det.kps[idx])
                embed_ms_total += embed_latency_ms
        return FakeFrameResult(
            boxes=det.boxes,
            kps=det.kps,
            detect_ms=detect_ms,
            embed_ms_total=embed_ms_total,
        )


class StubGallery:
    def __init__(
        self,
        *,
        count_value: int = 1,
        match_name: str | None = "alice",
        match_slug: str | None = "alice",
        matched: bool = True,
    ) -> None:
        self._count_value = count_value
        self._match_name = match_name
        self._match_slug = match_slug
        self._matched = matched
        self.match_calls = 0
        self.capture_unknown_calls = 0

    def enroll(
        self,
        name: str,
        uploads: list[tuple[str, bytes]],
        pipeline: object,
    ) -> EnrollmentResult:
        del uploads, pipeline
        return EnrollmentResult(
            name=name,
            slug="person",
            accepted_files=("face.jpg",),
            rejected_files=(),
            sample_count=1,
        )

    def match(self, embedding: Float32Array, threshold: float) -> GalleryMatch:
        del embedding, threshold
        self.match_calls += 1
        return GalleryMatch(
            name=self._match_name,
            slug=self._match_slug,
            score=0.99,
            matched=self._matched,
        )

    def capture_unknown(self, embedding: Float32Array, crop_bgr: UInt8Array) -> GalleryMatch:
        del embedding, crop_bgr
        self.capture_unknown_calls += 1
        return GalleryMatch(name="unknown-0001", slug="unknown-0001", score=0.0, matched=False)

    def count(self) -> int:
        return self._count_value

    def identities(self) -> list[IdentityRecord]:
        return []

    def unknowns(self) -> list[UnknownRecord]:
        return []

    def rename_identity(self, slug: str, new_name: str) -> IdentityRecord:
        del new_name
        return IdentityRecord(
            name=slug,
            slug=slug,
            template=np.zeros((3,), dtype=np.float32),
            sample_count=1,
            preview_filename=None,
        )

    def promote_unknown(self, unknown_slug: str, name: str) -> EnrollmentResult:
        del unknown_slug
        return EnrollmentResult(
            name=name,
            slug="person",
            accepted_files=("capture_001.jpg",),
            rejected_files=(),
            sample_count=1,
        )

    def delete_unknown(self, unknown_slug: str) -> None:
        del unknown_slug

    def read_image(self, kind: str, slug: str, filename: str) -> tuple[bytes, str]:
        del kind, slug, filename
        return b"123", "image/jpeg"


def build_runtime(
    pipeline: StubPipeline,
    gallery: StubGallery,
    *,
    det_every: int = 1,
    disable_embed_refresh: bool = False,
    embed_refresh_frames: int = 5,
) -> LiveRuntime:
    return LiveRuntime(
        pipeline=pipeline,
        gallery=gallery,
        config=LiveRuntimeConfig(
            max_faces=3,
            det_every=det_every,
            track_iou_thresh=0.3,
            track_max_missed=3,
            track_smoothing=0.65,
            match_threshold=0.228,
            embed_refresh_frames=embed_refresh_frames,
            embed_refresh_iou=0.85,
            disable_embed_refresh=disable_embed_refresh,
            metrics_json_path=None,
            metrics_write_every=10,
        ),
    )


def make_frame() -> UInt8Array:
    return np.zeros((32, 32, 3), dtype=np.uint8)


def make_detection(x1: float = 1.0, x2: float = 11.0) -> FakeDetectionResult:
    boxes = np.asarray([[x1, 2.0, x2, 12.0, 0.95]], dtype=np.float32)
    kps = np.asarray(
        [[[2.0, 3.0], [4.0, 3.0], [3.0, 5.0], [2.5, 7.0], [3.5, 7.0]]],
        dtype=np.float32,
    )
    return FakeDetectionResult(boxes=boxes, kps=kps)


def test_live_runtime_reuses_cached_embedding_between_frames() -> None:
    pipeline = StubPipeline([make_detection(), make_detection()])
    gallery = StubGallery()
    runtime = build_runtime(
        pipeline,
        gallery,
        det_every=1,
        disable_embed_refresh=False,
        embed_refresh_frames=10,
    )

    first = runtime.process_frame(make_frame())
    second = runtime.process_frame(make_frame())

    assert first.refreshes == 1
    assert first.reuses == 0
    assert second.refreshes == 0
    assert second.reuses == 1
    assert pipeline.embed_calls == 1
    assert gallery.match_calls == 1
    assert second.overlay.faces[0].match is not None
    assert second.overlay.faces[0].match.name == "alice"


def test_live_runtime_can_force_embedding_every_frame() -> None:
    pipeline = StubPipeline([make_detection(), make_detection()])
    gallery = StubGallery()
    runtime = build_runtime(
        pipeline,
        gallery,
        det_every=1,
        disable_embed_refresh=True,
    )

    first = runtime.process_frame(make_frame())
    second = runtime.process_frame(make_frame())

    assert first.refreshes == 1
    assert second.refreshes == 1
    assert pipeline.embed_calls == 2
    assert gallery.match_calls == 2


def test_live_runtime_det_every_holds_tracks_between_detection_passes() -> None:
    pipeline = StubPipeline([make_detection()])
    gallery = StubGallery()
    runtime = build_runtime(
        pipeline,
        gallery,
        det_every=2,
        disable_embed_refresh=False,
        embed_refresh_frames=10,
    )

    first = runtime.process_frame(make_frame())
    second = runtime.process_frame(make_frame())

    assert pipeline.detect_calls == 1
    assert first.overlay.detect_ms == 4.0
    assert second.overlay.detect_ms == 0.0
    assert first.overlay.faces[0].track.track_id == second.overlay.faces[0].track.track_id
    assert first.overlay.faces[0].track.matched is True
    assert second.overlay.faces[0].track.matched is False
    assert second.reuses == 1


def test_live_runtime_metrics_snapshot_reports_det_every() -> None:
    pipeline = StubPipeline([make_detection()])
    gallery = StubGallery()
    runtime = build_runtime(pipeline, gallery, det_every=3)

    frame_result = runtime.process_frame(make_frame())
    runtime.record_frame_metrics(
        frame_result=frame_result,
        loop_ms=8.0,
        memory=MemoryStats(current_rss_mb=12.0, peak_rss_mb=14.0),
    )
    runtime.record_error("camera disconnected", memory=MemoryStats(12.0, 14.0))

    snapshot = runtime.metrics_snapshot
    assert snapshot["det_every"] == 3
    assert snapshot["frames_processed"] == 1
    assert snapshot["current_fps"] == 125.0
    assert snapshot["accelerator_mode"] == "cpu-only (ONNX Runtime CPUExecutionProvider)"
    assert snapshot["last_error"] == "camera disconnected"


def test_live_runtime_auto_captures_unknown_faces() -> None:
    pipeline = StubPipeline([make_detection()])
    gallery = StubGallery(match_name=None, match_slug=None, matched=False)
    runtime = build_runtime(pipeline, gallery)

    frame_result = runtime.process_frame(make_frame())

    assert frame_result.refreshes == 1
    assert gallery.capture_unknown_calls == 1
    assert frame_result.overlay.faces[0].match is not None
    assert frame_result.overlay.faces[0].match.name == "unknown-0001"


def test_live_runtime_enroll_delegates_to_gallery() -> None:
    pipeline = StubPipeline([make_detection()])
    gallery = StubGallery()
    runtime = build_runtime(pipeline, gallery)

    result = runtime.enroll("Alice", [("face.jpg", b"123")])

    assert result.name == "Alice"
    assert result.sample_count == 1
