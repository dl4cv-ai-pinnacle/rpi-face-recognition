"""Face pipeline: compose detection + alignment + embedding.

Origins:
- FacePipeline, FrameResult: Valenia src/pipeline.py
- step() as testable single-frame API: Avdieienko src/pipeline.py
- PipelineSpec, build_face_pipeline: Valenia src/pipeline_factory.py
- Factory dispatch via config: Shalaiev detectors/__init__.py pattern
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.alignment import center_crop_fallback, create_aligner
from src.contracts import (
    AlignerLike,
    Detection,
    DetectorLike,
    EmbedderLike,
    Float32Array,
    UInt8Array,
)

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.contracts import DetectionResultLike


@dataclass
class DetectionResult:
    """Concrete detection result implementing DetectionResultLike."""

    boxes: Float32Array
    kps: Float32Array | None


@dataclass
class FrameResult:
    """Full frame processing result implementing FrameResultLike."""

    boxes: Float32Array
    kps: Float32Array | None
    embeddings: list[Float32Array]
    detect_ms: float
    embed_ms_total: float


class FacePipeline:
    """Compose detector + aligner + embedder behind one reusable interface."""

    def __init__(
        self,
        detector: DetectorLike,
        aligner: AlignerLike,
        embedder: EmbedderLike,
        max_faces: int = 3,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.max_faces = max_faces

    def detect(self, frame_bgr: UInt8Array, /) -> tuple[DetectionResultLike, float]:
        t0 = time.perf_counter()
        detections = self.detector.detect(frame_bgr)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        if not detections:
            return DetectionResult(
                boxes=np.zeros((0, 5), dtype=np.float32),
                kps=None,
            ), detect_ms

        boxes = np.array(
            [[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in detections],
            dtype=np.float32,
        )

        kps = None
        kps_list = [d.landmarks for d in detections if d.landmarks is not None]
        if len(kps_list) == len(detections):
            kps = np.stack(kps_list).astype(np.float32)

        return DetectionResult(boxes=boxes, kps=kps), detect_ms

    def embed_from_kps(
        self, frame_bgr: UInt8Array, landmarks: Float32Array, /
    ) -> tuple[Float32Array, float]:
        crop = self.aligner.align(frame_bgr, landmarks)
        if crop is None:
            crop = np.zeros((112, 112, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        emb = self.embedder.get_embedding(crop)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        return emb, embed_ms

    def _embed_from_bbox(
        self, frame_bgr: UInt8Array, det: Detection, /
    ) -> tuple[Float32Array, float]:
        """Embed using center-crop fallback when no landmarks are available."""
        crop = center_crop_fallback(frame_bgr, det.x1, det.y1, det.x2, det.y2)
        t0 = time.perf_counter()
        emb = self.embedder.get_embedding(crop)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        return emb, embed_ms

    def process_frame(
        self, frame_bgr: UInt8Array, /, max_faces: int | None = None
    ) -> FrameResult:
        det_result, detect_ms = self.detect(frame_bgr)

        face_limit = self.max_faces if max_faces is None else max_faces
        embeddings: list[Float32Array] = []
        embed_ms_total = 0.0

        n_faces = min(face_limit, len(det_result.boxes))
        for idx in range(n_faces):
            if det_result.kps is not None:
                emb, emb_ms = self.embed_from_kps(frame_bgr, det_result.kps[idx])
            else:
                # Center-crop fallback for landmark-free detectors (UltraFace).
                box = det_result.boxes[idx]
                det = Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(box[4]),
                )
                emb, emb_ms = self._embed_from_bbox(frame_bgr, det)
            embeddings.append(emb)
            embed_ms_total += emb_ms

        return FrameResult(
            boxes=det_result.boxes,
            kps=det_result.kps,
            embeddings=embeddings,
            detect_ms=detect_ms,
            embed_ms_total=embed_ms_total,
        )

    def step(
        self, frame_bgr: UInt8Array, /, max_faces: int | None = None
    ) -> FrameResult:
        """Alias for process_frame — testable single-frame API (Avdieienko pattern)."""
        return self.process_frame(frame_bgr, max_faces=max_faces)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_pipeline(config: AppConfig) -> FacePipeline:
    """Build a complete pipeline from AppConfig."""
    from src.detection import create_detector
    from src.embedding import create_embedder

    detector = create_detector(config.detection)
    aligner = create_aligner(config.alignment)
    embedder = create_embedder(config.embedding)

    return FacePipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        max_faces=config.live.max_faces,
    )
