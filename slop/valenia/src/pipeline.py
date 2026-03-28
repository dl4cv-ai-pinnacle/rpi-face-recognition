"""Shared pipeline helpers for detection, alignment, and embedding."""

from __future__ import annotations

import time
from dataclasses import dataclass

from contracts import DetectionLike, DetectorLike, EmbedderLike
from face_align import norm_crop
from runtime_utils import Float32Array, UInt8Array


@dataclass
class FrameResult:
    boxes: Float32Array
    kps: Float32Array | None
    embeddings: list[Float32Array]
    detect_ms: float
    embed_ms_total: float


class FacePipeline:
    """Compose detector + embedder behind one reusable interface."""

    def __init__(
        self,
        detector: DetectorLike,
        embedder: EmbedderLike,
        det_size: tuple[int, int] = (320, 320),
        max_faces: int = 3,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.det_size = det_size
        self.max_faces = max_faces

    def detect(self, frame_bgr: UInt8Array) -> tuple[DetectionLike, float]:
        t0 = time.perf_counter()
        det = self.detector.detect(frame_bgr, self.det_size)
        t1 = time.perf_counter()
        return det, (t1 - t0) * 1000.0

    def embed_from_kps(
        self, frame_bgr: UInt8Array, landmarks: Float32Array
    ) -> tuple[Float32Array, float]:
        crop = norm_crop(frame_bgr, landmarks, image_size=112)
        t0 = time.perf_counter()
        emb = self.embedder.get_embedding(crop)
        t1 = time.perf_counter()
        return emb, (t1 - t0) * 1000.0

    def process_frame(self, frame_bgr: UInt8Array, max_faces: int | None = None) -> FrameResult:
        det, detect_ms = self.detect(frame_bgr)
        if det.kps is None:
            return FrameResult(
                boxes=det.boxes,
                kps=None,
                embeddings=[],
                detect_ms=detect_ms,
                embed_ms_total=0.0,
            )

        face_limit = self.max_faces if max_faces is None else max_faces
        embeddings: list[Float32Array] = []
        embed_ms_total = 0.0
        for idx in range(min(face_limit, len(det.boxes))):
            emb, emb_ms = self.embed_from_kps(frame_bgr, det.kps[idx])
            embeddings.append(emb)
            embed_ms_total += emb_ms

        return FrameResult(
            boxes=det.boxes,
            kps=det.kps,
            embeddings=embeddings,
            detect_ms=detect_ms,
            embed_ms_total=embed_ms_total,
        )
