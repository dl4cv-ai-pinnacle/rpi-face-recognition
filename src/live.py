"""Live face processing runtime — tracking, matching, metrics, hot-swap.

Origins:
- LiveRuntime, process_frame logic, embed refresh, grace period: Valenia live_runtime.py
- RLock + swap_pipeline() for GUI hot-swap: new (validated design)
- annotate_in_place → moved to display.py
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.alignment import center_crop_fallback
from src.config import AppConfig
from src.contracts import Float32Array, GalleryLike, PipelineLike, UInt8Array
from src.gallery import EnrollmentResult, GalleryMatch, IdentityRecord, UnknownRecord
from src.metrics import LiveMetricsCollector, get_memory_stats
from src.quality import compute_face_quality
from src.tracking import SimpleFaceTracker, Track, box_iou


@dataclass(frozen=True)
class OverlayFace:
    track: Track
    match: GalleryMatch | None


@dataclass(frozen=True)
class TrackingOverlay:
    faces: list[OverlayFace]
    detect_ms: float
    track_ms: float
    embed_ms_total: float
    gallery_size: int


@dataclass(frozen=True)
class LiveFrameResult:
    overlay: TrackingOverlay
    fresh_tracks: int
    recognized_faces: int
    refreshes: int
    reuses: int


@dataclass(frozen=True)
class _TrackIdentityState:
    match: GalleryMatch
    last_embed_frame: int
    last_embed_box: Float32Array


class LiveRuntime:
    """Stateful live processing runtime with thread-safe pipeline hot-swap."""

    def __init__(
        self,
        pipeline: PipelineLike,
        gallery: GalleryLike,
        config: AppConfig,
    ) -> None:
        self.pipeline = pipeline
        self.gallery = gallery
        self.config = config
        self._pipeline_lock = threading.RLock()
        self._frame_counter = 0
        self._tracker = SimpleFaceTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_missed=config.tracking.max_missed,
            smoothing=config.tracking.smoothing,
        )
        self._last_tracks: list[Track] = []
        self._track_states: dict[int, _TrackIdentityState] = {}
        self._last_enrich_time: dict[str, float] = {}

        metrics_path = (
            Path(config.metrics.json_path) if config.metrics.json_path else None
        )
        self._metrics = LiveMetricsCollector(
            metrics_json_path=metrics_path,
            write_every_frames=config.metrics.write_every_frames,
            det_every=config.live.det_every,
            target_fps=config.live.target_fps,
        )

    # -- Pipeline hot-swap (thread-safe) -------------------------------

    def swap_pipeline(self, new_pipeline: PipelineLike) -> PipelineLike:
        """Swap the active pipeline. Returns the old one for cleanup.

        Clears tracker state since cached embeddings are from the old pipeline.
        """
        with self._pipeline_lock:
            old = self.pipeline
            self.pipeline = new_pipeline
            self._track_states.clear()
            self._last_tracks.clear()
            self._tracker.reset()
        return old

    # -- Metrics -------------------------------------------------------

    @property
    def metrics_snapshot(self) -> dict[str, object]:
        return self._metrics.snapshot_dict()

    def record_frame_metrics(self, *, frame_result: LiveFrameResult, loop_ms: float) -> None:
        memory = get_memory_stats()
        self._metrics.record_frame(
            detect_ms=frame_result.overlay.detect_ms,
            track_ms=frame_result.overlay.track_ms,
            embed_ms=frame_result.overlay.embed_ms_total,
            face_count=len(frame_result.overlay.faces),
            loop_ms=loop_ms,
            memory=memory,
            gallery_size=frame_result.overlay.gallery_size,
        )

    def record_error(self, error: str) -> None:
        self._metrics.record_error(error)

    # -- Gallery delegation --------------------------------------------

    def enroll(self, name: str, uploads: list[tuple[str, bytes]]) -> EnrollmentResult:
        return self.gallery.enroll(name, uploads, self.pipeline)

    def list_identities(self) -> list[IdentityRecord]:
        return self.gallery.identities()

    def list_unknowns(self) -> list[UnknownRecord]:
        return self.gallery.unknowns()

    def upload_to_identity(
        self, slug: str, uploads: list[tuple[str, bytes]]
    ) -> EnrollmentResult:
        return self.gallery.upload_to_identity(slug, uploads, self.pipeline)

    def rename_identity(self, slug: str, new_name: str) -> IdentityRecord:
        return self.gallery.rename_identity(slug, new_name)

    def promote_unknown(self, unknown_slug: str, name: str) -> EnrollmentResult:
        return self.gallery.promote_unknown(unknown_slug, name)

    def merge_unknowns(self, target_slug: str, source_slug: str) -> UnknownRecord:
        return self.gallery.merge_unknowns(target_slug, source_slug)

    def delete_unknown(self, unknown_slug: str) -> None:
        self.gallery.delete_unknown(unknown_slug)

    def read_gallery_image(
        self, kind: str, slug: str, filename: str
    ) -> tuple[bytes, str]:
        return self.gallery.read_image(kind, slug, filename)

    def list_identity_images(self, slug: str) -> list[str]:
        return self.gallery.list_identity_images(slug)

    def delete_identity(self, slug: str) -> None:
        self.gallery.delete_identity(slug)

    def delete_identity_sample(self, slug: str, filename: str) -> IdentityRecord:
        return self.gallery.delete_identity_sample(slug, filename)

    # -- Frame processing ----------------------------------------------

    def process_frame(self, frame_bgr: UInt8Array) -> LiveFrameResult:
        self._frame_counter += 1

        # Grab a stable reference to the pipeline (hot-swap safe).
        with self._pipeline_lock:
            pipeline = self.pipeline

        track_t0 = time.perf_counter()
        detect_ms = 0.0
        if self._should_run_detection():
            det, detect_ms = pipeline.detect(frame_bgr)
            tracks = self._tracker.update(
                det.boxes, det.kps, max_tracks=self.config.live.max_faces
            )
            self._last_tracks = list(tracks)
        else:
            tracks = self._build_held_tracks()
        track_ms = (time.perf_counter() - track_t0) * 1000.0

        live_ids = {track.track_id for track in tracks}
        self._track_states = {
            tid: state for tid, state in self._track_states.items() if tid in live_ids
        }

        overlay_faces: list[OverlayFace] = []
        embed_ms_total = 0.0
        refreshes = 0
        reuses = 0
        for track in tracks:
            state = self._track_states.get(track.track_id)
            match = state.match if state is not None else None
            if self._should_refresh_embedding(track, state):
                landmarks = track.kps
                if landmarks is not None:
                    emb, emb_ms = pipeline.embed_from_kps(frame_bgr, landmarks)
                else:
                    crop = center_crop_fallback(
                        frame_bgr,
                        float(track.box[0]),
                        float(track.box[1]),
                        float(track.box[2]),
                        float(track.box[3]),
                    )
                    t0 = time.perf_counter()
                    emb = pipeline.embedder.get_embedding(crop)  # type: ignore[union-attr]
                    emb_ms = (time.perf_counter() - t0) * 1000.0
                embed_ms_total += emb_ms
                refreshes += 1
                new_match = self.gallery.match(emb, self.config.live.match_threshold)
                current_box = np.asarray(track.box, dtype=np.float32)
                if self._should_keep_previous_known_match(track, state, new_match):
                    match = state.match  # type: ignore[union-attr]
                elif not new_match.matched:
                    crop = _crop_face_region(frame_bgr, track.box)
                    if crop is not None:
                        match = self.gallery.capture_unknown(emb, crop)
                    else:
                        match = new_match
                    self._track_states[track.track_id] = _TrackIdentityState(
                        match=match,
                        last_embed_frame=self._frame_counter,
                        last_embed_box=current_box,
                    )
                else:
                    match = new_match
                    self._track_states[track.track_id] = _TrackIdentityState(
                        match=match,
                        last_embed_frame=self._frame_counter,
                        last_embed_box=current_box,
                    )
                    self._try_enrich(new_match, emb, track.box, frame_bgr)
            elif match is not None:
                reuses += 1
            overlay_faces.append(OverlayFace(track=track, match=match))

        overlay = TrackingOverlay(
            faces=overlay_faces,
            detect_ms=detect_ms,
            track_ms=track_ms,
            embed_ms_total=embed_ms_total,
            gallery_size=self.gallery.count(),
        )
        fresh_tracks = sum(1 for f in overlay_faces if f.track.matched)
        recognized_faces = sum(
            1
            for f in overlay_faces
            if f.match is not None and f.match.matched and f.match.name is not None
        )
        return LiveFrameResult(
            overlay=overlay,
            fresh_tracks=fresh_tracks,
            recognized_faces=recognized_faces,
            refreshes=refreshes,
            reuses=reuses,
        )

    # -- Private helpers -----------------------------------------------

    def _try_enrich(
        self,
        match: GalleryMatch,
        embedding: Float32Array,
        box_arr: Float32Array,
        frame_bgr: UInt8Array,
    ) -> None:
        if match.slug is None:
            return
        margin = match.score - self.config.live.match_threshold
        if margin < self.config.gallery.enrich_margin:
            return
        q = compute_face_quality(box_arr)
        if q.score < self.config.gallery.enrich_min_quality:
            return
        now = time.perf_counter()
        last = self._last_enrich_time.get(match.slug, 0.0)
        if (now - last) < self.config.gallery.enrich_cooldown_seconds:
            return
        crop = _crop_face_region(frame_bgr, box_arr)
        added = self.gallery.enrich_identity(
            match.slug,
            embedding,
            q.score,
            max_samples=self.config.gallery.enrich_max_samples,
            crop_bgr=crop,
        )
        if added:
            self._last_enrich_time[match.slug] = now

    def _should_refresh_embedding(
        self, track: Track, state: _TrackIdentityState | None
    ) -> bool:
        if not track.matched:
            return False
        if state is None:
            return True
        refresh_frames = max(0, self.config.live.embed_refresh_frames)
        if state.match.matched and state.match.name is not None and refresh_frames > 0:
            refresh_frames = min(refresh_frames, 2)
        if refresh_frames > 0 and (self._frame_counter - state.last_embed_frame) >= refresh_frames:
            return True
        current_box = np.asarray(track.box, dtype=np.float32)
        iou = box_iou(current_box[:4], state.last_embed_box[:4])
        return iou < self.config.live.embed_refresh_iou

    def _should_keep_previous_known_match(
        self,
        track: Track,
        state: _TrackIdentityState | None,
        new_match: GalleryMatch,
    ) -> bool:
        if new_match.matched or state is None:
            return False
        if not state.match.matched or state.match.name is None:
            return False
        frames_since = self._frame_counter - state.last_embed_frame
        if frames_since > 1:
            return False
        prior_margin = state.match.score - self.config.live.match_threshold
        if prior_margin < 0.08:
            return False
        current_box = np.asarray(track.box, dtype=np.float32)
        grace_iou = max(0.9, self.config.live.embed_refresh_iou)
        return box_iou(current_box[:4], state.last_embed_box[:4]) >= grace_iou

    def _should_run_detection(self) -> bool:
        interval = max(1, self.config.live.det_every)
        return ((self._frame_counter - 1) % interval) == 0

    def _build_held_tracks(self) -> list[Track]:
        return [
            Track(
                track_id=t.track_id,
                box=np.asarray(t.box, dtype=np.float32),
                kps=None if t.kps is None else np.asarray(t.kps, dtype=np.float32),
                age=t.age + 1,
                hits=t.hits,
                missed=t.missed,
                matched=False,
            )
            for t in self._last_tracks
        ]


def _crop_face_region(frame_bgr: UInt8Array, box: Float32Array) -> UInt8Array | None:
    height, width = frame_bgr.shape[:2]
    x1 = max(0, int(np.floor(float(box[0]))))
    y1 = max(0, int(np.floor(float(box[1]))))
    x2 = min(width, int(np.ceil(float(box[2]))))
    y2 = min(height, int(np.ceil(float(box[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return np.asarray(crop, dtype=np.uint8)
