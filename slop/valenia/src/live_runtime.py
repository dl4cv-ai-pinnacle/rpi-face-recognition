"""Reusable live face runtime for tracking, matching, and metrics."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from contracts import GalleryLike, PipelineLike
from gallery import EnrollmentResult, GalleryMatch, IdentityRecord, UnknownRecord
from runtime_utils import (
    CpuUsageSampler,
    Float32Array,
    MemoryStats,
    UInt8Array,
    get_system_stats,
)
from tracking import SimpleFaceTracker, Track


@dataclass(frozen=True)
class LiveRuntimeConfig:
    max_faces: int
    target_fps: float
    det_every: int
    track_iou_thresh: float
    track_max_missed: int
    track_smoothing: float
    match_threshold: float
    embed_refresh_frames: int
    embed_refresh_iou: float
    disable_embed_refresh: bool
    metrics_json_path: Path | None
    metrics_write_every: int


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
class LiveRuntimeFrameResult:
    overlay: TrackingOverlay
    fresh_tracks: int
    recognized_faces: int
    refreshes: int
    reuses: int


@dataclass(frozen=True)
class TrackIdentityState:
    match: GalleryMatch
    last_embed_frame: int
    last_embed_box: Float32Array


@dataclass(frozen=True)
class LiveMetricsSnapshot:
    updated_at_epoch: float
    uptime_seconds: float
    frames_processed: int
    target_fps: float
    det_every: int
    current_output_fps: float
    avg_output_fps: float
    current_processing_fps: float
    last_loop_ms: float
    avg_loop_ms: float
    last_detect_ms: float
    avg_detect_ms: float
    last_track_ms: float
    avg_track_ms: float
    last_embed_ms: float
    avg_embed_ms: float
    last_faces: int
    avg_faces_per_frame: float
    last_fresh_tracks: int
    avg_fresh_tracks_per_frame: float
    last_recognized_faces: int
    avg_recognized_faces_per_frame: float
    last_refreshes: int
    avg_refreshes_per_frame: float
    last_reuses: int
    avg_reuses_per_frame: float
    gallery_size: int
    cpu_usage_pct: float
    loadavg_1m: float
    loadavg_5m: float
    loadavg_15m: float
    cpu_temp_c: float | None
    gpu_usage_pct: float | None
    current_rss_mb: float
    peak_rss_mb: float
    accelerator_mode: str
    embed_refresh_enabled: bool
    metrics_json_path: str | None
    last_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "updated_at_epoch": round(self.updated_at_epoch, 3),
            "uptime_seconds": round(self.uptime_seconds, 3),
            "frames_processed": self.frames_processed,
            "target_fps": round(self.target_fps, 3),
            "det_every": self.det_every,
            "current_output_fps": round(self.current_output_fps, 3),
            "avg_output_fps": round(self.avg_output_fps, 3),
            "current_processing_fps": round(self.current_processing_fps, 3),
            "last_loop_ms": round(self.last_loop_ms, 3),
            "avg_loop_ms": round(self.avg_loop_ms, 3),
            "last_detect_ms": round(self.last_detect_ms, 3),
            "avg_detect_ms": round(self.avg_detect_ms, 3),
            "last_track_ms": round(self.last_track_ms, 3),
            "avg_track_ms": round(self.avg_track_ms, 3),
            "last_embed_ms": round(self.last_embed_ms, 3),
            "avg_embed_ms": round(self.avg_embed_ms, 3),
            "last_faces": self.last_faces,
            "avg_faces_per_frame": round(self.avg_faces_per_frame, 3),
            "last_fresh_tracks": self.last_fresh_tracks,
            "avg_fresh_tracks_per_frame": round(self.avg_fresh_tracks_per_frame, 3),
            "last_recognized_faces": self.last_recognized_faces,
            "avg_recognized_faces_per_frame": round(self.avg_recognized_faces_per_frame, 3),
            "last_refreshes": self.last_refreshes,
            "avg_refreshes_per_frame": round(self.avg_refreshes_per_frame, 3),
            "last_reuses": self.last_reuses,
            "avg_reuses_per_frame": round(self.avg_reuses_per_frame, 3),
            "gallery_size": self.gallery_size,
            "cpu_usage_pct": round(self.cpu_usage_pct, 3),
            "loadavg_1m": round(self.loadavg_1m, 3),
            "loadavg_5m": round(self.loadavg_5m, 3),
            "loadavg_15m": round(self.loadavg_15m, 3),
            "cpu_temp_c": None if self.cpu_temp_c is None else round(self.cpu_temp_c, 3),
            "gpu_usage_pct": None if self.gpu_usage_pct is None else round(self.gpu_usage_pct, 3),
            "current_rss_mb": round(self.current_rss_mb, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 3),
            "accelerator_mode": self.accelerator_mode,
            "embed_refresh_enabled": self.embed_refresh_enabled,
            "metrics_json_path": self.metrics_json_path,
            "last_error": self.last_error,
        }


class LiveMetricsCollector:
    """Track rolling live metrics and periodically persist them to JSON."""

    def __init__(
        self,
        *,
        metrics_json_path: Path | None,
        write_every_frames: int,
        embed_refresh_enabled: bool,
        det_every: int,
        target_fps: float,
    ) -> None:
        self.metrics_json_path = metrics_json_path
        self.write_every_frames = max(1, write_every_frames)
        self.embed_refresh_enabled = embed_refresh_enabled
        self.det_every = max(1, det_every)
        self.target_fps = max(0.0, float(target_fps))
        self._start_mono = time.perf_counter()
        self._lock = threading.Lock()
        self._frames_processed = 0
        self._sum_loop_ms = 0.0
        self._sum_detect_ms = 0.0
        self._sum_track_ms = 0.0
        self._sum_embed_ms = 0.0
        self._sum_faces = 0
        self._sum_fresh_tracks = 0
        self._sum_recognized_faces = 0
        self._sum_refreshes = 0
        self._sum_reuses = 0
        self._last_error: str | None = None
        self._last_loop_ms = 0.0
        self._last_detect_ms = 0.0
        self._last_track_ms = 0.0
        self._last_embed_ms = 0.0
        self._last_faces = 0
        self._last_fresh_tracks = 0
        self._last_recognized_faces = 0
        self._last_refreshes = 0
        self._last_reuses = 0
        self._last_frame_mono: float | None = None
        self._last_output_fps = 0.0
        self._cpu_sampler = CpuUsageSampler()
        path_str = None
        if self.metrics_json_path is not None:
            self.metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
            path_str = str(self.metrics_json_path)
        system = get_system_stats(self._cpu_sampler)
        self._snapshot = LiveMetricsSnapshot(
            updated_at_epoch=time.time(),
            uptime_seconds=0.0,
            frames_processed=0,
            target_fps=self.target_fps,
            det_every=self.det_every,
            current_output_fps=0.0,
            avg_output_fps=0.0,
            current_processing_fps=0.0,
            last_loop_ms=0.0,
            avg_loop_ms=0.0,
            last_detect_ms=0.0,
            avg_detect_ms=0.0,
            last_track_ms=0.0,
            avg_track_ms=0.0,
            last_embed_ms=0.0,
            avg_embed_ms=0.0,
            last_faces=0,
            avg_faces_per_frame=0.0,
            last_fresh_tracks=0,
            avg_fresh_tracks_per_frame=0.0,
            last_recognized_faces=0,
            avg_recognized_faces_per_frame=0.0,
            last_refreshes=0,
            avg_refreshes_per_frame=0.0,
            last_reuses=0,
            avg_reuses_per_frame=0.0,
            gallery_size=0,
            cpu_usage_pct=system.cpu_usage_pct,
            loadavg_1m=system.loadavg_1m,
            loadavg_5m=system.loadavg_5m,
            loadavg_15m=system.loadavg_15m,
            cpu_temp_c=system.cpu_temp_c,
            gpu_usage_pct=system.gpu_usage_pct,
            current_rss_mb=0.0,
            peak_rss_mb=0.0,
            accelerator_mode="cpu-only (ONNX Runtime CPUExecutionProvider)",
            embed_refresh_enabled=self.embed_refresh_enabled,
            metrics_json_path=path_str,
            last_error=None,
        )
        self._write_snapshot(self._snapshot)

    def record_frame(
        self,
        *,
        frame_result: LiveRuntimeFrameResult,
        loop_ms: float,
        memory: MemoryStats,
    ) -> None:
        snapshot: LiveMetricsSnapshot
        should_write: bool
        with self._lock:
            now_mono = time.perf_counter()
            self._frames_processed += 1
            self._sum_loop_ms += loop_ms
            self._sum_detect_ms += frame_result.overlay.detect_ms
            self._sum_track_ms += frame_result.overlay.track_ms
            self._sum_embed_ms += frame_result.overlay.embed_ms_total
            self._sum_faces += len(frame_result.overlay.faces)
            self._sum_fresh_tracks += frame_result.fresh_tracks
            self._sum_recognized_faces += frame_result.recognized_faces
            self._sum_refreshes += frame_result.refreshes
            self._sum_reuses += frame_result.reuses
            self._last_loop_ms = loop_ms
            self._last_detect_ms = frame_result.overlay.detect_ms
            self._last_track_ms = frame_result.overlay.track_ms
            self._last_embed_ms = frame_result.overlay.embed_ms_total
            self._last_faces = len(frame_result.overlay.faces)
            self._last_fresh_tracks = frame_result.fresh_tracks
            self._last_recognized_faces = frame_result.recognized_faces
            self._last_refreshes = frame_result.refreshes
            self._last_reuses = frame_result.reuses
            if self._last_frame_mono is None:
                self._last_output_fps = 0.0
            else:
                delta_s = now_mono - self._last_frame_mono
                self._last_output_fps = (1.0 / delta_s) if delta_s > 0.0 else 0.0
            self._last_frame_mono = now_mono
            snapshot = self._build_snapshot_locked(
                gallery_size=frame_result.overlay.gallery_size,
                memory=memory,
            )
            self._snapshot = snapshot
            should_write = self.metrics_json_path is not None and (
                self._frames_processed % self.write_every_frames == 0
            )

        if should_write:
            self._write_snapshot(snapshot)

    def record_error(self, error: str, *, memory: MemoryStats, gallery_size: int) -> None:
        with self._lock:
            self._last_error = error
            self._snapshot = self._build_snapshot_locked(gallery_size=gallery_size, memory=memory)
            snapshot = self._snapshot
        self._write_snapshot(snapshot)

    def snapshot_dict(self) -> dict[str, object]:
        with self._lock:
            snapshot = self._snapshot
        return snapshot.as_dict()

    def _build_snapshot_locked(
        self, *, gallery_size: int, memory: MemoryStats
    ) -> LiveMetricsSnapshot:
        frames = self._frames_processed
        uptime = max(0.0, time.perf_counter() - self._start_mono)
        avg_loop_ms = self._sum_loop_ms / frames if frames else 0.0
        avg_output_fps = (float(frames) / uptime) if uptime > 0.0 else 0.0
        current_processing_fps = (1000.0 / self._last_loop_ms) if self._last_loop_ms > 0.0 else 0.0
        path_str = str(self.metrics_json_path) if self.metrics_json_path is not None else None
        system = get_system_stats(self._cpu_sampler)
        return LiveMetricsSnapshot(
            updated_at_epoch=time.time(),
            uptime_seconds=uptime,
            frames_processed=frames,
            target_fps=self.target_fps,
            det_every=self.det_every,
            current_output_fps=self._last_output_fps,
            avg_output_fps=avg_output_fps,
            current_processing_fps=current_processing_fps,
            last_loop_ms=self._last_loop_ms,
            avg_loop_ms=avg_loop_ms,
            last_detect_ms=self._last_detect_ms,
            avg_detect_ms=(self._sum_detect_ms / frames) if frames else 0.0,
            last_track_ms=self._last_track_ms,
            avg_track_ms=(self._sum_track_ms / frames) if frames else 0.0,
            last_embed_ms=self._last_embed_ms,
            avg_embed_ms=(self._sum_embed_ms / frames) if frames else 0.0,
            last_faces=self._last_faces,
            avg_faces_per_frame=(self._sum_faces / frames) if frames else 0.0,
            last_fresh_tracks=self._last_fresh_tracks,
            avg_fresh_tracks_per_frame=(self._sum_fresh_tracks / frames) if frames else 0.0,
            last_recognized_faces=self._last_recognized_faces,
            avg_recognized_faces_per_frame=(self._sum_recognized_faces / frames if frames else 0.0),
            last_refreshes=self._last_refreshes,
            avg_refreshes_per_frame=(self._sum_refreshes / frames) if frames else 0.0,
            last_reuses=self._last_reuses,
            avg_reuses_per_frame=(self._sum_reuses / frames) if frames else 0.0,
            gallery_size=gallery_size,
            cpu_usage_pct=system.cpu_usage_pct,
            loadavg_1m=system.loadavg_1m,
            loadavg_5m=system.loadavg_5m,
            loadavg_15m=system.loadavg_15m,
            cpu_temp_c=system.cpu_temp_c,
            gpu_usage_pct=system.gpu_usage_pct,
            current_rss_mb=memory.current_rss_mb,
            peak_rss_mb=memory.peak_rss_mb,
            accelerator_mode="cpu-only (ONNX Runtime CPUExecutionProvider)",
            embed_refresh_enabled=self.embed_refresh_enabled,
            metrics_json_path=path_str,
            last_error=self._last_error,
        )

    def _write_snapshot(self, snapshot: LiveMetricsSnapshot) -> None:
        if self.metrics_json_path is None:
            return
        self.metrics_json_path.write_text(
            json.dumps(snapshot.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class LiveRuntime:
    """Stateful live processing runtime shared by camera, web, or benchmarks."""

    def __init__(
        self,
        pipeline: PipelineLike,
        gallery: GalleryLike,
        config: LiveRuntimeConfig,
    ) -> None:
        self.pipeline = pipeline
        self.gallery = gallery
        self.config = config
        self._frame_counter = 0
        self._tracker = SimpleFaceTracker(
            iou_threshold=config.track_iou_thresh,
            max_missed=config.track_max_missed,
            smoothing=config.track_smoothing,
        )
        self._last_tracks: list[Track] = []
        self._track_states: dict[int, TrackIdentityState] = {}
        self._metrics = LiveMetricsCollector(
            metrics_json_path=config.metrics_json_path,
            write_every_frames=config.metrics_write_every,
            embed_refresh_enabled=not config.disable_embed_refresh,
            det_every=config.det_every,
            target_fps=config.target_fps,
        )

    @property
    def metrics_snapshot(self) -> dict[str, object]:
        return self._metrics.snapshot_dict()

    def enroll(self, name: str, uploads: list[tuple[str, bytes]]) -> EnrollmentResult:
        return self.gallery.enroll(name, uploads, self.pipeline)

    def list_identities(self) -> list[IdentityRecord]:
        return self.gallery.identities()

    def list_unknowns(self) -> list[UnknownRecord]:
        return self.gallery.unknowns()

    def rename_identity(self, slug: str, new_name: str) -> IdentityRecord:
        return self.gallery.rename_identity(slug, new_name)

    def promote_unknown(self, unknown_slug: str, name: str) -> EnrollmentResult:
        return self.gallery.promote_unknown(unknown_slug, name)

    def delete_unknown(self, unknown_slug: str) -> None:
        self.gallery.delete_unknown(unknown_slug)

    def read_gallery_image(self, kind: str, slug: str, filename: str) -> tuple[bytes, str]:
        return self.gallery.read_image(kind, slug, filename)

    def process_frame(self, frame_bgr: UInt8Array) -> LiveRuntimeFrameResult:
        self._frame_counter += 1
        track_t0 = time.perf_counter()
        detect_ms = 0.0
        if self._should_run_detection():
            det, detect_ms = self.pipeline.detect(frame_bgr)
            tracks = self._tracker.update(det.boxes, det.kps, max_tracks=self.config.max_faces)
            self._last_tracks = list(tracks)
        else:
            tracks = self._build_held_tracks()
        track_ms = (time.perf_counter() - track_t0) * 1000.0

        live_ids = {track.track_id for track in tracks}
        self._track_states = {
            track_id: state
            for track_id, state in self._track_states.items()
            if track_id in live_ids
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
                if landmarks is None:
                    overlay_faces.append(OverlayFace(track=track, match=match))
                    continue
                emb, emb_ms = self.pipeline.embed_from_kps(frame_bgr, landmarks)
                embed_ms_total += emb_ms
                refreshes += 1
                new_match = self.gallery.match(emb, self.config.match_threshold)
                current_box = np.asarray(track.box, dtype=np.float32)
                if self._should_keep_previous_known_match(track, state, new_match):
                    assert state is not None
                    match = state.match
                elif not new_match.matched:
                    crop = _crop_face_region(frame_bgr, track.box)
                    if crop is not None:
                        match = self.gallery.capture_unknown(emb, crop)
                    else:
                        match = new_match
                    self._track_states[track.track_id] = TrackIdentityState(
                        match=match,
                        last_embed_frame=self._frame_counter,
                        last_embed_box=current_box,
                    )
                else:
                    match = new_match
                    self._track_states[track.track_id] = TrackIdentityState(
                        match=match,
                        last_embed_frame=self._frame_counter,
                        last_embed_box=current_box,
                    )
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
        fresh_tracks = sum(1 for face in overlay_faces if face.track.matched)
        recognized_faces = sum(
            1
            for face in overlay_faces
            if face.match is not None and face.match.matched and face.match.name is not None
        )
        return LiveRuntimeFrameResult(
            overlay=overlay,
            fresh_tracks=fresh_tracks,
            recognized_faces=recognized_faces,
            refreshes=refreshes,
            reuses=reuses,
        )

    def record_frame_metrics(
        self,
        *,
        frame_result: LiveRuntimeFrameResult,
        loop_ms: float,
        memory: MemoryStats,
    ) -> None:
        self._metrics.record_frame(frame_result=frame_result, loop_ms=loop_ms, memory=memory)

    def record_error(self, error: str, *, memory: MemoryStats) -> None:
        self._metrics.record_error(error, memory=memory, gallery_size=self.gallery.count())

    def _should_refresh_embedding(
        self,
        track: Track,
        state: TrackIdentityState | None,
    ) -> bool:
        if track.kps is None or not track.matched:
            return False
        if self.config.disable_embed_refresh:
            return True
        if state is None:
            return True

        refresh_frames = max(0, int(self.config.embed_refresh_frames))
        if state.match.matched and state.match.name is not None and refresh_frames > 0:
            refresh_frames = min(refresh_frames, 2)
        if refresh_frames > 0 and (self._frame_counter - state.last_embed_frame) >= refresh_frames:
            return True

        current_box = np.asarray(track.box, dtype=np.float32)
        return _box_iou(current_box[:4], state.last_embed_box[:4]) < self.config.embed_refresh_iou

    def _should_keep_previous_known_match(
        self,
        track: Track,
        state: TrackIdentityState | None,
        new_match: GalleryMatch,
    ) -> bool:
        if new_match.matched or state is None:
            return False
        if not state.match.matched or state.match.name is None:
            return False

        frames_since_confirmed = self._frame_counter - state.last_embed_frame
        if frames_since_confirmed > 1:
            return False

        prior_margin = state.match.score - self.config.match_threshold
        if prior_margin < 0.08:
            return False

        current_box = np.asarray(track.box, dtype=np.float32)
        grace_iou = max(0.9, self.config.embed_refresh_iou)
        return _box_iou(current_box[:4], state.last_embed_box[:4]) >= grace_iou

    def _should_run_detection(self) -> bool:
        interval = max(1, int(self.config.det_every))
        return ((self._frame_counter - 1) % interval) == 0

    def _build_held_tracks(self) -> list[Track]:
        held_tracks: list[Track] = []
        for track in self._last_tracks:
            held_tracks.append(
                Track(
                    track_id=track.track_id,
                    box=np.asarray(track.box, dtype=np.float32),
                    kps=None if track.kps is None else np.asarray(track.kps, dtype=np.float32),
                    age=track.age + 1,
                    hits=track.hits,
                    missed=track.missed,
                    matched=False,
                )
            )
        return held_tracks


def annotate_in_place(frame_bgr: UInt8Array, overlay: TrackingOverlay) -> None:
    for face in overlay.faces:
        track = face.track
        box = track.box
        x1, y1, x2, y2, _score = box
        color = (0, 220, 140)
        if face.match is not None and face.match.matched:
            color = (64, 224, 208)
        cv2.rectangle(
            frame_bgr,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=color,
            thickness=2,
        )

        label: str | None = None
        if face.match is not None and face.match.matched and face.match.name is not None:
            label = face.match.name
        elif face.match is not None:
            unknown_label = face.match.name if face.match.name is not None else "unknown"
            label = unknown_label

        if label is not None:
            cv2.putText(
                frame_bgr,
                label,
                (int(x1), max(16, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            cv2.putText(
                frame_bgr,
                label,
                (int(x1), max(16, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (12, 18, 20),
                1,
            )
        if track.kps is None:
            continue
        for point in track.kps:
            cv2.circle(frame_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)


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


def _box_iou(box_a: Float32Array, box_b: Float32Array) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union
