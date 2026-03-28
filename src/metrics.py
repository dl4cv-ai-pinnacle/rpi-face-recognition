"""System telemetry and live metrics collection.

Origins:
- MemoryStats, get_memory_stats, enforce_memory_cap: Valenia runtime_utils.py
- CpuUsageSampler, SystemStats, thermal/GPU readers: Valenia runtime_utils.py
- LiveMetricsCollector, LiveMetricsSnapshot: Valenia live_runtime.py
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# System stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryStats:
    current_rss_mb: float
    peak_rss_mb: float


@dataclass(frozen=True)
class SystemStats:
    cpu_usage_pct: float
    loadavg_1m: float
    loadavg_5m: float
    loadavg_15m: float
    cpu_temp_c: float | None
    gpu_usage_pct: float | None


class CpuUsageSampler:
    """Estimate whole-system CPU usage from /proc/stat deltas."""

    def __init__(self) -> None:
        self._previous = _read_proc_cpu_totals()

    def sample_pct(self) -> float:
        current = _read_proc_cpu_totals()
        previous = self._previous
        self._previous = current
        if current is None or previous is None:
            return 0.0
        prev_total, prev_idle = previous
        curr_total, curr_idle = current
        total_delta = curr_total - prev_total
        idle_delta = curr_idle - prev_idle
        if total_delta <= 0:
            return 0.0
        busy_delta = max(0, total_delta - idle_delta)
        return max(0.0, min(100.0, (float(busy_delta) / float(total_delta)) * 100.0))


def get_memory_stats() -> MemoryStats:
    current_rss_kib = _read_proc_status_kib("VmRSS:")
    peak_rss_kib = _read_proc_status_kib("VmHWM:")
    if peak_rss_kib is None:
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return MemoryStats(
        current_rss_mb=_bytes_to_mib(0 if current_rss_kib is None else current_rss_kib * 1024),
        peak_rss_mb=_bytes_to_mib(peak_rss_kib * 1024),
    )


def enforce_memory_cap(limit_mb: float, context: str) -> MemoryStats:
    stats = get_memory_stats()
    if limit_mb > 0.0 and max(stats.current_rss_mb, stats.peak_rss_mb) > limit_mb:
        raise MemoryError(
            f"{context} exceeded RAM cap: "
            f"current={stats.current_rss_mb:.2f} MiB, peak={stats.peak_rss_mb:.2f} MiB, "
            f"limit={limit_mb:.2f} MiB"
        )
    return stats


def get_system_stats(cpu_sampler: CpuUsageSampler | None = None) -> SystemStats:
    load_1m = 0.0
    load_5m = 0.0
    load_15m = 0.0
    with contextlib.suppress(OSError):
        load_1m, load_5m, load_15m = os.getloadavg()

    cpu_usage_pct = 0.0 if cpu_sampler is None else cpu_sampler.sample_pct()
    return SystemStats(
        cpu_usage_pct=cpu_usage_pct,
        loadavg_1m=float(load_1m),
        loadavg_5m=float(load_5m),
        loadavg_15m=float(load_15m),
        cpu_temp_c=_read_thermal_celsius("/sys/class/thermal/thermal_zone0/temp"),
        gpu_usage_pct=_read_gpu_busy_percent(),
    )


# ---------------------------------------------------------------------------
# Live metrics collector
# ---------------------------------------------------------------------------


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
    gallery_size: int
    cpu_usage_pct: float
    loadavg_1m: float
    cpu_temp_c: float | None
    gpu_usage_pct: float | None
    current_rss_mb: float
    peak_rss_mb: float
    last_error: str | None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            if isinstance(val, float):
                result[key] = round(val, 3)
            else:
                result[key] = val
        return result


class LiveMetricsCollector:
    """Track rolling live metrics and periodically persist them to JSON."""

    def __init__(
        self,
        *,
        metrics_json_path: Path | None,
        write_every_frames: int,
        det_every: int,
        target_fps: float,
    ) -> None:
        self.metrics_json_path = metrics_json_path
        self.write_every_frames = max(1, write_every_frames)
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
        self._last_error: str | None = None
        self._last_loop_ms = 0.0
        self._last_detect_ms = 0.0
        self._last_track_ms = 0.0
        self._last_embed_ms = 0.0
        self._last_faces = 0
        self._last_frame_mono: float | None = None
        self._last_output_fps = 0.0
        self._cpu_sampler = CpuUsageSampler()

        if self.metrics_json_path is not None:
            self.metrics_json_path.parent.mkdir(parents=True, exist_ok=True)

        self._snapshot = self._build_initial_snapshot()

    def record_frame(
        self,
        *,
        detect_ms: float,
        track_ms: float,
        embed_ms: float,
        face_count: int,
        loop_ms: float,
        memory: MemoryStats,
        gallery_size: int,
    ) -> None:
        should_write: bool
        with self._lock:
            now_mono = time.perf_counter()
            self._frames_processed += 1
            self._sum_loop_ms += loop_ms
            self._sum_detect_ms += detect_ms
            self._sum_track_ms += track_ms
            self._sum_embed_ms += embed_ms
            self._sum_faces += face_count
            self._last_loop_ms = loop_ms
            self._last_detect_ms = detect_ms
            self._last_track_ms = track_ms
            self._last_embed_ms = embed_ms
            self._last_faces = face_count
            if self._last_frame_mono is None:
                self._last_output_fps = 0.0
            else:
                delta_s = now_mono - self._last_frame_mono
                self._last_output_fps = (1.0 / delta_s) if delta_s > 0.0 else 0.0
            self._last_frame_mono = now_mono
            self._snapshot = self._build_snapshot_locked(
                gallery_size=gallery_size, memory=memory
            )
            snapshot = self._snapshot
            should_write = self.metrics_json_path is not None and (
                self._frames_processed % self.write_every_frames == 0
            )
        if should_write:
            self._write_snapshot(snapshot)

    def record_error(self, error: str) -> None:
        with self._lock:
            self._last_error = error

    def snapshot_dict(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot.as_dict()

    def _build_initial_snapshot(self) -> LiveMetricsSnapshot:
        system = get_system_stats(self._cpu_sampler)
        memory = get_memory_stats()
        return LiveMetricsSnapshot(
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
            gallery_size=0,
            cpu_usage_pct=system.cpu_usage_pct,
            loadavg_1m=system.loadavg_1m,
            cpu_temp_c=system.cpu_temp_c,
            gpu_usage_pct=system.gpu_usage_pct,
            current_rss_mb=memory.current_rss_mb,
            peak_rss_mb=memory.peak_rss_mb,
            last_error=None,
        )

    def _build_snapshot_locked(
        self, *, gallery_size: int, memory: MemoryStats
    ) -> LiveMetricsSnapshot:
        frames = self._frames_processed
        uptime = max(0.0, time.perf_counter() - self._start_mono)
        avg_loop_ms = self._sum_loop_ms / frames if frames else 0.0
        avg_output_fps = (float(frames) / uptime) if uptime > 0.0 else 0.0
        processing_fps = (1000.0 / self._last_loop_ms) if self._last_loop_ms > 0.0 else 0.0
        system = get_system_stats(self._cpu_sampler)
        return LiveMetricsSnapshot(
            updated_at_epoch=time.time(),
            uptime_seconds=uptime,
            frames_processed=frames,
            target_fps=self.target_fps,
            det_every=self.det_every,
            current_output_fps=self._last_output_fps,
            avg_output_fps=avg_output_fps,
            current_processing_fps=processing_fps,
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
            gallery_size=gallery_size,
            cpu_usage_pct=system.cpu_usage_pct,
            loadavg_1m=system.loadavg_1m,
            cpu_temp_c=system.cpu_temp_c,
            gpu_usage_pct=system.gpu_usage_pct,
            current_rss_mb=memory.current_rss_mb,
            peak_rss_mb=memory.peak_rss_mb,
            last_error=self._last_error,
        )

    def _write_snapshot(self, snapshot: LiveMetricsSnapshot) -> None:
        if self.metrics_json_path is None:
            return
        self.metrics_json_path.write_text(
            json.dumps(snapshot.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _bytes_to_mib(value_bytes: int) -> float:
    return float(value_bytes) / (1024.0 * 1024.0)


def _read_proc_status_kib(field_name: str) -> int | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(field_name):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _read_proc_cpu_totals() -> tuple[int, int] | None:
    stat_path = Path("/proc/stat")
    if not stat_path.exists():
        return None
    with stat_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line.startswith("cpu "):
        return None
    parts = first_line.split()
    values: list[int] = []
    for raw_value in parts[1:]:
        try:
            values.append(int(raw_value))
        except ValueError:
            return None
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def _read_thermal_celsius(path_str: str) -> float | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        raw_value = path.read_text(encoding="utf-8").strip()
        value_milli_c = int(raw_value)
    except (OSError, ValueError):
        return None
    return float(value_milli_c) / 1000.0


def _read_gpu_busy_percent() -> float | None:
    candidates = (
        "/sys/class/drm/card0/device/gpu_busy_percent",
        "/sys/class/drm/card1/device/gpu_busy_percent",
    )
    for path_str in candidates:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            return float(int(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            continue
    return None
