"""Runtime helpers for noisy native libraries on Debian ARM."""

from __future__ import annotations

import contextlib
import os
import resource
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]
type Int32Array = npt.NDArray[np.int32]
type UInt8Array = npt.NDArray[np.uint8]


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


@dataclass(frozen=True)
class QuantizationReport:
    model_path: str
    quantized_model_path: str
    reused_existing: bool
    quantize_ms: float
    original_size_mb: float
    quantized_size_mb: float

    @property
    def size_delta_mb(self) -> float:
        return self.quantized_size_mb - self.original_size_mb


def get_memory_stats() -> MemoryStats:
    current_rss_kib = _read_proc_status_kib("VmRSS:")
    peak_rss_kib = _read_proc_status_kib("VmHWM:")
    if peak_rss_kib is None:
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return MemoryStats(
        current_rss_mb=_bytes_to_mib(0 if current_rss_kib is None else current_rss_kib * 1024),
        peak_rss_mb=_bytes_to_mib(peak_rss_kib * 1024),
    )


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


def enforce_memory_cap(limit_mb: float, context: str) -> MemoryStats:
    stats = get_memory_stats()
    if limit_mb > 0.0 and max(stats.current_rss_mb, stats.peak_rss_mb) > limit_mb:
        raise MemoryError(
            f"{context} exceeded RAM cap: "
            f"current={stats.current_rss_mb:.2f} MiB, peak={stats.peak_rss_mb:.2f} MiB, "
            f"limit={limit_mb:.2f} MiB"
        )
    return stats


def quantize_onnx_model(
    model_path: Path,
    quantized_path: Path | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, QuantizationReport]:
    if quantized_path is None:
        quantized_path = model_path.with_name(f"{model_path.stem}.int8{model_path.suffix}")
    if quantized_path.resolve() == model_path.resolve():
        raise ValueError("Quantized model path must differ from the source model path")

    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    reused_existing = quantized_path.exists() and not overwrite
    quantize_ms = 0.0

    if not reused_existing:
        try:
            with suppress_stderr_fd():
                from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as exc:
            raise RuntimeError(
                "ONNX quantization is not available in this environment. "
                "The current Debian python3-onnx/python3-onnxruntime combination "
                "fails to import onnxruntime.quantization on this Pi. "
                "Use a compatible external build environment to create the quantized "
                "model, then rerun with --rec-model pointing at that file."
            ) from exc

        t0 = time.perf_counter()
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(quantized_path),
            op_types_to_quantize=["MatMul", "Gemm"],
            weight_type=QuantType.QInt8,
            per_channel=False,
        )
        quantize_ms = (time.perf_counter() - t0) * 1000.0

    report = QuantizationReport(
        model_path=str(model_path),
        quantized_model_path=str(quantized_path),
        reused_existing=reused_existing,
        quantize_ms=quantize_ms,
        original_size_mb=_bytes_to_mib(model_path.stat().st_size),
        quantized_size_mb=_bytes_to_mib(quantized_path.stat().st_size),
    )
    return quantized_path, report


@contextlib.contextmanager
def suppress_stderr_fd(enabled: bool = True) -> Iterator[None]:
    """Suppress C/C++ stderr output by temporarily redirecting file descriptor 2."""
    if not enabled:
        yield
        return

    saved_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


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
