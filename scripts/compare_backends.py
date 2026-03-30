#!/usr/bin/env python3
"""Compare two pipeline configurations on the same benchmark image.

Usage example:
    uv run --python 3.13 python scripts/compare_backends.py \
        --config-a config.yaml --override-a embedding.quantize_int8=false --label-a fp32 \
        --config-b config.yaml --override-b embedding.quantize_int8=true --label-b int8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from src.config import AppConfig, load_config
from src.contracts import UInt8Array
from src.metrics import MemoryStats, enforce_memory_cap
from src.pipeline import FacePipeline, build_pipeline

# ---------------------------------------------------------------------------
# Config override helpers
# ---------------------------------------------------------------------------


def _auto_cast(value: str) -> object:
    """Cast a CLI string to bool / int / float / str automatically."""
    lower = value.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    if lower == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_overrides(raw: dict[str, object], overrides: list[str]) -> None:
    """Apply dotted key=value overrides to a nested dict.

    Example: ``embedding.quantize_int8=true`` sets
    ``raw["embedding"]["quantize_int8"] = True``.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        key, _, value_str = override.partition("=")
        parts = key.split(".")
        target: dict[str, object] = raw
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                raise KeyError(f"Cannot traverse into {part!r} (full key: {key!r})")
            target = child
        target[parts[-1]] = _auto_cast(value_str)


def _load_config_with_overrides(config_path: str, overrides: list[str]) -> AppConfig:
    """Load a YAML config, apply CLI overrides, and return a typed AppConfig."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _apply_overrides(raw, overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(raw, tmp)
        tmp_path = tmp.name
    config = load_config(tmp_path)
    Path(tmp_path).unlink()
    return config


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSummary:
    label: str
    avg_loop_ms: float
    avg_detect_ms: float
    avg_embed_ms: float
    avg_faces: float
    avg_fps: float
    final_memory: MemoryStats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="data/lena.jpg")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Unmeasured warmup iterations before collecting timings; values <0 are treated as 0",
    )
    parser.add_argument("--ram-cap-mb", type=float, default=4096.0)
    parser.add_argument("--output-json", default="")

    parser.add_argument("--config-a", default="config.yaml", help="YAML config for variant A")
    parser.add_argument(
        "--override-a",
        action="append",
        default=[],
        help="Dotted key=value override for variant A (repeatable)",
    )
    parser.add_argument("--label-a", default="A")

    parser.add_argument("--config-b", default="config.yaml", help="YAML config for variant B")
    parser.add_argument(
        "--override-b",
        action="append",
        default=[],
        help="Dotted key=value override for variant B (repeatable)",
    )
    parser.add_argument("--label-b", default="B")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------


def benchmark_variant(
    *,
    label: str,
    pipeline: FacePipeline,
    image: UInt8Array,
    runs: int,
    warmup_runs: int,
    face_limit: int,
    ram_cap_mb: float,
) -> VariantSummary:
    loop_ms: list[float] = []
    detect_latencies_ms: list[float] = []
    embed_latencies_ms: list[float] = []
    faces: list[int] = []
    final_memory = enforce_memory_cap(ram_cap_mb, f"{label} comparison startup")
    total_runs = warmup_runs + max(1, runs)

    for idx in range(total_runs):
        t0 = time.perf_counter()
        det, detect_latency_ms = pipeline.detect(image)
        embed_ms_total = 0.0
        if det.kps is not None:
            for face_idx in range(min(len(det.boxes), face_limit)):
                _, embed_latency_ms = pipeline.embed_from_kps(image, det.kps[face_idx])
                embed_ms_total += embed_latency_ms
        if idx >= warmup_runs:
            loop_ms.append((time.perf_counter() - t0) * 1000.0)
            detect_latencies_ms.append(detect_latency_ms)
            embed_latencies_ms.append(embed_ms_total)
            faces.append(len(det.boxes))
            measured_idx = idx - warmup_runs + 1
            if measured_idx % 10 == 0 or measured_idx == runs:
                final_memory = enforce_memory_cap(
                    ram_cap_mb,
                    f"{label} comparison iteration {measured_idx}",
                )

    loop_avg = statistics.fmean(loop_ms)
    avg_fps = 1000.0 / loop_avg if loop_avg > 0.0 else 0.0
    return VariantSummary(
        label=label,
        avg_loop_ms=loop_avg,
        avg_detect_ms=statistics.fmean(detect_latencies_ms) if detect_latencies_ms else 0.0,
        avg_embed_ms=statistics.fmean(embed_latencies_ms) if embed_latencies_ms else 0.0,
        avg_faces=statistics.fmean(faces) if faces else 0.0,
        avg_fps=avg_fps,
        final_memory=final_memory,
    )


def as_dict(summary: VariantSummary) -> dict[str, float | str]:
    return {
        "label": summary.label,
        "avg_loop_ms": round(summary.avg_loop_ms, 3),
        "avg_detect_ms": round(summary.avg_detect_ms, 3),
        "avg_embed_ms": round(summary.avg_embed_ms, 3),
        "avg_faces": round(summary.avg_faces, 3),
        "avg_fps": round(summary.avg_fps, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        args = parse_args()
        image_path = ROOT / args.image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path}")
            return 2

        config_a = _load_config_with_overrides(args.config_a, args.override_a)
        config_b = _load_config_with_overrides(args.config_b, args.override_b)

        enforce_memory_cap(args.ram_cap_mb, "comparison startup")
        pipeline_a = build_pipeline(config_a)
        pipeline_b = build_pipeline(config_b)
        enforce_memory_cap(args.ram_cap_mb, "comparison initialization")

        frame = np.asarray(image, dtype=np.uint8)
        summary_a = benchmark_variant(
            label=args.label_a,
            pipeline=pipeline_a,
            image=frame,
            runs=max(1, args.runs),
            warmup_runs=max(0, args.warmup_runs),
            face_limit=config_a.live.max_faces,
            ram_cap_mb=args.ram_cap_mb,
        )
        summary_b = benchmark_variant(
            label=args.label_b,
            pipeline=pipeline_b,
            image=frame,
            runs=max(1, args.runs),
            warmup_runs=max(0, args.warmup_runs),
            face_limit=config_b.live.max_faces,
            ram_cap_mb=args.ram_cap_mb,
        )

        print("\n=== Variant comparison ===")
        print(json.dumps({"a": as_dict(summary_a), "b": as_dict(summary_b)}, indent=2))
        print("\n=== Delta (b - a) ===")
        print(f"avg loop latency: {summary_b.avg_loop_ms - summary_a.avg_loop_ms:.3f} ms")
        print(f"avg detection latency: {summary_b.avg_detect_ms - summary_a.avg_detect_ms:.3f} ms")
        print(f"avg embedding latency: {summary_b.avg_embed_ms - summary_a.avg_embed_ms:.3f} ms")
        print(f"avg FPS: {summary_b.avg_fps - summary_a.avg_fps:.3f}")
        print("memory note: use benchmark_pipeline.py for isolated RSS comparisons per variant")

        if args.output_json:
            out_path = ROOT / args.output_json
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "a": as_dict(summary_a),
                        "b": as_dict(summary_b),
                        "delta": {
                            "avg_loop_ms": summary_b.avg_loop_ms - summary_a.avg_loop_ms,
                            "avg_detect_ms": summary_b.avg_detect_ms - summary_a.avg_detect_ms,
                            "avg_embed_ms": summary_b.avg_embed_ms - summary_a.avg_embed_ms,
                            "avg_fps": summary_b.avg_fps - summary_a.avg_fps,
                        },
                        "config": {
                            "image": args.image,
                            "runs": int(max(1, args.runs)),
                            "warmup_runs": int(max(0, args.warmup_runs)),
                            "ram_cap_mb": float(args.ram_cap_mb),
                        },
                        "notes": {
                            "memory_comparison": (
                                "Not reported here because both variants run in one process. "
                                "Use benchmark_pipeline.py with --output-json for "
                                "isolated RSS numbers."
                            )
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved JSON report: {out_path}")
        return 0
    except MemoryError as exc:
        print(exc)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
