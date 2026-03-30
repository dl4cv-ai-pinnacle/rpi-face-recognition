"""YAML configuration loader with frozen typed dataclasses.

Pattern from Avdieienko's config.py — expanded to cover all pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CaptureConfig:
    resolution: tuple[int, int]
    format: str


@dataclass(frozen=True)
class DetectionConfig:
    backend: str  # "insightface" | "ultraface"
    confidence_threshold: float
    nms_threshold: float


@dataclass(frozen=True)
class AlignmentConfig:
    method: str  # "cv2" | "skimage"
    output_size: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_path: str
    embedding_dim: int
    quantize_int8: bool


@dataclass(frozen=True)
class MatchingConfig:
    threshold: float


@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold: float
    max_missed: int
    smoothing: float
    method: str  # "simple" | "kalman"


@dataclass(frozen=True)
class GalleryConfig:
    root_dir: str
    enrich_margin: float
    enrich_min_quality: float
    enrich_cooldown_seconds: float
    enrich_max_samples: int


@dataclass(frozen=True)
class LiveConfig:
    max_faces: int
    target_fps: float
    det_every: int
    match_threshold: float
    embed_refresh_frames: int
    embed_refresh_iou: float


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True)
class MetricsConfig:
    json_path: str | None
    write_every_frames: int


@dataclass(frozen=True)
class DisplayConfig:
    show_window: bool
    window_name: str


@dataclass(frozen=True)
class AppConfig:
    capture: CaptureConfig
    detection: DetectionConfig
    alignment: AlignmentConfig
    embedding: EmbeddingConfig
    matching: MatchingConfig
    tracking: TrackingConfig
    gallery: GalleryConfig
    live: LiveConfig
    server: ServerConfig
    metrics: MetricsConfig
    display: DisplayConfig
    log_level: str
    memory_limit_mb: float


def load_config(path: str | Path) -> AppConfig:
    """Load and validate YAML config into frozen dataclass tree."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        capture=CaptureConfig(
            resolution=tuple(raw["capture"]["resolution"]),
            format=raw["capture"]["format"],
        ),
        detection=DetectionConfig(
            backend=raw["detection"]["backend"],
            confidence_threshold=raw["detection"]["confidence_threshold"],
            nms_threshold=raw["detection"]["nms_threshold"],
        ),
        alignment=AlignmentConfig(
            method=raw["alignment"]["method"],
            output_size=raw["alignment"]["output_size"],
        ),
        embedding=EmbeddingConfig(
            model_path=raw["embedding"]["model_path"],
            embedding_dim=raw["embedding"]["embedding_dim"],
            quantize_int8=raw["embedding"].get("quantize_int8", False),
        ),
        matching=MatchingConfig(
            threshold=raw["matching"]["threshold"],
        ),
        tracking=TrackingConfig(
            iou_threshold=raw["tracking"]["iou_threshold"],
            max_missed=raw["tracking"]["max_missed"],
            smoothing=raw["tracking"]["smoothing"],
            method=raw["tracking"].get("method", "simple"),
        ),
        gallery=GalleryConfig(
            root_dir=raw["gallery"]["root_dir"],
            enrich_margin=raw["gallery"]["enrich_margin"],
            enrich_min_quality=raw["gallery"]["enrich_min_quality"],
            enrich_cooldown_seconds=raw["gallery"]["enrich_cooldown_seconds"],
            enrich_max_samples=raw["gallery"]["enrich_max_samples"],
        ),
        live=LiveConfig(
            max_faces=raw["live"]["max_faces"],
            target_fps=raw["live"]["target_fps"],
            det_every=raw["live"]["det_every"],
            match_threshold=raw["live"]["match_threshold"],
            embed_refresh_frames=raw["live"]["embed_refresh_frames"],
            embed_refresh_iou=raw["live"]["embed_refresh_iou"],
        ),
        server=ServerConfig(
            host=raw["server"]["host"],
            port=raw["server"]["port"],
        ),
        metrics=MetricsConfig(
            json_path=raw["metrics"].get("json_path"),
            write_every_frames=raw["metrics"]["write_every_frames"],
        ),
        display=DisplayConfig(
            show_window=raw["display"]["show_window"],
            window_name=raw["display"]["window_name"],
        ),
        log_level=raw.get("log_level", "INFO"),
        memory_limit_mb=raw.get("memory_limit_mb", 3072.0),
    )
