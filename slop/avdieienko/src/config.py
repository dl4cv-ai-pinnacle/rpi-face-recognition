from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class CaptureConfig:
    resolution: tuple[int, int]
    detection_resolution: tuple[int, int]
    format: str


@dataclass
class DetectionConfig:
    model_path: str
    confidence_threshold: float
    nms_threshold: float
    input_size: tuple[int, int]


@dataclass
class AlignmentConfig:
    output_size: int


@dataclass
class EmbeddingConfig:
    model_path: str
    embedding_dim: int
    input_size: int


@dataclass
class MatchingConfig:
    db_path: str
    index_path: str
    threshold: float


@dataclass
class DisplayConfig:
    show_window: bool
    window_name: str
    box_color: tuple[int, int, int]
    text_color: tuple[int, int, int]
    unknown_label: str


@dataclass
class AppConfig:
    capture: CaptureConfig
    detection: DetectionConfig
    alignment: AlignmentConfig
    embedding: EmbeddingConfig
    matching: MatchingConfig
    display: DisplayConfig
    log_level: str


def load_config(path: str | Path) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        capture=CaptureConfig(
            resolution=tuple(raw["capture"]["resolution"]),
            detection_resolution=tuple(raw["capture"]["detection_resolution"]),
            format=raw["capture"]["format"],
        ),
        detection=DetectionConfig(
            model_path=raw["detection"]["model_path"],
            confidence_threshold=raw["detection"]["confidence_threshold"],
            nms_threshold=raw["detection"]["nms_threshold"],
            input_size=tuple(raw["detection"]["input_size"]),
        ),
        alignment=AlignmentConfig(
            output_size=raw["alignment"]["output_size"],
        ),
        embedding=EmbeddingConfig(
            model_path=raw["embedding"]["model_path"],
            embedding_dim=raw["embedding"]["embedding_dim"],
            input_size=raw["embedding"]["input_size"],
        ),
        matching=MatchingConfig(
            db_path=raw["matching"]["db_path"],
            index_path=raw["matching"]["index_path"],
            threshold=raw["matching"]["threshold"],
        ),
        display=DisplayConfig(
            show_window=raw["display"]["show_window"],
            window_name=raw["display"]["window_name"],
            box_color=tuple(raw["display"]["box_color"]),
            text_color=tuple(raw["display"]["text_color"]),
            unknown_label=raw["display"]["unknown_label"],
        ),
        log_level=raw.get("log_level", "INFO"),
    )
