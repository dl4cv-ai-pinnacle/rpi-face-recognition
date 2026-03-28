"""Tests for config loading and frozen enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.config import load_config


def test_load_config_parses_all_sections(tmp_path: Path) -> None:
    """A complete config.yaml produces a fully populated AppConfig."""
    config = load_config("config.yaml")

    assert config.detection.backend == "insightface"
    assert config.alignment.method == "cv2"
    assert config.embedding.embedding_dim == 512
    assert config.matching.threshold == 0.4
    assert config.tracking.smoothing == 0.65
    assert config.gallery.enrich_max_samples == 48
    assert config.live.det_every == 3
    assert config.server.port == 8080
    assert config.memory_limit_mb == 3072.0


def test_config_is_frozen() -> None:
    """Config dataclasses cannot be mutated after creation."""
    config = load_config("config.yaml")

    with pytest.raises(AttributeError):
        config.detection.backend = "ultraface"  # type: ignore[misc]


def test_load_config_rejects_missing_section(tmp_path: Path) -> None:
    """Missing a required section raises KeyError."""
    incomplete = {"capture": {"resolution": [640, 480], "format": "RGB888"}}
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.dump(incomplete))

    with pytest.raises(KeyError):
        load_config(config_path)


def test_load_config_defaults_log_level(tmp_path: Path) -> None:
    """log_level defaults to INFO when omitted."""
    config = load_config("config.yaml")
    assert config.log_level == "INFO"
