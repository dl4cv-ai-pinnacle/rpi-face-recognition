"""Shared config-loading helpers for benchmark scripts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from src.config import AppConfig, load_config


def auto_cast(value: str) -> object:
    """Cast a CLI string to bool / int / float / str."""
    lower = value.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(raw: dict[str, object], overrides: list[str]) -> None:
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
        target[parts[-1]] = auto_cast(value_str)


def load_config_with_overrides(config_path: str, overrides: list[str]) -> AppConfig:
    """Load a YAML config, apply CLI overrides, return typed ``AppConfig``."""
    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, object] = yaml.safe_load(f)
    if overrides:
        apply_overrides(raw, overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(raw, tmp)
        tmp_path = tmp.name
    config = load_config(tmp_path)
    Path(tmp_path).unlink()
    return config
