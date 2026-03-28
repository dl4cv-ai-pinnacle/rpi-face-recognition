"""Django-style settings loaded from .env file."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import os

# Load .env from the code/ directory (next to pyproject.toml).
_CODE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_CODE_DIR / ".env")


def _parse_resolution(value: str) -> tuple[int, int]:
    w, h = value.split("x")
    return int(w), int(h)


CAPTURE_RESOLUTION: tuple[int, int] = _parse_resolution(
    os.getenv("CAPTURE_RESOLUTION", "640x480")
)

DETECTION_RESOLUTION: tuple[int, int] = _parse_resolution(
    os.getenv("DETECTION_RESOLUTION", "320x320")
)

FACE_DETECTOR: str = os.getenv("FACE_DETECTOR", "scrfd")

DETECTION_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.5")
)

MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", str(_CODE_DIR / "models")))

WINDOW_TITLE: str = os.getenv("WINDOW_TITLE", "Butler")

# --- Recognition ---
FACE_EMBEDDER: str = os.getenv("FACE_EMBEDDER", "mobilefacenet")

RECOGNITION_THRESHOLD: float = float(
    os.getenv("RECOGNITION_THRESHOLD", "0.4")
)

DATABASE_DIR: Path = Path(os.getenv("DATABASE_DIR", str(_CODE_DIR / "data")))
