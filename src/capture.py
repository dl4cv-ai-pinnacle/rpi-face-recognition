"""Picamera2 frame capture wrapper.

Origins:
- Error handling and logging: Avdieienko src/capture/picamera2.py
- Context manager lifecycle: Shalaiev butler/camera.py
- video_configuration (not preview): Shalaiev butler/camera.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config import CaptureConfig
    from src.contracts import UInt8Array

logger = logging.getLogger(__name__)


class PiCamera2Capture:
    """Captures frames from a Raspberry Pi camera via Picamera2."""

    def __init__(self, config: CaptureConfig) -> None:
        from picamera2 import Picamera2

        self._camera = Picamera2()
        cam_config = self._camera.create_video_configuration(
            main={"format": config.format, "size": config.resolution}
        )
        self._camera.configure(cam_config)
        self._camera.start()
        logger.info(
            "Camera started: %dx%d %s",
            config.resolution[0],
            config.resolution[1],
            config.format,
        )

    def read(self) -> UInt8Array | None:
        """Capture a single frame. Returns None on failure."""
        try:
            frame = self._camera.capture_array("main")
            return np.asarray(frame, dtype=np.uint8)
        except Exception:
            logger.exception("Failed to capture frame")
            return None

    def release(self) -> None:
        self._camera.stop()
        self._camera.close()
        logger.info("Camera released")

    def __enter__(self) -> PiCamera2Capture:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
