from __future__ import annotations

import logging

import numpy as np
from picamera2 import Picamera2

from src.config import CaptureConfig

logger = logging.getLogger(__name__)


class PiCamera2Capture:
    """Frame capture using Picamera2 on Raspberry Pi."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._camera = Picamera2()

        cam_config = self._camera.create_preview_configuration(
            main={
                "format": config.format,
                "size": config.resolution,
            }
        )
        self._camera.configure(cam_config)
        self._camera.start()
        logger.info(
            "Camera started: %dx%d %s",
            config.resolution[0],
            config.resolution[1],
            config.format,
        )

    def read(self) -> np.ndarray | None:
        """Capture a single RGB frame."""
        try:
            return self._camera.capture_array("main")
        except Exception:
            logger.exception("Failed to capture frame")
            return None

    def release(self) -> None:
        """Stop the camera."""
        self._camera.stop()
        self._camera.close()
        logger.info("Camera released")
