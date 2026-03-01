"""Picamera2 wrapper for frame capture."""

from picamera2 import Picamera2
import numpy as np


class Camera:
    """Captures RGB frames from a Raspberry Pi camera via Picamera2."""

    def __init__(self, resolution: tuple[int, int] = (640, 480)):
        self._resolution = resolution
        self._cam: Picamera2 | None = None

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def start(self) -> None:
        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": self._resolution, "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None

    def capture(self) -> np.ndarray:
        """Return the latest frame as an RGB numpy array."""
        assert self._cam is not None, "Camera not started"
        return self._cam.capture_array()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
