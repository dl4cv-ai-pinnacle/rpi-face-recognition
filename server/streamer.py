"""MJPEG camera streamer — capture frames, process, publish as JPEG.

Origin: Valenia scripts/live_camera_server.py — CameraStreamer class.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from src.config import AppConfig
from src.display import annotate_faces
from src.live import LiveRuntime

logger = logging.getLogger(__name__)


@dataclass
class StreamSnapshot:
    frame_id: int
    jpeg_bytes: bytes | None
    error: str | None


class CameraStreamer:
    """Capture frames in one background loop and keep only the latest JPEG."""

    def __init__(self, runtime: LiveRuntime, config: AppConfig) -> None:
        self._runtime = runtime
        self._config = config
        self._camera: object | None = None
        self._condition = threading.Condition()
        self._frame_id = 0
        self._jpeg_bytes: bytes | None = None
        self._error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_interval = (
            1.0 / config.live.target_fps if config.live.target_fps > 0 else 0.0
        )

    @property
    def runtime(self) -> LiveRuntime:
        return self._runtime

    def start(self) -> None:
        from picamera2 import Picamera2

        try:
            self._camera = Picamera2()
        except IndexError as exc:
            raise RuntimeError(
                "No camera detected by Picamera2. Check cable/enablement and rerun."
            ) from exc

        resolution = self._config.capture.resolution
        cam_config = self._camera.create_video_configuration(  # type: ignore[union-attr]
            main={"size": resolution, "format": self._config.capture.format}
        )
        self._camera.configure(cam_config)  # type: ignore[union-attr]
        self._camera.start()  # type: ignore[union-attr]
        time.sleep(1.0)

        self._thread = threading.Thread(
            target=self._capture_loop, name="live-camera", daemon=True
        )
        self._thread.start()
        logger.info("Camera streamer started at %dx%d", resolution[0], resolution[1])

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._camera is not None:
            self._camera.stop()  # type: ignore[union-attr]
        with self._condition:
            self._condition.notify_all()

    def wait_for_frame(self, last_seen: int, timeout: float = 5.0) -> StreamSnapshot:
        with self._condition:
            if (
                self._frame_id == last_seen
                and self._error is None
                and not self._stop_event.is_set()
            ):
                self._condition.wait(timeout=timeout)
            return StreamSnapshot(
                frame_id=self._frame_id,
                jpeg_bytes=self._jpeg_bytes,
                error=self._error,
            )

    def _capture_loop(self) -> None:
        camera = self._camera
        if camera is None:
            self._publish(None, error="Camera was not initialized")
            return

        next_frame_due = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                loop_t0 = time.perf_counter()
                frame_rgb = np.asarray(camera.capture_array(), dtype=np.uint8)  # type: ignore[union-attr]
                frame_bgr = np.asarray(
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8
                )
                frame_result = self._runtime.process_frame(frame_bgr)

                tracks = [f.track for f in frame_result.overlay.faces]
                matches = [f.match for f in frame_result.overlay.faces]
                annotate_faces(frame_bgr, tracks, matches)

                ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("OpenCV failed to encode a JPEG frame")

                loop_ms = (time.perf_counter() - loop_t0) * 1000.0
                self._runtime.record_frame_metrics(
                    frame_result=frame_result, loop_ms=loop_ms
                )
                self._publish(encoded.tobytes())
            except Exception as exc:
                self._runtime.record_error(str(exc))
                self._publish(None, error=str(exc))
                break

            if self._frame_interval <= 0:
                continue
            next_frame_due += self._frame_interval
            sleep_for = next_frame_due - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame_due = time.perf_counter()

    def _publish(self, jpeg_bytes: bytes | None, error: str | None = None) -> None:
        with self._condition:
            self._frame_id += 1
            self._jpeg_bytes = jpeg_bytes
            self._error = error
            self._condition.notify_all()
