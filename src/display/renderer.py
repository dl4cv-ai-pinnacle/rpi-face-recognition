from __future__ import annotations

import time

import cv2
import numpy as np

from src.config import DisplayConfig
from src.detection.base import Detection
from src.matching.database import MatchResult


class Renderer:
    """Draw detection results and FPS on frames using OpenCV."""

    def __init__(self, config: DisplayConfig) -> None:
        self.config = config
        self._prev_time = time.monotonic()
        self._fps = 0.0

    def render(
        self,
        frame: np.ndarray,
        results: list[tuple[Detection, MatchResult | None]],
    ) -> None:
        """Draw bounding boxes, names, and FPS, then display the frame.

        Args:
            frame: RGB image (H, W, 3).
            results: List of (Detection, optional MatchResult) pairs.
        """
        # Convert RGB to BGR for OpenCV display
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for detection, match in results:
            x1, y1, x2, y2 = detection.bbox.astype(int)
            color = self.config.box_color

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            if match is not None:
                label = f"{match.name} ({match.score:.2f})"
            else:
                label = self.config.unknown_label

            text_y = max(y1 - 10, 20)
            cv2.putText(
                display,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.config.text_color,
                2,
            )

        # FPS counter
        now = time.monotonic()
        dt = now - self._prev_time
        if dt > 0:
            self._fps = 1.0 / dt
        self._prev_time = now

        cv2.putText(
            display,
            f"FPS: {self._fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if self.config.show_window:
            cv2.imshow(self.config.window_name, display)
