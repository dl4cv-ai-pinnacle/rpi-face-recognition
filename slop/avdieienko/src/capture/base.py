from __future__ import annotations

from typing import Protocol

import numpy as np


class FrameCapture(Protocol):
    def read(self) -> np.ndarray | None:
        """Capture a single frame. Returns RGB image (H, W, 3) or None on failure."""
        ...

    def release(self) -> None:
        """Release camera resources."""
        ...
