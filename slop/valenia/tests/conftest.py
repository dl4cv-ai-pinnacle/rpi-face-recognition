from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


def _noop(*args: object, **kwargs: object) -> None:
    del args, kwargs


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__dict__["FONT_HERSHEY_SIMPLEX"] = 0
    cv2_stub.__dict__["IMREAD_COLOR"] = 1
    cv2_stub.__dict__["rectangle"] = _noop
    cv2_stub.__dict__["putText"] = _noop
    cv2_stub.__dict__["circle"] = _noop
    cv2_stub.__dict__["imdecode"] = _noop
    cv2_stub.__dict__["imencode"] = lambda ext, frame: (True, np.asarray([1, 2, 3], dtype=np.uint8))
    sys.modules["cv2"] = cv2_stub
