from __future__ import annotations

import numpy as np
from quality import compute_face_quality


def test_high_confidence_large_face_scores_high() -> None:
    box = np.asarray([0.0, 0.0, 200.0, 200.0, 0.99], dtype=np.float32)
    q = compute_face_quality(box)
    assert q.score > 0.6
    assert q.det_confidence > 0.9
    assert q.face_area == 40000.0


def test_low_confidence_scores_low() -> None:
    box = np.asarray([0.0, 0.0, 200.0, 200.0, 0.10], dtype=np.float32)
    q = compute_face_quality(box)
    assert q.score < 0.4


def test_tiny_face_scores_zero() -> None:
    box = np.asarray([0.0, 0.0, 10.0, 10.0, 0.99], dtype=np.float32)
    q = compute_face_quality(box)
    assert q.score == 0.0
    assert q.face_area == 100.0


def test_zero_area_scores_zero() -> None:
    box = np.asarray([5.0, 5.0, 5.0, 5.0, 0.99], dtype=np.float32)
    q = compute_face_quality(box)
    assert q.score == 0.0
    assert q.face_area == 0.0
