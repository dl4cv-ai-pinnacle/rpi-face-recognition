"""Face alignment to canonical ArcFace 112×112 template.

Two swappable solvers behind the same interface, plus a center-crop fallback
for detectors that produce no landmarks (e.g., UltraFace).

Origins:
- align_cv2: Shalaiev butler/recognition/align.py (estimateAffinePartial2D with LMEDS)
- align_skimage: Valenia src/face_align.py (SimilarityTransform, 128-size support)
- center_crop_fallback: new — honest fallback for landmark-free detectors
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.contracts import Float32Array, UInt8Array

if TYPE_CHECKING:
    from src.config import AlignmentConfig

# Standard ArcFace 5-point reference in 112×112 space.
ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# cv2 solver (default) — from Shalaiev
# ---------------------------------------------------------------------------


class Cv2Aligner:
    """Alignment using cv2.estimateAffinePartial2D with LMEDS robust estimator."""

    @property
    def name(self) -> str:
        return "cv2"

    def align(
        self, frame_bgr: UInt8Array, landmarks: Float32Array, /
    ) -> UInt8Array | None:
        if landmarks.shape != (5, 2):
            return None
        transform, _ = cv2.estimateAffinePartial2D(
            landmarks, ARCFACE_REF_LANDMARKS, method=cv2.LMEDS
        )
        aligned = cv2.warpAffine(frame_bgr, transform, (112, 112), borderValue=(0, 0, 0))
        return np.asarray(aligned, dtype=np.uint8)


# ---------------------------------------------------------------------------
# skimage solver (alternative) — from Valenia
# ---------------------------------------------------------------------------


class SkimageAligner:
    """Alignment using skimage SimilarityTransform. Handles 128-aligned sizes."""

    @property
    def name(self) -> str:
        return "skimage"

    def align(
        self, frame_bgr: UInt8Array, landmarks: Float32Array, /, image_size: int = 112
    ) -> UInt8Array | None:
        if landmarks.shape != (5, 2):
            return None

        from skimage import transform as trans

        if image_size % 112 == 0:
            ratio = float(image_size) / 112.0
            diff_x = 0.0
        elif image_size % 128 == 0:
            ratio = float(image_size) / 128.0
            diff_x = 8.0 * ratio
        else:
            return None

        dst = ARCFACE_REF_LANDMARKS * ratio
        dst[:, 0] += diff_x
        tform = trans.SimilarityTransform()
        if not tform.estimate(landmarks, dst):
            return None
        params = tform.params
        if params is None:
            return None
        matrix = np.asarray(params[0:2, :], dtype=np.float32)
        cropped = cv2.warpAffine(frame_bgr, matrix, (image_size, image_size), borderValue=0.0)
        return np.asarray(cropped, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Center-crop fallback for landmark-free detectors
# ---------------------------------------------------------------------------


def center_crop_fallback(
    frame_bgr: UInt8Array,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    output_size: int = 112,
) -> UInt8Array:
    """Crop face bbox from frame and resize to output_size × output_size.

    Used when the detector does not produce landmarks. No rotation correction,
    but no compounded estimation error either — more predictable than synthetic
    landmark estimation for non-frontal faces.
    """
    h, w = frame_bgr.shape[:2]
    cx1 = max(0, int(x1))
    cy1 = max(0, int(y1))
    cx2 = min(w, int(x2))
    cy2 = min(h, int(y2))
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)
    resized = cv2.resize(crop, (output_size, output_size))
    return np.asarray(resized, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_aligner(config: AlignmentConfig) -> Cv2Aligner | SkimageAligner:
    """Create an aligner based on config.method."""
    match config.method:
        case "cv2":
            return Cv2Aligner()
        case "skimage":
            return SkimageAligner()
        case _:
            msg = f"Unknown alignment method: {config.method!r}"
            raise ValueError(msg)
