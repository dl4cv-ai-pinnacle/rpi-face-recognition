"""ArcFace 5-point face alignment utility."""

from __future__ import annotations

import numpy as np
import cv2

from butler.detectors.base import Landmark

# Standard ArcFace reference landmarks in 112x112 space.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def align_face(
    frame_rgb: np.ndarray,
    landmarks: tuple[Landmark, ...],
) -> np.ndarray | None:
    """Align a face to 112x112 using 5-point landmarks and ArcFace template.

    Returns the aligned 112x112 RGB crop, or None if the transform fails.
    """
    if len(landmarks) != 5:
        return None

    src_pts = np.array(
        [[lm.x, lm.y] for lm in landmarks],
        dtype=np.float32,
    )

    transform, inliers = cv2.estimateAffinePartial2D(
        src_pts, ARCFACE_TEMPLATE_112, method=cv2.LMEDS,
    )
    if transform is None:
        return None

    aligned = cv2.warpAffine(
        frame_rgb, transform, (112, 112), borderValue=(0, 0, 0),
    )
    return aligned
