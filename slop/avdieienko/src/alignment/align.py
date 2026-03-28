from __future__ import annotations

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

# Standard ArcFace alignment template for 112x112 output
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


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """Warp face to aligned crop using similarity transform.

    Args:
        image: RGB image (H, W, 3) uint8.
        landmarks: Detected landmarks, shape (5, 2).
        output_size: Output square size (default 112 for ArcFace).

    Returns:
        Aligned face crop (output_size, output_size, 3) uint8.
    """
    ref = ARCFACE_REF_LANDMARKS.copy()

    # Scale reference landmarks if output size differs from default 112
    if output_size != 112:
        scale = output_size / 112.0
        ref *= scale

    tform = SimilarityTransform()
    tform.estimate(landmarks, ref)

    matrix = tform.params[:2]
    aligned = cv2.warpAffine(image, matrix, (output_size, output_size))
    return aligned
