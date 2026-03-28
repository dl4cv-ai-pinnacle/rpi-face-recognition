"""ArcFace-compatible 5-point face alignment."""

from __future__ import annotations

import cv2
import numpy as np
from runtime_utils import Float32Array, UInt8Array
from skimage import transform as trans

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(landmarks: Float32Array, image_size: int = 112) -> Float32Array:
    if landmarks.shape != (5, 2):
        raise ValueError(f"Expected landmarks shape (5,2), got {landmarks.shape}")
    if image_size % 112 != 0 and image_size % 128 != 0:
        raise ValueError("image_size must be a multiple of 112 or 128")

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    if not tform.estimate(landmarks, dst):
        raise RuntimeError("Failed to estimate face alignment transform")
    params = tform.params
    if params is None:
        raise RuntimeError("Alignment transform returned no parameters")
    return np.asarray(params[0:2, :], dtype=np.float32)


def norm_crop(image: UInt8Array, landmarks: Float32Array, image_size: int = 112) -> UInt8Array:
    matrix = estimate_norm(landmarks=landmarks, image_size=image_size)
    cropped = cv2.warpAffine(image, matrix, (image_size, image_size), borderValue=0.0)
    return np.asarray(cropped, dtype=np.uint8)
