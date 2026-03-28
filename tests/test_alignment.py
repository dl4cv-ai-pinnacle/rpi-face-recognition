"""Tests for face alignment — cv2, skimage, and center-crop fallback."""

from __future__ import annotations

import numpy as np
import pytest
from src.alignment import Cv2Aligner, center_crop_fallback, create_aligner
from src.config import AlignmentConfig

from tests.conftest import make_frame, make_landmarks


class TestCv2Aligner:
    def test_produces_112x112_crop(self) -> None:
        aligner = Cv2Aligner()
        frame = make_frame()
        landmarks = make_landmarks()

        result = aligner.align(frame, landmarks)

        assert result is not None
        assert result.shape == (112, 112, 3)
        assert result.dtype == np.uint8

    def test_rejects_wrong_landmark_count(self) -> None:
        aligner = Cv2Aligner()
        frame = make_frame()
        bad_landmarks = np.array([[100, 100], [200, 200]], dtype=np.float32)

        result = aligner.align(frame, bad_landmarks)

        assert result is None

    def test_name(self) -> None:
        assert Cv2Aligner().name == "cv2"


class TestCenterCropFallback:
    def test_produces_correct_shape(self) -> None:
        frame = make_frame(480, 640)

        crop = center_crop_fallback(frame, 100.0, 100.0, 300.0, 300.0)

        assert crop.shape == (112, 112, 3)
        assert crop.dtype == np.uint8

    def test_handles_bbox_at_frame_edge(self) -> None:
        frame = make_frame(480, 640)

        crop = center_crop_fallback(frame, -10.0, -10.0, 100.0, 100.0)

        assert crop.shape == (112, 112, 3)

    def test_handles_zero_area_bbox(self) -> None:
        frame = make_frame(480, 640)

        crop = center_crop_fallback(frame, 100.0, 100.0, 100.0, 100.0)

        assert crop.shape == (112, 112, 3)
        assert np.all(crop == 0)


class TestCreateAligner:
    def test_creates_cv2_by_default(self) -> None:
        config = AlignmentConfig(method="cv2", output_size=112)
        aligner = create_aligner(config)
        assert aligner.name == "cv2"

    def test_creates_skimage(self) -> None:
        config = AlignmentConfig(method="skimage", output_size=112)
        aligner = create_aligner(config)
        assert aligner.name == "skimage"

    def test_rejects_unknown_method(self) -> None:
        config = AlignmentConfig(method="magic", output_size=112)
        with pytest.raises(ValueError, match="Unknown alignment method"):
            create_aligner(config)
