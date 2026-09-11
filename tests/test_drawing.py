import numpy as np
import pytest

from norfair.camera_motion import HomographyTransformation, TranslationTransformation
from norfair.drawing.color import hex_to_bgr
from norfair.drawing.fixed_camera import FixedCamera


def test_hex_parsing():
    assert hex_to_bgr("#010203") == (3, 2, 1)
    assert hex_to_bgr("#123") == (51, 34, 17)  # (16*3+3, 16*2+2, 16*1+1)
    assert hex_to_bgr("#ffffff") == (255, 255, 255)


def test_fixed_camera_with_homography_identity():
    # MotionEstimator defaults to HomographyTransformation; FixedCamera used to
    # crash because it passes a single (2,) corner into rel_to_abs.
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[0, 0] = [1, 2, 3]
    frame[-1, -1] = [4, 5, 6]
    out = FixedCamera(scale=2).adjust_frame(frame, HomographyTransformation(np.eye(3)))
    assert out.shape == (40, 60, 3)
    np.testing.assert_array_equal(out[10:30, 15:45], frame)


def test_fixed_camera_with_translation():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[0, 0] = [7, 8, 9]
    out = FixedCamera(scale=2).adjust_frame(
        frame, TranslationTransformation(np.array([5.0, 0.0]))
    )
    np.testing.assert_array_equal(out[10, 10], [7, 8, 9])
