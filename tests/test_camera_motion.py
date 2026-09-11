import numpy as np

from norfair.camera_motion import HomographyTransformation


def _translation_homography(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=float)


def test_homography_rel_to_abs_accepts_single_point():
    homography = HomographyTransformation(_translation_homography(10, 20))
    point = np.array([5.0, 7.0])

    result_1d = homography.rel_to_abs(point)
    result_2d = homography.rel_to_abs(point.reshape(1, 2))

    assert result_1d.shape == (2,)
    assert result_2d.shape == (1, 2)
    np.testing.assert_allclose(result_1d, result_2d[0])
    np.testing.assert_allclose(result_1d, np.array([-5.0, -13.0]))


def test_homography_abs_to_rel_accepts_single_point():
    homography = HomographyTransformation(_translation_homography(10, 20))
    point = np.array([5.0, 7.0])

    result_1d = homography.abs_to_rel(point)
    result_2d = homography.abs_to_rel(point.reshape(1, 2))

    assert result_1d.shape == (2,)
    assert result_2d.shape == (1, 2)
    np.testing.assert_allclose(result_1d, result_2d[0])
    np.testing.assert_allclose(result_1d, np.array([15.0, 27.0]))


def test_homography_multiple_points_unchanged():
    homography = HomographyTransformation(_translation_homography(1, 2))
    points = np.array([[0.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        homography.abs_to_rel(points), np.array([[1.0, 2.0], [4.0, 6.0]])
    )
    np.testing.assert_allclose(
        homography.rel_to_abs(points), np.array([[-1.0, -2.0], [2.0, 2.0]])
    )
