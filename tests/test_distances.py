import numpy as np
import pytest

from norfair.distances import (
    create_keypoints_voting_distance,
    create_normalized_mean_euclidean_distance,
    get_distance_by_name,
)


def test_frobenius(mock_obj, mock_det):
    fro = get_distance_by_name("frobenius")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(fro(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(fro(det, obj), np.sqrt(1))

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(fro(det, obj), 2)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(fro(det, obj), np.sqrt(4))

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro(det, obj), 2)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro(det, obj), 0)


def test_mean_manhattan(mock_det, mock_obj):
    man = get_distance_by_name("mean_manhattan")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(man(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(man(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(man(det, obj), 1 / 2)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(man(det, obj), 1)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(man(det, obj), 2)

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(man(det, obj), 1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(man(det, obj), 0)


def test_mean_euclidean(mock_det, mock_obj):
    euc = get_distance_by_name("mean_euclidean")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(euc(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(euc(det, obj), 1 / 2)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(euc(det, obj), 1)

    # distance of 2 in 1 dimension of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [5, 4]])
    np.testing.assert_almost_equal(euc(det, obj), 2)

    # distance of 2 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 4], [5, 6]])
    np.testing.assert_almost_equal(euc(det, obj), np.sqrt(8))

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(euc(det, obj), np.sqrt(2))

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc(det, obj), 1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc(det, obj), 0)


def test_iou(mock_det, mock_obj):
    iou = get_distance_by_name("iou")
    iou_opt = get_distance_by_name("iou_opt")

    # perfect match
    det = mock_det([[0, 0], [1, 1]])
    obj = mock_obj([[0, 0], [1, 1]])
    np.testing.assert_almost_equal(iou(det, obj), 0)
    np.testing.assert_almost_equal(iou_opt(det, obj), 0)

    # float type
    det = mock_det([[0.0, 0.0], [1.1, 1.1]])
    obj = mock_obj([[0.0, 0.0], [1.1, 1.1]])
    np.testing.assert_almost_equal(iou(det, obj), 0)
    np.testing.assert_almost_equal(iou_opt(det, obj), 0)

    # det contained in obj
    det = mock_det([[0, 0], [1, 1]])
    obj = mock_obj([[0, 0], [2, 2]])
    np.testing.assert_almost_equal(iou(det, obj), 1 - 1 / 4)
    np.testing.assert_almost_equal(iou_opt(det, obj), 1 - 1 / 4)

    # no overlap
    det = mock_det([[0, 0], [1, 1]])
    obj = mock_obj([[1, 1], [2, 2]])
    np.testing.assert_almost_equal(iou(det, obj), 1)
    np.testing.assert_almost_equal(iou_opt(det, obj), 1)

    # obj fully contained on det
    det = mock_det([[0, 0], [4, 4]])
    obj = mock_obj([[1, 1], [2, 2]])
    np.testing.assert_almost_equal(iou(det, obj), 1 - 1 / 16)
    np.testing.assert_almost_equal(iou_opt(det, obj), 1 - 1 / 16)

    # partial overlap
    det = mock_det([[0, 0], [2, 2]])
    obj = mock_obj([[1, 1], [3, 3]])
    np.testing.assert_almost_equal(iou(det, obj), 1 - 1 / (8 - 1))
    np.testing.assert_almost_equal(iou_opt(det, obj), 1 - 1 / (8 - 1))

    # invalid bbox
    det = mock_det([[0, 0]])
    obj = mock_obj([[0, 0]])
    with pytest.raises(AssertionError):
        iou(det, obj)

    # invalid bbox
    det = mock_det([[0, 0], [1, 1], [2, 2]])
    obj = mock_obj([[0, 0], [2, 2]])
    with pytest.raises(AssertionError):
        iou(det, obj)

    # invalid det bbox, this will be fixed soon
    det = mock_det([[4, 4], [0, 0]])
    obj = mock_obj([[1, 1], [2, 2]])
    with pytest.raises(AssertionError):
        iou(det, obj)

    # invalid obj bbox, this will be fixed soon
    det = mock_det([[0, 0], [4, 4]])
    obj = mock_obj([[4, 4], [0, 0]])
    with pytest.raises(AssertionError):
        iou(det, obj)


def test_keypoint_vote(mock_obj, mock_det):
    vote_d = create_keypoints_voting_distance(
        keypoint_distance_threshold=np.sqrt(8), detection_threshold=0.5
    )

    # perfect match
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 4)  # 3 matches

    # just under distance threshold
    det = mock_det(points=[[0, 0], [1, 1], [2, 2.0]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [4, 3.9]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 4)  # 3 matches

    # just above distance threshold
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [4, 4]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # just under score threshold on detection
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.5])
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.6])
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # just under score threshold on tracked_object's last detection
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.6])
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.5])
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # no match because of scores
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.5)
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=0.5)
    np.testing.assert_almost_equal(vote_d(det, obj), 1)  # 0 matches

    # no match because of distances
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[2, 2], [3, 3], [4, 4]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1)  # 0 matches


def test_normalized_euclidean(mock_obj, mock_det):
    norm_e = create_normalized_mean_euclidean_distance(10, 10)

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.05)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.1)

    # distance of 2 in 1 dimension of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [5, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.2)

    # distance of 2 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 4], [5, 6]])
    np.testing.assert_almost_equal(norm_e(det, obj), np.sqrt(8) / 10)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(norm_e(det, obj), np.sqrt(2) / 10)

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)
