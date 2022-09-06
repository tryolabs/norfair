"""Predefined distances"""
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .tracker import Detection, TrackedObject


def frobenius(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Frobernius norm on the difference of the points in detection and the estimates in tracked_object.

    The Frobenius distance and norm are given by:

    $$
    d_f(a, b) = ||a - b||_F
    $$

    $$
    ||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}
    $$

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    return np.linalg.norm(detection.points - tracked_object.estimate)


def mean_euclidean(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Average euclidean distance between the points in detection and estimates in tracked_object.

    $$
    d(a, b) = \\frac{\\sum_{i=0}^N ||a_i - b_i||_2}{N}
    $$

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    return np.linalg.norm(detection.points - tracked_object.estimate, axis=1).mean()


def mean_manhattan(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Average manhattan distance between the points in detection and the estimates in tracked_object

    Given by:

    $$
    d(a, b) = \\frac{\\sum_{i=0}^N ||a_i - b_i||_1}{N}
    $$

    Where $||a||_1$ is the manhattan norm.

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        a tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    return np.linalg.norm(
        detection.points - tracked_object.estimate, ord=1, axis=1
    ).mean()


def _validate_bboxes(bbox: np.ndarray):
    """Validates that the numpy array a is a valid bounding box"""
    assert bbox.shape == (
        2,
        2,
    ), f"incorrect bbox, expecting shape (2, 2) but received {bbox.shape}"

    assert bbox[0, 0] < bbox[1, 0] and bbox[0, 1] < bbox[1, 1], f"incorrect bbox {bbox}"


def _iou(box_a, box_b) -> float:
    """
    Underlying iou distance. See `Norfair.distances.iou`.
    """

    # Detection points will be box A
    # Tracked objects point will be box B.
    box_a = np.concatenate(box_a)
    box_b = np.concatenate(box_b)
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # Compute the area of both the prediction and tracker
    # rectangles
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + tracker
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # Since 0 <= IoU <= 1, we define 1-IoU as a distance.
    # Distance values will be in [0, 1]
    return 1 - iou


def iou(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Intersection over union distance between the bounding boxes.

    Assumes that `detection.points` (and by consecuence `tracked_object.estimate`)
    define a bounding box in the form `[[x0, y0], [x1, y1]]`.

    Normal IoU is 1 when the boxes are the same and 0 when they don't overlap,
    to transform that into a distance that makes sense we return `1 - iou`.

    Performs checks that the bounding boxes are valid to give better error messages.
    For a faster implementation without checks use [`iou_opt`][norfair.distances.iou_opt].

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object.

    Returns
    -------
    float
        The distance.
    """
    boxa = detection.points.copy()
    boxa.sort(axis=0)
    _validate_bboxes(boxa)
    boxb = tracked_object.estimate.copy()
    boxb.sort(axis=0)
    _validate_bboxes(boxb)
    return _iou(boxa, boxb)


def iou_opt(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Optimized version of [`iou`][norfair.distances.iou].

    Performs faster but errors might be cryptic if the bounding boxes are not valid.

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object.

    Returns
    -------
    float
        The distance.
    """
    return _iou(detection.points, tracked_object.estimate)


_DISTANCE_FUNCTIONS = {
    "frobenius": frobenius,
    "mean_manhattan": mean_manhattan,
    "mean_euclidean": mean_euclidean,
    "iou": iou,
    "iou_opt": iou_opt,
}


def get_distance_by_name(name: str) -> Callable[["Detection", "TrackedObject"], float]:
    """
    Select a distance by name.

    Valid names are: `["frobenius", "mean_euclidean", "mean_manhattan", "iou", "iou_opt"]`.
    """
    try:
        return _DISTANCE_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"Invalid distance '{name}', expecting one of {_DISTANCE_FUNCTIONS.keys()}"
        )


def create_keypoints_voting_distance(
    keypoint_distance_threshold: float, detection_threshold: float
) -> Callable[["Detection", "TrackedObject"], float]:
    """
    Construct a keypoint voting distance function configured with the thresholds.

    Count how many points in a detection match the with a tracked_object.
    A match is considered when distance between the points is < `keypoint_distance_threshold`
    and the score of the last_detection of the tracked_object is > `detection_threshold`.
    Notice the if multiple points are tracked, the ith point in detection can only match the ith
    point in the tracked object.

    Distance is 1 if no point matches and approximates 0 as more points are matched.

    Parameters
    ----------
    keypoint_distance_threshold: float
        Points closer than this threshold are considered a match.
    detection_threshold: float
        Detections and objects with score lower than this threshold are ignored.

    Returns
    -------
    Callable
        The distance funtion that must be passed to the Tracker.
    """

    def keypoints_voting_distance(
        detection: "Detection", tracked_object: "TrackedObject"
    ) -> float:
        distances = np.linalg.norm(detection.points - tracked_object.estimate, axis=1)
        match_num = np.count_nonzero(
            (distances < keypoint_distance_threshold)
            * (detection.scores > detection_threshold)
            * (tracked_object.last_detection.scores > detection_threshold)
        )
        return 1 / (1 + match_num)

    return keypoints_voting_distance


def create_normalized_mean_euclidean_distance(
    height: int, width: int
) -> Callable[["Detection", "TrackedObject"], float]:
    """
    Construct a normalized mean euclidean distance function configured with the max height and width.

    The result distance is bound to [0, 1] where 1 indicates oposite corners of the image.

    Parameters
    ----------
    height: int
        Height of the image.
    width: int
        Width of the image.

    Returns
    -------
    Callable
        The distance funtion that must be passed to the Tracker.
    """

    def normalized__mean_euclidean_distance(
        detection: "Detection", tracked_object: "TrackedObject"
    ) -> float:
        """Normalized mean euclidean distance"""
        # calculate distances and normalized it by width and height
        difference = (detection.points - tracked_object.estimate).astype(float)
        difference[:, 0] /= width
        difference[:, 1] /= height

        # calculate eucledean distance and average
        return np.linalg.norm(difference, axis=1).mean()

    return normalized__mean_euclidean_distance


__all__ = [
    "frobenius",
    "mean_manhattan",
    "mean_euclidean",
    "iou",
    "iou_opt",
    "get_distance_by_name",
    "create_keypoints_voting_distance",
    "create_normalized_mean_euclidean_distance",
]
