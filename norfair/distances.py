"""Predefined distances"""
from abc import ABC, abstractmethod
from logging import warning
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from .tracker import Detection, TrackedObject


class Distance(ABC):
    """
    Abstract class representing a distance.

    Subclasses must implement the method `get_distances`
    """

    @abstractmethod
    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        pass


class ScalarDistance(Distance):
    """
    ScalarDistance class represents a distance that is calculated pointwise.

    Parameters
    ----------
    distance_function : Union[Callable[["Detection", "TrackedObject"], float], Callable[["TrackedObject", "TrackedObject"], float]]
        Distance function used to determine the pointwise distance between new candidates and objects.
        This function should take 2 input arguments, the first being a `Union[Detection, TrackedObject]`,
        and the second [TrackedObject][norfair.tracker.TrackedObject]. It has to return a `float` with the distance it calculates.
    """

    def __init__(
        self,
        distance_function: Union[
            Callable[["Detection", "TrackedObject"], float],
            Callable[["TrackedObject", "TrackedObject"], float],
        ],
    ):
        self.distance_function = distance_function

    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        distance_matrix = np.full(
            (len(candidates), len(objects)),
            fill_value=np.inf,
            dtype=np.float32,
        )
        if not objects or not candidates:
            return distance_matrix
        for c, candidate in enumerate(candidates):
            for o, obj in enumerate(objects):
                if candidate.label != obj.label:
                    if (candidate.label is None) or (obj.label is None):
                        print("\nThere are detections with and without label!")
                    continue
                distance = self.distance_function(candidate, obj)
                distance_matrix[c, o] = distance
        return distance_matrix


class VectorizedDistance(Distance):
    """
    VectorizedDistance class represents a distance that is calculated in a vectorized way. This means
    that instead of going through every pair and explicitly calculating its distance, VectorizedDistance
    uses the entire vectors to compare to each other in a single operation.

    Parameters
    ----------
    distance_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Distance function used to determine the distances between new candidates and objects.
        This function should take 2 input arguments, the first being a `np.ndarray` and the second
        `np.ndarray`. It has to return a `np.ndarray` with the distance matrix it calculates.
    """

    def __init__(
        self,
        distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.distance_function = distance_function

    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        distance_matrix = np.full(
            (len(candidates), len(objects)),
            fill_value=np.inf,
            dtype=np.float32,
        )
        if not objects or not candidates:
            return distance_matrix

        object_labels = np.array([o.label for o in objects]).astype(str)
        candidate_labels = np.array([c.label for c in candidates]).astype(str)

        # iterate over labels that are present both in objects and detections
        for label in np.intersect1d(
            np.unique(object_labels), np.unique(candidate_labels)
        ):
            # generate masks of the subset of object and detections for this label
            obj_mask = object_labels == label
            cand_mask = candidate_labels == label

            stacked_objects = []
            for o in objects:
                if str(o.label) == label:
                    stacked_objects.append(o.estimate.ravel())
            stacked_objects = np.stack(stacked_objects)

            stacked_candidates = []
            for c in candidates:
                if str(c.label) == label:
                    if "Detection" in str(type(c)):
                        stacked_candidates.append(c.points.ravel())
                    else:
                        stacked_candidates.append(c.estimate.ravel())
            stacked_candidates = np.stack(stacked_candidates)

            # calculate the pairwise distances between objects and candidates with this label
            # and assign the result to the correct positions inside distance_matrix
            distance_matrix[np.ix_(cand_mask, obj_mask)] = self._compute_distance(
                stacked_candidates, stacked_objects
            )

        return distance_matrix

    def _compute_distance(
        self, stacked_candidates: np.ndarray, stacked_objects: np.ndarray
    ) -> np.ndarray:
        """
        Method that computes the pairwise distances between new candidates and objects.
        It is intended to use the entire vectors to compare to each other in a single operation.

        Parameters
        ----------
        stacked_candidates : np.ndarray
            np.ndarray containing a stack of candidates to be compared with the stacked_objects.
        stacked_objects : np.ndarray
            np.ndarray containing a stack of objects to be compared with the stacked_objects.

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        return self.distance_function(stacked_candidates, stacked_objects)


class ScipyDistance(VectorizedDistance):
    """
    ScipyDistance class extends VectorizedDistance for the use of Scipy's vectorized distances.

    This class uses `scipy.spatial.distance.cdist` to calculate distances between two `np.ndarray`.

    Parameters
    ----------
    metric : str, optional
        Defines the specific Scipy metric to use to calculate the pairwise distances between
        new candidates and objects.

    See Also
    --------
    [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    """

    def __init__(
        self,
        metric: str = "euclidean",
    ):
        self.metric = metric

    def _compute_distance(
        self, stacked_candidates: np.ndarray, stacked_objects: np.ndarray
    ) -> np.ndarray:
        """
        Method that computes the pairwise distances between new candidates and objects.
        It is intended to use the entire vectors to compare to each other in a single operation.

        Parameters
        ----------
        stacked_candidates : np.ndarray
            np.ndarray containing a stack of candidates to be compared with the stacked_objects.
        stacked_objects : np.ndarray
            np.ndarray containing a stack of objects to be compared with the stacked_objects.

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        return cdist(stacked_candidates, stacked_objects, metric=self.metric)


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


_SCALAR_DISTANCE_FUNCTIONS = {
    "frobenius": frobenius,
    "mean_manhattan": mean_manhattan,
    "mean_euclidean": mean_euclidean,
    "iou": iou,
    "iou_opt": iou_opt,
}
_VECTORIZED_DISTANCE_FUNCTIONS = {}
_SCIPY_DISTANCE_FUNCTIONS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
AVAILABLE_VECTORIZED_DISTANCES = (
    list(_VECTORIZED_DISTANCE_FUNCTIONS.keys()) + _SCIPY_DISTANCE_FUNCTIONS
)


def get_distance_by_name(name: str) -> Distance:
    """
    Select a distance by name.

    Parameters
    ----------
    name : str
        A string defining the metric to get.

    Returns
    -------
    Distance
        The distance object.
    """

    if name in _SCALAR_DISTANCE_FUNCTIONS:
        warning(
            "You are using a scalar distance function. If you want to speed up the"
            " tracking process please consider using a vectorized distance function"
            f" such as {AVAILABLE_VECTORIZED_DISTANCES}."
        )
        distance = _SCALAR_DISTANCE_FUNCTIONS[name]
        distance_function = ScalarDistance(distance)
    elif name in _SCIPY_DISTANCE_FUNCTIONS:
        distance_function = ScipyDistance(name)
    elif name in _VECTORIZED_DISTANCE_FUNCTIONS:
        distance = _VECTORIZED_DISTANCE_FUNCTIONS[name]
        distance_function = VectorizedDistance(distance)
    else:
        raise ValueError(
            f"Invalid distance '{name}', expecting one of"
            f" {list(_SCALAR_DISTANCE_FUNCTIONS.keys()) + AVAILABLE_VECTORIZED_DISTANCES}"
        )

    return distance_function


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
