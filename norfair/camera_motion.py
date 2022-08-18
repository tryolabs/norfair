from abc import ABC, abstractmethod

import cv2
import numpy as np


#
# Abstract interfaces
#
class CoordinatesTransformation(ABC):
    @abstractmethod
    def abs_to_rel(self, points: np.array):
        pass

    @abstractmethod
    def rel_to_abs(self, points: np.array):
        pass


class TransformationGetter(ABC):
    @abstractmethod
    def __call__(self, curr_pts, prev_pts):
        pass

#
# Translation
#
class TranslationTransformation(CoordinatesTransformation):
    def __init__(self, movement_vector):
        self.movement_vector = movement_vector

    def abs_to_rel(self, points: np.array):
        return points + self.movement_vector

    def rel_to_abs(self, points: np.array):
        return points - self.movement_vector

class TranslationTransformationGetter(TransformationGetter):
    def __init__(self, bin_size=0.2, proportion_points_used_threshold=0.9) -> None:
        self.data = None
        self.bin_size = bin_size
        self.proportion_points_used_threshold = proportion_points_used_threshold

    def __call__(self, curr_pts, prev_pts):
        # get flow
        flow = curr_pts - prev_pts

        # get mode
        flow = np.around(flow / self.bin_size) * self.bin_size
        unique_flows, counts = np.unique(flow, axis=0, return_counts=True)

        max_index = counts.argmax()

        proportion_points_used = counts[max_index] / len(prev_pts)
        update_prvs = proportion_points_used < self.proportion_points_used_threshold

        flow_mode = unique_flows[max_index]

        try:
            flow_mode += self.data
        except TypeError:
            pass

        if update_prvs:
            self.data = flow_mode

        return update_prvs, TranslationTransformation(flow_mode)

#
# Homography
#
class HomographyTransformation(CoordinatesTransformation):
    def __init__(self, homography_matrix):
        self.homography_matrix = homography_matrix
        self.inverse_homography_matrix = np.linalg.inv(homography_matrix)


    def abs_to_rel(self, points: np.array):
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ self.homography_matrix.T
        points_transformed = points_transformed / points_transformed[:, -1].reshape(
            -1, 1
        )
        return points_transformed[:, :2]

    def rel_to_abs(self, points: np.array):
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ self.inverse_homography_matrix.T
        points_transformed = points_transformed / points_transformed[:, -1].reshape(
            -1, 1
        )
        return points_transformed[:, :2]



class HomographyTransformationGetter(TransformationGetter):
    def __init__(
        self,
        method=cv2.RANSAC,
        ransacReprojThreshold=3,
        maxIters=2000,
        confidence=0.995,
        proportion_points_used_threshold=0.9,
    ) -> None:
        self.data = None
        self.method = method
        self.ransacReprojThreshold = ransacReprojThreshold
        self.maxIters = maxIters
        self.confidence = confidence
        self.proportion_points_used_threshold = proportion_points_used_threshold

    def __call__(self, curr_pts, prev_pts):

        homography_matrix, points_used = cv2.findHomography(
            prev_pts,
            curr_pts,
            method=self.method,
            ransacReprojThreshold=self.ransacReprojThreshold,
            maxIters=self.maxIters,
            confidence=self.confidence,
        )

        proportion_points_used = np.sum(points_used) / len(points_used)

        update_prvs = proportion_points_used < self.proportion_points_used_threshold

        try:
            homography_matrix = homography_matrix @ self.data
        except (TypeError, ValueError):
            pass

        if update_prvs:
            self.data = homography_matrix

        return update_prvs, HomographyTransformation(homography_matrix)

#
# Motion estimation
#
def get_sparse_flow(
    gray_next, gray_prvs, prev_pts=None, max_points=300, min_distance=15, block_size=3
):
    if prev_pts is None:
        # get points
        prev_pts = cv2.goodFeaturesToTrack(
            gray_prvs,
            maxCorners=max_points,
            qualityLevel=0.01,
            minDistance=min_distance,
            blockSize=block_size,
        )

    # compute optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        gray_prvs, gray_next, prev_pts, None
    )
    # filter valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx].reshape((-1, 2))
    curr_pts = curr_pts[idx].reshape((-1, 2))
    return curr_pts, prev_pts


class MotionEstimator:
    def __init__(
        self,
        max_points=200,
        min_distance=15,
        block_size=3,
        transformations_getter=None,
    ):
        self.max_points = max_points
        self.min_distance = min_distance
        self.block_size = block_size

        self.gray_prvs = None
        self.prev_pts = None
        if transformations_getter is None:
            transformations_getter = HomographyTransformationGetter()

        self.transformations_getter = transformations_getter

    def update(self, frame):
        self.gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray_prvs is None:
            self.gray_prvs = self.gray_next

        curr_pts, self.prev_pts = get_sparse_flow(
            self.gray_next,
            self.gray_prvs,
            self.prev_pts,
            self.max_points,
            self.min_distance,
            self.block_size,
        )

        update_prvs, coord_transformations = self.transformations_getter(
            curr_pts,
            self.prev_pts,
        )

        if update_prvs:
            self.gray_prvs = self.gray_next
            self.prev_pts = None

        return coord_transformations
