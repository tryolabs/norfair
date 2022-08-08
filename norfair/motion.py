import numpy as np
import cv2


def get_translation_mode_transformations(curr_pts, prev_pts, data, bin_size=0.2):
    # get flow
    flow = curr_pts - prev_pts

    # get mode
    flow = np.around(flow / bin_size) * bin_size
    unique_flows, counts = np.unique(flow, axis=0, return_counts=True)
    flow_mode = unique_flows[counts.argmax()]

    try:
        data += flow_mode
    except TypeError:
        data = flow_mode

    def abs_to_rel(points: np.array):
        return points + data

    def rel_to_abs(points: np.array):
        return points - data

    return data, abs_to_rel, rel_to_abs


def get_homography_transformations(
    curr_pts,
    prev_pts,
    data,
    method=cv2.RANSAC,
    ransacReprojThreshold=3,
    maxIters=2000,
    confidence=0.995,
):
    homography_matrix = cv2.findHomography(
        prev_pts,
        curr_pts,
        method=method,
        ransacReprojThreshold=ransacReprojThreshold,
        maxIters=maxIters,
        confidence=confidence,
    )[0]

    try:
        data[0] = homography_matrix @ data[0]
        data[1] = np.linalg.inv(data[0])
    except TypeError:
        data = (homography_matrix, np.linalg.inv(homography_matrix))

    def abs_to_rel(points: np.array):
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ data[0].T
        points_transformed = points_transformed / points_transformed[:, -1].reshape(
            -1, 1
        )
        return points_transformed[:, :2]

    def rel_to_abs(points: np.array):
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ data[1].T
        points_transformed = points_transformed / points_transformed[:, -1].reshape(
            -1, 1
        )
        return points_transformed[:, :2]

    return data, abs_to_rel, rel_to_abs


def get_sparse_flow(
    gray_next, gray_prvs, max_points=300, min_distance=15, block_size=3
):
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
        transformations_getter=get_translation_mode_transformations,
    ):
        self.max_points = max_points
        self.min_distance = min_distance
        self.block_size = block_size

        self.gray_prvs = None

        self.transformations_getter = transformations_getter

        self.data = (
            None  # data that our transformation_getter needs to store and update
        )

    def update(self, frame, additional_arguments={}):
        self.gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray_prvs is None:
            self.gray_prvs = self.gray_next

        curr_pts, prev_pts = get_sparse_flow(
            self.gray_next,
            self.gray_prvs,
            self.max_points,
            self.min_distance,
            self.block_size,
        )

        self.data, abs_to_rel, rel_to_abs = self.transformations_getter(
            curr_pts, prev_pts, self.data, **additional_arguments
        )

        self.gray_prvs = self.gray_next

        return abs_to_rel, rel_to_abs
