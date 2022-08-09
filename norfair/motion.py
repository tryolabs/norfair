import numpy as np
import cv2


def get_translation_mode_transformations(
    curr_pts, prev_pts, data, bin_size=0.2, proportion_points_used_threshold=0.9
):
    # get flow
    flow = curr_pts - prev_pts

    # get mode
    flow = np.around(flow / bin_size) * bin_size
    unique_flows, counts = np.unique(flow, axis=0, return_counts=True)

    max_index = counts.argmax()

    proportion_points_used = counts[max_index] / len(prev_pts)
    update_prvs = proportion_points_used < proportion_points_used_threshold

    flow_mode = unique_flows[max_index]

    try:
        flow_mode += data
    except TypeError:
        pass

    if update_prvs:
        data = flow_mode

    def abs_to_rel(points: np.array):
        return points + flow_mode

    def rel_to_abs(points: np.array):
        return points - flow_mode

    return update_prvs, data, abs_to_rel, rel_to_abs


def get_homography_transformations(
    curr_pts,
    prev_pts,
    data,
    method=cv2.RANSAC,
    ransacReprojThreshold=3,
    maxIters=2000,
    confidence=0.995,
    proportion_points_used_threshold=0.9,
):
    homography_matrix, points_used = cv2.findHomography(
        prev_pts,
        curr_pts,
        method=method,
        ransacReprojThreshold=ransacReprojThreshold,
        maxIters=maxIters,
        confidence=confidence,
    )

    proportion_points_used = np.sum(points_used) / len(points_used)

    update_prvs = proportion_points_used < proportion_points_used_threshold

    try:
        homography_matrix = homography_matrix @ data
    except (TypeError, ValueError):
        pass

    inverse_homography_matrix = np.linalg.inv(homography_matrix)

    if update_prvs:
        data = homography_matrix

    def abs_to_rel(points: np.array):
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ homography_matrix.T
        points_transformed = points_transformed / points_transformed[:, -1].reshape(
            -1, 1
        )
        return points_transformed[:, :2]

    def rel_to_abs(points: np.array):
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ inverse_homography_matrix.T
        points_transformed = points_transformed / points_transformed[:, -1].reshape(
            -1, 1
        )
        return points_transformed[:, :2]

    return update_prvs, data, abs_to_rel, rel_to_abs


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
        transformations_getter=get_homography_transformations,
    ):
        self.max_points = max_points
        self.min_distance = min_distance
        self.block_size = block_size

        self.gray_prvs = None

        self.transformations_getter = transformations_getter

        self.data = (
            None  # data that our transformation_getter needs to store and update
        )
        self.prev_pts = None

    def update(self, frame, additional_arguments={}):
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

        update_prvs, self.data, abs_to_rel, rel_to_abs = self.transformations_getter(
            curr_pts, self.prev_pts, self.data, **additional_arguments
        )

        if update_prvs:
            self.gray_prvs = self.gray_next
            self.prev_pts = None

        return abs_to_rel, rel_to_abs
