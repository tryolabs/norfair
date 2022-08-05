import numpy as np
import cv2


def get_motion(gray_next, gray_prvs, max_points=200, min_distance=15, block_size=3, bin_size=0.2):
    # get points 
    prev_pts = cv2.goodFeaturesToTrack(gray_prvs, maxCorners=max_points, qualityLevel=0.01, minDistance=min_distance, blockSize=block_size)
    # compute optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_prvs, gray_next, prev_pts, None)
    # filter valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx].reshape((-1, 2))
    curr_pts = curr_pts[idx].reshape((-1, 2))
    # get flow
    flow = curr_pts - prev_pts

    # get mode
    flow = np.around(flow / bin_size)*bin_size
    unique_flows, counts = np.unique(flow, axis=0, return_counts=True)
    return unique_flows[counts.argmax()]


class MotionEstimator:
    def __init__(self, max_points=200, min_distance=15, block_size=3, bin_size=0.2):
        self.bin_size = bin_size
        self.max_points = max_points
        self.min_distance = min_distance
        self.block_size = block_size
        self.bin_size = bin_size
        
        self.gray_prvs = None
        self.accumulated_flow = np.array([0.0, 0.0])

    def update(self, frame):
        self.gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray_prvs is None:
            self.gray_prvs = self.gray_next
            return np.array([0, 0])
        
        self.accumulated_flow += get_motion(self.gray_next, self.gray_prvs, self.max_points, self.min_distance, self.block_size, self.bin_size)

        self.gray_prvs = self.gray_next

        return self.accumulated_flow