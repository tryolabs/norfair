import numpy as np
from filterpy.kalman import KalmanFilter


class FilterSetup:
    def __init__(self, R: float = 4.0, Q: float = 0.1, P: float = 10.0):
        self.R = R
        self.Q = Q
        self.P = P

    def create_filter(self, initial_detection: np.array):
        num_points = initial_detection.shape[0]
        dim_z = 2 * num_points
        dim_x = 2 * 2 * num_points  # We need to accommodate for velocities

        filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        # State transition matrix (models physics): numpy.array()
        filter.F = np.eye(dim_x)
        dt = 1  # At each step we update pos with v * dt

        filter.F[:dim_z, dim_z:] = dt * np.eye(dim_z)

        # Measurement function: numpy.array(dim_z, dim_x)
        filter.H = np.eye(dim_z, dim_x,)

        # Measurement uncertainty (sensor noise): numpy.array(dim_z, dim_z)
        filter.R *= self.R

        # Process uncertainty: numpy.array(dim_x, dim_x)
        # Don't decrease it too much or trackers pay too little attention to detections
        filter.Q[dim_z:, dim_z:] *= self.Q

        # Initial state: numpy.array(dim_x, 1)
        filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T

        # Estimation uncertainty: numpy.array(dim_x, dim_x)
        filter.P[dim_z:, dim_z:] *= self.P

        return filter
