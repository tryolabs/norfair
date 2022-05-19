import numpy as np
from filterpy.kalman import KalmanFilter


class FilterPyKalmanFilterFactory:
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
        filter.H = np.eye(
            dim_z,
            dim_x,
        )

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


class NoFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))

        self.kp_undetected = np.ones((dim_z, 1))

    def predict(self):
        return

    def update(self, detection_points_flatten, R=None, H=None):

        if H is not None:
            diagonal = np.diagonal(H).reshape((self.dim_z, 1))
            one_minus_diagonal = 1 - diagonal
            self.kp_undetected = np.logical_and(one_minus_diagonal, self.kp_undetected)

            detection_points_flatten = np.multiply(
                diagonal, detection_points_flatten
            ) + np.multiply(one_minus_diagonal, self.x[: self.dim_z])

        self.x[: self.dim_z] = detection_points_flatten


class NoFilterFactory:
    def create_filter(self, initial_detection: np.array):
        num_points = initial_detection.shape[0]
        dim_z = 2 * num_points  # flattened positions
        dim_x = 2 * 2 * num_points  # We need to accommodate for velocities

        no_filter = NoFilter(
            dim_x,
            dim_z,
        )
        no_filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T
        return no_filter


class OptimizedKalmanFilter:
    def __init__(
        self,
        dim_x,
        dim_z,
        pos_variance=10,
        pos_vel_covariance=0,
        vel_variance=1,
        q=0.1,
        r=4,
    ):
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))

        # matrix P from Kalman
        self.pos_variance = np.zeros((dim_z, 1)) + pos_variance
        self.pos_vel_covariance = np.zeros((dim_z, 1)) + pos_vel_covariance
        self.vel_variance = np.zeros((dim_z, 1)) + vel_variance

        self.q_Q = q

        self.default_r = r * np.ones((dim_z, 1))

        self.kp_undetected = np.ones((dim_z, 1))

    def predict(self):
        self.x[: self.dim_z] += self.x[self.dim_z :]

    def update(self, detection_points_flatten, R=None, H=None):

        if H is not None:
            diagonal = np.diagonal(H).reshape((self.dim_z, 1))

            kp_just_detected = np.logical_and(diagonal, self.kp_undetected)

            one_minus_diagonal = 1 - diagonal
            self.kp_undetected = np.logical_and(one_minus_diagonal, self.kp_undetected)

            detection_points_flatten = np.multiply(
                diagonal, detection_points_flatten
            ) + np.multiply(one_minus_diagonal, self.x[: self.dim_z])

            kp_just_detected = np.argwhere(kp_just_detected.flatten())
            self.x[: self.dim_z][kp_just_detected] = detection_points_flatten[
                kp_just_detected
            ]
        else:
            diagonal = np.ones((self.dim_z, 1))
            one_minus_diagonal = np.zeros((self.dim_z, 1))

        if R is not None:
            kalman_r = np.diagonal(R).reshape((self.dim_z, 1))
        else:
            kalman_r = self.default_r

        error = detection_points_flatten - self.x[: self.dim_z]

        vel_var_plus_pos_vel_cov = self.pos_vel_covariance + self.vel_variance
        added_variances = (
            self.pos_variance
            + self.pos_vel_covariance
            + vel_var_plus_pos_vel_cov
            + self.q_Q
            + kalman_r
        )

        kalman_r_over_added_variances = np.divide(kalman_r, added_variances)
        vel_var_plus_pos_vel_cov_over_added_variances = np.divide(
            vel_var_plus_pos_vel_cov, added_variances
        )

        added_variances_or_kalman_r = np.multiply(
            added_variances, one_minus_diagonal
        ) + np.multiply(kalman_r, diagonal)

        self.x[: self.dim_z] += np.multiply(
            diagonal, np.multiply(1 - kalman_r_over_added_variances, error)
        )
        self.x[self.dim_z :] += np.multiply(
            diagonal, np.multiply(vel_var_plus_pos_vel_cov_over_added_variances, error)
        )

        self.pos_variance = np.multiply(
            1 - kalman_r_over_added_variances, added_variances_or_kalman_r
        )
        self.pos_vel_covariance = np.multiply(
            vel_var_plus_pos_vel_cov_over_added_variances, added_variances_or_kalman_r
        )
        self.vel_variance += self.q_Q - np.multiply(
            diagonal,
            np.multiply(
                np.square(vel_var_plus_pos_vel_cov_over_added_variances),
                added_variances,
            ),
        )


class OptimizedKalmanFilterFactory:
    def __init__(
        self,
        R: float = 4.0,
        Q: float = 0.1,
        pos_variance: float = 10,
        pos_vel_covariance: float = 0,
        vel_variance: float = 1,
    ):
        self.R = R
        self.Q = Q

        # entrances P matrix of KF
        self.pos_variance = pos_variance
        self.pos_vel_covariance = pos_vel_covariance
        self.vel_variance = vel_variance

    def create_filter(self, initial_detection: np.array):
        num_points = initial_detection.shape[0]
        dim_z = 2 * num_points  # flattened positions
        dim_x = 2 * 2 * num_points  # We need to accommodate for velocities

        custom_filter = OptimizedKalmanFilter(
            dim_x,
            dim_z,
            pos_variance=self.pos_variance,
            pos_vel_covariance=self.pos_vel_covariance,
            vel_variance=self.vel_variance,
            q=self.Q,
            r=self.R,
        )
        custom_filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T

        return custom_filter
