from abc import ABC, abstractmethod

import numpy as np
from filterpy.kalman import KalmanFilter


class FilterFactory(ABC):
    """Abstract class representing a generic Filter factory

    Subclasses must implement the method `create_filter`
    """

    @abstractmethod
    def create_filter(self, initial_detection: np.ndarray):
        pass


class FilterPyKalmanFilterFactory(FilterFactory):
    """
    This class can be used either to change some parameters of the [KalmanFilter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)
    that the tracker uses, or to fully customize the predictive filter implementation to use (as long as the methods and properties are compatible).

    The former case only requires changing the default parameters upon tracker creation: `tracker = Tracker(..., filter_factory=FilterPyKalmanFilterFactory(R=100))`,
    while the latter requires creating your own class extending `FilterPyKalmanFilterFactory`, and rewriting its `create_filter` method to return your own customized filter.

    Parameters
    ----------
    R : float, optional
        Multiplier for the sensor measurement noise matrix, by default 4.0
    Q : float, optional
        Multiplier for the process uncertainty, by default 0.1
    P : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to position (not speed) variables, by default 10.0

    See Also
    --------
    [`filterpy.KalmanFilter`](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html).
    """

    def __init__(self, R: float = 4.0, Q: float = 0.1, P: float = 10.0):
        self.R = R
        self.Q = Q
        self.P = P

    def create_filter(self, initial_detection: np.ndarray) -> KalmanFilter:
        """
        This method returns a new predictive filter instance with the current setup, to be used by each new [`TrackedObject`][norfair.tracker.TrackedObject] that is created.
        This predictive filter will be used to estimate speed and future positions of the object, to better match the detections during its trajectory.

        Parameters
        ----------
        initial_detection : np.ndarray
            numpy array of shape `(number of points per object, 2)`, corresponding to the [`Detection.points`][norfair.tracker.Detection] of the tracked object being born,
            which shall be used as initial position estimation for it.

        Returns
        -------
        KalmanFilter
            The kalman filter
        """
        num_points = initial_detection.shape[0]
        dim_points = initial_detection.shape[1]
        dim_z = dim_points * num_points
        dim_x = 2 * dim_z  # We need to accommodate for velocities

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
        filter.x[dim_z:] = 0

        # Estimation uncertainty: numpy.array(dim_x, dim_x)
        filter.P[dim_z:, dim_z:] *= self.P

        return filter


class NoFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))

    def predict(self):
        return

    def update(self, detection_points_flatten, R=None, H=None):

        if H is not None:
            diagonal = np.diagonal(H).reshape((self.dim_z, 1))
            one_minus_diagonal = 1 - diagonal

            detection_points_flatten = np.multiply(
                diagonal, detection_points_flatten
            ) + np.multiply(one_minus_diagonal, self.x[: self.dim_z])

        self.x[: self.dim_z] = detection_points_flatten


class NoFilterFactory(FilterFactory):
    """
    This class allows the user to try Norfair without any predictive filter or velocity estimation.

    This track only by comparing the position of the previous detections to the ones in the current frame.

    The throughput of this class in FPS is similar to the one achieved by the
    [`OptimizedKalmanFilterFactory`](#optimizedkalmanfilterfactory) class, so this class exists only for
    comparative purposes and it is not advised to use it for tracking on a real application.

    Parameters
    ----------
    FilterFactory : _type_
        _description_
    """

    def create_filter(self, initial_detection: np.ndarray):
        num_points = initial_detection.shape[0]
        dim_points = initial_detection.shape[1]
        dim_z = dim_points * num_points  # flattened positions
        dim_x = 2 * dim_z  # We need to accommodate for velocities

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

    def predict(self):
        self.x[: self.dim_z] += self.x[self.dim_z :]

    def update(self, detection_points_flatten, R=None, H=None):

        if H is not None:
            diagonal = np.diagonal(H).reshape((self.dim_z, 1))
            one_minus_diagonal = 1 - diagonal
        else:
            diagonal = np.ones((self.dim_z, 1))
            one_minus_diagonal = np.zeros((self.dim_z, 1))

        if R is not None:
            kalman_r = np.diagonal(R).reshape((self.dim_z, 1))
        else:
            kalman_r = self.default_r

        error = np.multiply(detection_points_flatten - self.x[: self.dim_z], diagonal)

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


class OptimizedKalmanFilterFactory(FilterFactory):
    """
    Creates faster Filters than [`FilterPyKalmanFilterFactory`][norfair.filter.FilterPyKalmanFilterFactory].

    It allows the user to create Kalman Filter optimized for tracking and set its parameters.

    Parameters
    ----------
    R : float, optional
        Multiplier for the sensor measurement noise matrix.
    Q : float, optional
        Multiplier for the process uncertainty.
    pos_variance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to position (not speed) variables.
    pos_vel_covariance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to the covariance between position and speed.
    vel_variance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to velocity (not position) variables.
    """

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

    def create_filter(self, initial_detection: np.ndarray):
        num_points = initial_detection.shape[0]
        dim_points = initial_detection.shape[1]
        dim_z = dim_points * num_points  # flattened positions
        dim_x = 2 * dim_z  # We need to accommodate for velocities

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
