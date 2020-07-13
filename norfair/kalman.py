import numpy as np
from filterpy.kalman import KalmanFilter
import random

class KalmanTracker(object):
    count = 0
    def __init__(self, initial_detection):
        tracked_points_num = initial_detection.shape[0]
        dim_x = 2 * 2 * tracked_points_num  # We need to accomodate for velocities
        dim_z = 2 * tracked_points_num
        self.dim_z = dim_z
        self.filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
  
        # State transition matrix (models physics): numpy.array()
        self.filter.F = np.eye(dim_x)
        dt = 1  # At each step we update pos with v * dt
        for p in range(dim_z):
            self.filter.F[p, p + dim_z] = dt
  
        # Measurement function: numpy.array(dim_z, dim_x)
        self.filter.H = np.eye(dim_z, dim_x,)
  
        # Measurement uncertainty (sensor noise): numpy.array(dim_z, dim_z)
        self.filter.R *= 1.  # TODO: Open to users so they can chose their own model vs sensor balance?
  
        # Initial state: numpy.array(dim_x, 1)
        self.filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T
  
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.last_detection = initial_detection
  
    def update(self, detection, R=None, H=None):
        # TODO Isn't there a better way to do this than just not updating it?
        # At least tell the filter that a time step happened?
        self.last_detection = detection
        self.filter.update(np.expand_dims(detection.flatten(), 0).T, R, H)
        return self.current()
  
    def current(self):
        positions = self.filter.x.T.flatten()[:self.dim_z].reshape(-1, 2)
        velocities = self.filter.x.T.flatten()[self.dim_z:].reshape(-1, 2)
        return positions
