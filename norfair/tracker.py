import numpy as np

from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import random


class Tracker:
    def __init__(self, distance_function, hit_inertia_min=10, hit_inertia_max=25, match_distance_threshold=1,
                 detection_threshold=0):
        self.objects = []
        self.distance_function = distance_function
        self.hit_inertia_min = hit_inertia_min
        self.hit_inertia_max = hit_inertia_max
        self.match_distance_threshold = match_distance_threshold
        self.detection_threshold = detection_threshold

    def update(self, detections=None, period=1):
        # Remove stale trackers and make candidate object real if it has hit inertia
        self.objects = [o for o in self.objects if o.has_inertia]

        # Update tracker
        for obj in self.objects:
            obj.tracker_step()

        # Update/create trackers
        if detections is not None and len(detections) > 0:
            distance_matrix = np.zeros((len(detections), len(self.objects)), dtype=np.float32)
            for d, detection in enumerate(detections):
                for o, obj in enumerate(self.objects):
                    distance_matrix[d, o] = self.distance_function(detection, obj)

            # Filter detections and objects with no chance of getting matched so we
            # dont force the hungarian algorithm to minimize them and therefore
            # introduce the possibility of sub optimal results.
            # The following 10000000's are just arbitrary very large numbers
            distance_matrix[np.all(distance_matrix > self.match_distance_threshold, axis=1)] = 10000000
            distance_matrix[:, np.all(distance_matrix > self.match_distance_threshold, axis=0)] = 10000000

            if np.isnan(distance_matrix).any():
                print("Found nan values in distance matrix, check your distance function for bugs")
            if np.isinf(distance_matrix).any():
                print("Found inf values in distance matrix, check your distance function for bugs")
            matched_row_indices, matched_col_indices = linear_sum_assignment(distance_matrix)
            if len(matched_row_indices) > 0:
                unmatched_detections = [d for i, d in enumerate(detections) if i not in matched_row_indices]

                # Handle matched people/detections
                for match_pair in zip(matched_row_indices, matched_col_indices):
                    match_distance = distance_matrix[match_pair[0], match_pair[1]]
                    matched_detection = detections[match_pair[0]]
                    matched_object = self.objects[match_pair[1]]
                    if match_distance < self.match_distance_threshold:
                        matched_object.hit(matched_detection, period=period)
                        matched_object.last_distance = match_distance
                    else:
                        unmatched_detections.append(matched_detection)

                # Create new objects from unmatched detections
                for d, detection in enumerate(detections):
                    if d not in matched_row_indices:
                        self.objects.append(
                            TrackedObject(
                                detection,
                                self.hit_inertia_min,
                                self.hit_inertia_max,
                                self.detection_threshold,
                                period
                            )
                        )
            else:
                # Create new objects from remaining unmatched detections
                for detection in detections:
                    self.objects.append(
                        TrackedObject(
                            detection,
                            self.hit_inertia_min,
                            self.hit_inertia_max,
                            self.detection_threshold,
                            period
                        )
                    )

        # Remove stale objects from self.objects list
        self.objects = [p for p in self.objects if p.has_inertia]

        return [p for p in self.objects if not p.is_initializing]


class TrackedObject:
    count = 0
    def __init__(self, initial_detection, hit_inertia_min, hit_inertia_max, detection_threshold, period=1):
        self.hit_inertia_min = hit_inertia_min
        self.hit_inertia_max = hit_inertia_max
        self.detection_threshold = detection_threshold
        self.hit_counter = hit_inertia_min + period
        self.last_distance = None
        self.age = 0
        self.is_initializing_flag = True
        self.id = None
        self.initializing_id = random.randint(0, 9999)
        self.setup_kf(initial_detection.points)

    def setup_kf(self, initial_detection):
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
        # TODO: maybe we should open this one to the users, as it lets them
        #       chose between giving more/less importance to the detections
        self.filter.R *= 4.

        # Initial state: numpy.array(dim_x, 1)
        self.filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T

    def tracker_step(self):
        self.hit_counter -= 1
        self.age += 1
        # Advances the tracker's state
        self.filter.predict()

    @property
    def is_initializing(self):
        if self.is_initializing_flag and self.hit_counter > (self.hit_inertia_min + self.hit_inertia_max) / 2:
            self.is_initializing_flag = False
            TrackedObject.count += 1
            self.id = TrackedObject.count
        return self.is_initializing_flag

    @property
    def has_inertia(self):
        return self.hit_counter >= self.hit_inertia_min

    @property
    def estimate(self):
        positions = self.filter.x.T.flatten()[:self.dim_z].reshape(-1, 2)
        velocities = self.filter.x.T.flatten()[self.dim_z:].reshape(-1, 2)
        return positions

    def hit(self, detection, period=1):
        if self.hit_counter < self.hit_inertia_max:
            self.hit_counter += 2 * period

        # We use a kalman filter in which we consider each coordinate on each point as a sensor.
        # This is a hacky way to update only certain sensors (only x, y coordinates for
        # points which were detected).
        # TODO: Use keypoint confidence information to change R on each sensor instead?
        points_over_threshold_idx = detection.scores > self.detection_threshold
        matched_sensors_idx = np.array([[s, s] for s in points_over_threshold_idx]).flatten()
        H_pos = np.diag(matched_sensors_idx).astype(float)  # We measure x, y positions
        H_vel = np.zeros(H_pos.shape)  # But we don't directly measure velocity
        H = np.hstack([H_pos, H_vel])
        self.filter.update(np.expand_dims(detection.points.flatten(), 0).T, None, H)

    def __repr__(self):
        if self.last_distance is None:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {}, init_id: {})"
        else:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {:.2f}, init_id: {})"
        return placeholder_text.format(self.id, self.age, self.hit_counter, self.last_distance, self.initializing_id)


class Detection:
    def __init__(self, points, scores=None, data=None):
        self.points = points
        self.scores = scores
        self.data = data
