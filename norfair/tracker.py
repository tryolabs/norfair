import numpy as np

from filterpy.kalman import KalmanFilter
from .utils import validate_points
import random


class Tracker:
    def __init__(self, distance_function, hit_inertia_min=10, hit_inertia_max=25, match_distance_threshold=1,
                 detection_threshold=0):
        self.tracked_objects = []
        self.distance_function = distance_function
        self.hit_inertia_min = hit_inertia_min
        self.hit_inertia_max = hit_inertia_max
        self.match_distance_threshold = match_distance_threshold
        self.detection_threshold = detection_threshold
        TrackedObject.count = 0


    def update(self, detections=None, period=1):
        self.period = period

        # Remove stale trackers and make candidate object real if it has hit inertia
        self.tracked_objects = [o for o in self.tracked_objects if o.has_inertia]

        # Update tracker
        for obj in self.tracked_objects:
            obj.tracker_step()

        # Update initialized tracked objects with detections
        unmatched_detections = self.update_objects_in_place(
            [o for o in self.tracked_objects if not o.is_initializing],
            detections
        )

        # Update not yet initialized tracked objects with yet unmatched detections
        unmatched_detections = self.update_objects_in_place(
            [o for o in self.tracked_objects if o.is_initializing],
            unmatched_detections
        )

        # Create new tracked objects from remaining unmatched detections
        for detection in unmatched_detections:
            self.tracked_objects.append(
                TrackedObject(
                    detection,
                    self.hit_inertia_min,
                    self.hit_inertia_max,
                    self.detection_threshold,
                    self.period
                )
            )

        return [p for p in self.tracked_objects if not p.is_initializing]

    def update_objects_in_place(self, objects, detections):
        if detections is not None and len(detections) > 0:
            distance_matrix = np.ones((len(detections), len(objects)), dtype=np.float32)
            distance_matrix *= self.match_distance_threshold + 1
            for d, detection in enumerate(detections):
                for o, obj in enumerate(objects):
                    distance = self.distance_function(detection, obj)
                    # Cap detections and objects with no chance of getting matched so we
                    # dont force the hungarian algorithm to minimize them and therefore
                    # introduce the possibility of sub optimal results.
                    # Note: This is probably not needed with the new distance minimizing algorithm
                    if distance > self.match_distance_threshold:
                        distance_matrix[d, o] = self.match_distance_threshold + 1
                    else:
                        distance_matrix[d, o] = distance

            if np.isnan(distance_matrix).any():
                print("\nReceived nan values from distance function, please check your distance function for errors!")
                exit()
            if np.isinf(distance_matrix).any():
                print("\nReceived inf values from distance function, please check your distance function for errors!")
                print("If you want to explicitly ignore a certain detection - tracked object pair, just")
                print("return match_distance_threshold + 1 from your distance function.")
                exit()

            matched_det_indices, matched_obj_indices = self.match_dets_and_objs(distance_matrix)
            if len(matched_det_indices) > 0:
                unmatched_detections = [d for i, d in enumerate(detections) if i not in matched_det_indices]

                # Handle matched people/detections
                for (match_det_idx, match_obj_idx) in zip(matched_det_indices, matched_obj_indices):
                    match_distance = distance_matrix[match_det_idx, match_obj_idx]
                    matched_detection = detections[match_det_idx]
                    matched_object = objects[match_obj_idx]
                    if match_distance < self.match_distance_threshold:
                        matched_object.hit(matched_detection, period=self.period)
                        matched_object.last_distance = match_distance
                    else:
                        unmatched_detections.append(matched_detection)
            else:
                unmatched_detections = detections
        else:
            unmatched_detections = detections

        return unmatched_detections

    def match_dets_and_objs(self, distance_matrix):
        """Matches detections with tracked_objects from a distance matrix

        I used to match by minimizing the global distances, but found several
        cases in which this was not optimal. So now I just match by starting
        with the global minimum distance and matching the det-obj corresponding
        to that distance, then taking the second minimum, and so on until we
        reach the match_distance_threshold.

        This avoids the the algorithm getting cute with us and matching things
        that shouldn't be matching just for the sake of minimizing the global
        distance, which is what used to happen
        """
        # NOTE: This implementation is terribly inefficient, but it doesn't
        #       seem to affect the fps at all.
        distance_matrix = distance_matrix.copy()
        if distance_matrix.size > 0:
            det_idxs = []
            obj_idxs = []
            current_min = distance_matrix.min()

            while current_min < self.match_distance_threshold:
                flattened_arg_min = distance_matrix.argmin()
                det_idx = flattened_arg_min // distance_matrix.shape[1]
                obj_idx = flattened_arg_min % distance_matrix.shape[1]
                det_idxs.append(det_idx)
                obj_idxs.append(obj_idx)
                distance_matrix[det_idx, :] = self.match_distance_threshold + 1
                distance_matrix[:, obj_idx] = self.match_distance_threshold + 1
                current_min = distance_matrix.min()

            return det_idxs, obj_idxs
        else:
            return [], []


class TrackedObject:
    count = 0
    initializing_count = 0
    def __init__(self, initial_detection, hit_inertia_min, hit_inertia_max, detection_threshold, period=1):
        self.hit_inertia_min = hit_inertia_min
        self.hit_inertia_max = hit_inertia_max
        self.detection_threshold = detection_threshold
        self.hit_counter = hit_inertia_min + period
        self.last_distance = None
        self.last_detection = initial_detection
        self.age = 0
        self.is_initializing_flag = True
        self.id = None
        self.initializing_id = TrackedObject.initializing_count  # Just for debugging
        TrackedObject.initializing_count += 1
        self.setup_kf(initial_detection.points)

    def setup_kf(self, initial_detection):
        initial_detection = validate_points(initial_detection)

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

        # Process uncertainty: numpy.array(dim_x, dim_x)
        # Don't decrease it too much or trackers pay too little attention to detections
        self.filter.Q[dim_z:, dim_z:] /= 10

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
        points = detection.points
        points = validate_points(points)

        self.last_detection = detection
        if self.hit_counter < self.hit_inertia_max:
            self.hit_counter += 2 * period

        # We use a kalman filter in which we consider each coordinate on each point as a sensor.
        # This is a hacky way to update only certain sensors (only x, y coordinates for
        # points which were detected).
        # TODO: Use keypoint confidence information to change R on each sensor instead?
        if detection.scores is not None:
            points_over_threshold_idx = detection.scores > self.detection_threshold
            matched_sensors_idx = np.array([[s, s] for s in points_over_threshold_idx]).flatten()
            H_pos = np.diag(matched_sensors_idx).astype(float)  # We measure x, y positions
        else:
            H_pos = np.identity(points.size)
        H_vel = np.zeros(H_pos.shape)  # But we don't directly measure velocity
        H = np.hstack([H_pos, H_vel])
        self.filter.update(np.expand_dims(points.flatten(), 0).T, None, H)

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
