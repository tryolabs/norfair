import numpy as np

from .kalman import KalmanTracker
from sklearn.utils.linear_assignment_ import linear_assignment  # TODO: Remove
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self, distance_function, hit_inertia_min=4, hit_inertia_max=10, match_distance_threshold=0.5):
        self.objects = []
        self.distance_function = distance_function
        # TODO: Make the inertias depend on fps and dt??
        self.hit_inertia_min = hit_inertia_min
        self.hit_inertia_max = hit_inertia_max
        self.match_distance_threshold = match_distance_threshold

    def update(self, detections, dt=1):
        # TODO: Handle dt != 1
        # Remove stale trackers and make candidate object real if it has hit inertia
        self.objects = [o for o in self.objects if o.has_inertia]

        # Update tracker
        for obj in self.objects:
            obj.tracker_step()

        # Update/create trackers
        if len(detections) > 0:
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
                        matched_object.hit(matched_detection)
                        matched_object.last_distance = match_distance
                    else:
                        unmatched_detections.append(matched_detection)

                # Handle remaining unmatched detections
                for d, detection in enumerate(detections):
                    if d not in matched_row_indices:
                        self.objects.append(Object(detection, self.hit_inertia_min, self.hit_inertia_max))
            else:
                # Create new objects from unmatched detections
                for detection in detections:
                    self.objects.append(Object(detection, self.hit_inertia_min, self.hit_inertia_max))


        # Remove stale objects from self.objects list
        self.objects = [p for p in self.objects if p.has_inertia]

        return [p for p in self.objects if not p.is_initializing]


class Object():
    """ TODO: This class and the kalman tracker class should be merged """
    def __init__(self, initial_detection, hit_inertia_min, hit_inertia_max):
        self.hit_inertia_min = hit_inertia_min
        self.hit_inertia_max = hit_inertia_max
        self.hit_counter = hit_inertia_min
        self.last_distance = None
        self.tracker = KalmanTracker(initial_detection)
        self.age = 0
        self.last_detection = initial_detection
        self.is_initializing = True
        self.id = None

    def __repr__(self):
        return "<\033[1mObject {}\033[0m (age {})>".format(self.id, self.age)

    def tracker_step(self):
        self.hit_counter -= 1
        self.age += 1
        if self.is_initializing and self.hit_counter > (self.hit_inertia_min + self.hit_inertia_max) / 2:
            self.is_initializing = False
            KalmanTracker.count += 1
            self.id = KalmanTracker.count
        # Advances the tracker's state
        self.tracker.filter.predict()

    @property
    def has_inertia(self):
        return self.hit_counter >= self.hit_inertia_min

    @property
    def estimate(self):
        return self.tracker.current()

    def hit(self, detection):
        if self.hit_counter < self.hit_inertia_max:
            self.hit_counter += 2

        # We use a kalman filter in which we consider each point as a sensor.
        # This is a hacky way to update only certain sensors (only points which were detected).
        # Hardcoded to our case in which x contains pos and vel for each pos.
        matched_parts_idx = (detection != 0).flatten()
        H_pos = np.diag(matched_parts_idx).astype(float)
        H_vel = np.zeros(H_pos.shape)
        H = np.hstack([H_pos, H_vel])
        self.last_detection = detection
        self.tracker.update(detection, H=H)
