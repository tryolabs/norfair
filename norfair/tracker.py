import math
from typing import Callable, List, Optional, Sequence

import numpy as np
from rich import print

from .utils import validate_points
from .filter import FilterSetup


class Tracker:
    def __init__(
        self,
        distance_function: Callable[["Detection", "TrackedObject"], float],
        distance_threshold: float,
        hit_inertia_min: int = 10,
        hit_inertia_max: int = 25,
        initialization_delay: Optional[int] = None,
        detection_threshold: float = 0,
        point_transience: int = 4,
        filter_setup: "FilterSetup" = FilterSetup(),
        past_detections_length: int = 4
    ):
        self.tracked_objects: Sequence["TrackedObject"] = []
        self.distance_function = distance_function
        self.hit_inertia_min = hit_inertia_min
        self.hit_inertia_max = hit_inertia_max
        self.filter_setup = filter_setup
        if past_detections_length >= 0:
            self.past_detections_length = past_detections_length
        else:
            raise ValueError(f"Argument `past_detections_length` is {past_detections_length} and should be larger than 0.")

        if initialization_delay is None:
            self.initialization_delay = int(
                (self.hit_inertia_max - self.hit_inertia_min) / 2
            )
        elif (
            initialization_delay < 0
            or initialization_delay > self.hit_inertia_max - self.hit_inertia_min
        ):
            raise ValueError(
                f"Argument 'initialization_delay' for 'Tracker' class should be an int between 0 and (hit_inertia_max - hit_inertia_min = {hit_inertia_max - hit_inertia_min}). The selected value is {initialization_delay}.\n"
            )
        else:
            self.initialization_delay = initialization_delay

        self.distance_threshold = distance_threshold
        self.detection_threshold = detection_threshold
        self.point_transience = point_transience
        TrackedObject.count = 0

    def update(self, detections: Optional[List["Detection"]] = None, period: int = 1):
        self.period = period

        # Remove stale trackers and make candidate object real if it has hit inertia
        self.tracked_objects = [o for o in self.tracked_objects if o.has_inertia]

        # Update tracker
        for obj in self.tracked_objects:
            obj.tracker_step()

        # Update initialized tracked objects with detections
        unmatched_detections = self.update_objects_in_place(
            [o for o in self.tracked_objects if not o.is_initializing], detections
        )

        # Update not yet initialized tracked objects with yet unmatched detections
        unmatched_detections = self.update_objects_in_place(
            [o for o in self.tracked_objects if o.is_initializing], unmatched_detections
        )

        # Create new tracked objects from remaining unmatched detections
        for detection in unmatched_detections:
            self.tracked_objects.append(
                TrackedObject(
                    detection,
                    self.hit_inertia_min,
                    self.hit_inertia_max,
                    self.initialization_delay,
                    self.detection_threshold,
                    self.period,
                    self.point_transience,
                    self.filter_setup,
                    self.past_detections_length
                )
            )

        return [p for p in self.tracked_objects if not p.is_initializing]

    def update_objects_in_place(
        self,
        objects: Sequence["TrackedObject"],
        detections: Optional[List["Detection"]],
    ):
        if detections is not None and len(detections) > 0:
            distance_matrix = np.ones((len(detections), len(objects)), dtype=np.float32)
            distance_matrix *= self.distance_threshold + 1
            for d, detection in enumerate(detections):
                for o, obj in enumerate(objects):
                    distance = self.distance_function(detection, obj)
                    # Cap detections and objects with no chance of getting matched so we
                    # dont force the hungarian algorithm to minimize them and therefore
                    # introduce the possibility of sub optimal results.
                    # Note: This is probably not needed with the new distance minimizing algorithm
                    if distance > self.distance_threshold:
                        distance_matrix[d, o] = self.distance_threshold + 1
                    else:
                        distance_matrix[d, o] = distance

            if np.isnan(distance_matrix).any():
                print(
                    "\nReceived nan values from distance function, please check your distance function for errors!"
                )
                exit()
            if np.isinf(distance_matrix).any():
                print(
                    "\nReceived inf values from distance function, please check your distance function for errors!"
                )
                print(
                    "If you want to explicitly ignore a certain detection - tracked object pair, just"
                )
                print("return distance_threshold + 1 from your distance function.")
                exit()

            # Used just for debugging distance function
            if distance_matrix.any():
                for i, minimum in enumerate(distance_matrix.min(axis=0)):
                    objects[i].current_min_distance = (
                        minimum if minimum < self.distance_threshold else None
                    )

            matched_det_indices, matched_obj_indices = self.match_dets_and_objs(
                distance_matrix
            )
            if len(matched_det_indices) > 0:
                unmatched_detections = [
                    d for i, d in enumerate(detections) if i not in matched_det_indices
                ]

                # Handle matched people/detections
                for (match_det_idx, match_obj_idx) in zip(
                    matched_det_indices, matched_obj_indices
                ):
                    match_distance = distance_matrix[match_det_idx, match_obj_idx]
                    matched_detection = detections[match_det_idx]
                    matched_object = objects[match_obj_idx]
                    if match_distance < self.distance_threshold:
                        matched_object.hit(matched_detection, period=self.period)
                        matched_object.last_distance = match_distance
                    else:
                        unmatched_detections.append(matched_detection)
            else:
                unmatched_detections = detections
        else:
            unmatched_detections = []

        return unmatched_detections

    def match_dets_and_objs(self, distance_matrix: np.array):
        """Matches detections with tracked_objects from a distance matrix

        I used to match by minimizing the global distances, but found several
        cases in which this was not optimal. So now I just match by starting
        with the global minimum distance and matching the det-obj corresponding
        to that distance, then taking the second minimum, and so on until we
        reach the distance_threshold.

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

            while current_min < self.distance_threshold:
                flattened_arg_min = distance_matrix.argmin()
                det_idx = flattened_arg_min // distance_matrix.shape[1]
                obj_idx = flattened_arg_min % distance_matrix.shape[1]
                det_idxs.append(det_idx)
                obj_idxs.append(obj_idx)
                distance_matrix[det_idx, :] = self.distance_threshold + 1
                distance_matrix[:, obj_idx] = self.distance_threshold + 1
                current_min = distance_matrix.min()

            return det_idxs, obj_idxs
        else:
            return [], []


class TrackedObject:
    count = 0
    initializing_count = 0

    def __init__(
        self,
        initial_detection: "Detection",
        hit_inertia_min: int,
        hit_inertia_max: int,
        initialization_delay: int,
        detection_threshold: float,
        period: int,
        point_transience: int,
        filter_setup: "FilterSetup",
        past_detections_length: int
    ):
        try:
            initial_detection_points = validate_points(initial_detection.points)
        except AttributeError:
            print(
                f"\n[red]ERROR[/red]: The detection list fed into `tracker.update()` should be composed of {Detection} objects not {type(initial_detection)}.\n"
            )
            exit()
        self.num_points = initial_detection_points.shape[0]
        self.hit_inertia_min: int = hit_inertia_min
        self.hit_inertia_max: int = hit_inertia_max
        self.initialization_delay = initialization_delay
        self.point_hit_inertia_min: int = math.floor(hit_inertia_min / point_transience)
        self.point_hit_inertia_max: int = math.ceil(hit_inertia_max / point_transience)
        if (self.point_hit_inertia_max - self.point_hit_inertia_min) < period:
            self.point_hit_inertia_max = self.point_hit_inertia_min + period
        self.detection_threshold: float = detection_threshold
        self.initial_period: int = period
        self.hit_counter: int = hit_inertia_min + period
        self.point_hit_counter: np.ndarray = (
            np.ones(self.num_points) * self.point_hit_inertia_min
        )
        self.last_distance: Optional[float] = None
        self.current_min_distance: Optional[float] = None
        self.last_detection: "Detection" = initial_detection
        self.age: int = 0
        self.is_initializing_flag: bool = True
        self.id: Optional[int] = None
        self.initializing_id: int = (
            TrackedObject.initializing_count
        )  # Just for debugging
        TrackedObject.initializing_count += 1
        self.detected_at_least_once_points = np.array([False] * self.num_points)
        initial_detection.age = self.age
        self.past_detections_length = past_detections_length
        if past_detections_length > 0:
            self.past_detections: Sequence["Detection"] = [initial_detection]
        else:
            self.past_detections: Sequence["Detection"] = []

        # Create Kalman Filter
        self.filter = filter_setup.create_filter(initial_detection_points)
        self.dim_z = 2 * self.num_points

    def tracker_step(self):
        self.hit_counter -= 1
        self.point_hit_counter -= 1
        self.age += 1
        # Advances the tracker's state
        self.filter.predict()

    @property
    def is_initializing(self):
        if (
            self.is_initializing_flag
            and self.hit_counter > self.hit_inertia_min + self.initialization_delay
        ):
            self.is_initializing_flag = False
            TrackedObject.count += 1
            self.id = TrackedObject.count
        return self.is_initializing_flag

    @property
    def has_inertia(self):
        return self.hit_counter >= self.hit_inertia_min

    @property
    def estimate(self):
        positions = self.filter.x.T.flatten()[: self.dim_z].reshape(-1, 2)
        velocities = self.filter.x.T.flatten()[self.dim_z :].reshape(-1, 2)
        return positions

    @property
    def live_points(self):
        return self.point_hit_counter > self.point_hit_inertia_min

    def hit(self, detection: "Detection", period: int = 1):
        points = validate_points(detection.points)
        self.conditionally_add_to_past_detections(detection)

        self.last_detection = detection
        if self.hit_counter < self.hit_inertia_max:
            self.hit_counter += 2 * period

        # We use a kalman filter in which we consider each coordinate on each point as a sensor.
        # This is a hacky way to update only certain sensors (only x, y coordinates for
        # points which were detected).
        # TODO: Use keypoint confidence information to change R on each sensor instead?
        if detection.scores is not None:
            assert len(detection.scores.shape) == 1
            points_over_threshold_mask = detection.scores > self.detection_threshold
            matched_sensors_mask = np.array(
                [[m, m] for m in points_over_threshold_mask]
            ).flatten()
            H_pos = np.diag(matched_sensors_mask).astype(
                float
            )  # We measure x, y positions
            self.point_hit_counter[points_over_threshold_mask] += 2 * period
        else:
            points_over_threshold_mask = np.array([True] * self.num_points)
            H_pos = np.identity(points.size)
            self.point_hit_counter += 2 * period
        self.point_hit_counter[
            self.point_hit_counter >= self.point_hit_inertia_max
        ] = self.point_hit_inertia_max
        self.point_hit_counter[self.point_hit_counter < 0] = 0
        H_vel = np.zeros(H_pos.shape)  # But we don't directly measure velocity
        H = np.hstack([H_pos, H_vel])
        self.filter.update(np.expand_dims(points.flatten(), 0).T, None, H)

        # Force points being detected for the first time to have velocity = 0
        # This is needed because some detectors (like OpenPose) set points with
        # low confidence to coordinates (0, 0). And when they then get their first
        # real detection this creates a huge velocity vector in our KalmanFilter
        # and causes the tracker to start with wildly inaccurate estimations which
        # eventually coverge to the real detections.
        detected_at_least_once_mask = np.array(
            [[m, m] for m in self.detected_at_least_once_points]
        ).flatten()
        self.filter.x[self.dim_z :][np.logical_not(detected_at_least_once_mask)] = 0
        self.detected_at_least_once_points = np.logical_or(
            self.detected_at_least_once_points, points_over_threshold_mask
        )

    def __repr__(self):
        if self.last_distance is None:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {}, init_id: {})"
        else:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {:.2f}, init_id: {})"
        return placeholder_text.format(
            self.id,
            self.age,
            self.hit_counter,
            self.last_distance,
            self.initializing_id,
        )

    def conditionally_add_to_past_detections(self, detection):
        """Adds detections into (and pops detections away) from `past_detections`

        It does so by keeping a fixed amount of past detections saved into each 
        TrackedObject, while maintaining them distributed uniformly through the object's
        lifetime.
        """
        if self.past_detections_length == 0: return
        if len(self.past_detections) < self.past_detections_length:
            detection.age = self.age
            self.past_detections.append(detection)
        elif self.age >= self.past_detections[0].age * self.past_detections_length:
            self.past_detections.pop(0)
            detection.age = self.age
            self.past_detections.append(detection)


class Detection:
    def __init__(self, points: np.array, scores=None, data=None):
        self.points = points
        self.scores = scores
        self.data = data
