import math
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from rich import print

from norfair.camera_motion import TranslationTransformation

from .distances import get_distance_by_name
from .filter import OptimizedKalmanFilterFactory
from .utils import validate_points


class Tracker:
    def __init__(
        self,
        distance_function: Union[str, Callable[["Detection", "TrackedObject"], float]],
        distance_threshold: float,
        hit_counter_max: int = 15,
        initialization_delay: Optional[int] = None,
        pointwise_hit_counter_max: int = 4,
        detection_threshold: float = 0,
        filter_factory: "OptimizedKalmanFilterFactory" = OptimizedKalmanFilterFactory(),
        past_detections_length: int = 4,
        reid_distance_function: Optional[
            Callable[["TrackedObject", "TrackedObject"], float]
        ] = None,
        reid_distance_threshold: float = 0,
        reid_hit_counter_max: Optional[int] = None,
    ):
        self.tracked_objects: Sequence["TrackedObject"] = []

        if isinstance(distance_function, str):
            distance_function = get_distance_by_name(distance_function)
        self.distance_function = distance_function

        self.hit_counter_max = hit_counter_max
        self.reid_hit_counter_max = reid_hit_counter_max
        self.pointwise_hit_counter_max = pointwise_hit_counter_max
        self.filter_factory = filter_factory
        if past_detections_length >= 0:
            self.past_detections_length = past_detections_length
        else:
            raise ValueError(
                f"Argument `past_detections_length` is {past_detections_length} and should be larger than 0."
            )

        if initialization_delay is None:
            self.initialization_delay = int(self.hit_counter_max / 2)
        elif initialization_delay < 0 or initialization_delay > self.hit_counter_max:
            raise ValueError(
                f"Argument 'initialization_delay' for 'Tracker' class should be an int between 0 and (hit_counter_max = {hit_counter_max}). The selected value is {initialization_delay}.\n"
            )
        else:
            self.initialization_delay = initialization_delay

        self.distance_threshold = distance_threshold
        self.detection_threshold = detection_threshold
        TrackedObject.count = 0
        self.reid_distance_function = reid_distance_function
        self.reid_distance_threshold = reid_distance_threshold
        self.abs_to_rel = None

    def update(
        self,
        detections: Optional[List["Detection"]] = None,
        period: int = 1,
        coord_transformations: Optional[TranslationTransformation] = None,
    ):
        if coord_transformations is not None:
            for det in detections:
                det.absolute_points = coord_transformations.rel_to_abs(
                    det.absolute_points
                )
            self.abs_to_rel = coord_transformations.abs_to_rel
        self.period = period

        # Remove stale trackers and make candidate object real if the hit counter is positive
        alive_objects = []
        dead_objects = []
        if self.reid_hit_counter_max is None:
            self.tracked_objects = [
                o for o in self.tracked_objects if o.hit_counter_is_positive
            ]
            alive_objects = self.tracked_objects
        else:
            tracked_objects = []
            for o in self.tracked_objects:
                if o.reid_hit_counter_is_positive:
                    tracked_objects.append(o)
                    if o.hit_counter_is_positive:
                        alive_objects.append(o)
                    else:
                        dead_objects.append(o)
            self.tracked_objects = tracked_objects

        # Update tracker
        for obj in self.tracked_objects:
            obj.tracker_step()
            obj.abs_to_rel = self.abs_to_rel

        # Update initialized tracked objects with detections
        unmatched_detections, _, unmatched_init_trackers = self.update_objects_in_place(
            self.distance_function,
            self.distance_threshold,
            [o for o in alive_objects if not o.is_initializing],
            detections,
        )

        # Update not yet initialized tracked objects with yet unmatched detections
        (
            unmatched_detections,
            matched_not_init_trackers,
            _,
        ) = self.update_objects_in_place(
            self.distance_function,
            self.distance_threshold,
            [o for o in alive_objects if o.is_initializing],
            unmatched_detections,
        )

        if self.reid_distance_function is not None:
            # Match unmatched initialized tracked objects with not yet initialized tracked objects
            _, _, _ = self.update_objects_in_place(
                self.reid_distance_function,
                self.reid_distance_threshold,
                unmatched_init_trackers + dead_objects,
                matched_not_init_trackers,
            )

        # Create new tracked objects from remaining unmatched detections
        for detection in unmatched_detections:
            self.tracked_objects.append(
                TrackedObject(
                    detection,
                    self.hit_counter_max,
                    self.initialization_delay,
                    self.pointwise_hit_counter_max,
                    self.detection_threshold,
                    self.period,
                    self.filter_factory,
                    self.past_detections_length,
                    self.reid_hit_counter_max,
                    self.abs_to_rel,
                )
            )

        return [p for p in self.tracked_objects if not p.is_initializing]

    def _get_distances(
        self,
        distance_function,
        distance_threshold,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ):
        distance_matrix = np.ones((len(candidates), len(objects)), dtype=np.float32)
        distance_matrix *= distance_threshold + 1
        for c, candidate in enumerate(candidates):
            for o, obj in enumerate(objects):
                if candidate.label != obj.label:
                    distance_matrix[c, o] = distance_threshold + 1
                    if (candidate.label is None) or (obj.label is None):
                        print("\nThere are detections with and without label!")
                    continue
                distance = distance_function(candidate, obj)
                # Cap detections and objects with no chance of getting matched so we
                # dont force the hungarian algorithm to minimize them and therefore
                # introduce the possibility of sub optimal results.
                # Note: This is probably not needed with the new distance minimizing algorithm
                if distance > distance_threshold:
                    distance_matrix[c, o] = distance_threshold + 1
                else:
                    distance_matrix[c, o] = distance
        return distance_matrix

    def update_objects_in_place(
        self,
        distance_function,
        distance_threshold,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ):
        if candidates is not None and len(candidates) > 0:
            distance_matrix = self._get_distances(
                distance_function, distance_threshold, objects, candidates
            )
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
                        minimum if minimum < distance_threshold else None
                    )

            matched_cand_indices, matched_obj_indices = self.match_dets_and_objs(
                distance_matrix, distance_threshold
            )
            if len(matched_cand_indices) > 0:
                unmatched_candidates = [
                    d for i, d in enumerate(candidates) if i not in matched_cand_indices
                ]
                unmatched_objects = [
                    d for i, d in enumerate(objects) if i not in matched_obj_indices
                ]
                matched_objects = []

                # Handle matched people/detections
                for (match_cand_idx, match_obj_idx) in zip(
                    matched_cand_indices, matched_obj_indices
                ):
                    match_distance = distance_matrix[match_cand_idx, match_obj_idx]
                    matched_candidate = candidates[match_cand_idx]
                    matched_object = objects[match_obj_idx]
                    if match_distance < distance_threshold:
                        if isinstance(matched_candidate, Detection):
                            matched_object.hit(matched_candidate, period=self.period)
                            matched_object.last_distance = match_distance
                            matched_objects.append(matched_object)
                        elif isinstance(matched_candidate, TrackedObject):
                            # Merge new TrackedObject with the old one
                            matched_object.merge(matched_candidate)
                            # If we are matching TrackedObject instances we want to get rid of the
                            # already matched candidate to avoid matching it again in future frames
                            self.tracked_objects.remove(matched_candidate)
                    else:
                        unmatched_candidates.append(matched_candidate)
                        unmatched_objects.append(matched_object)
            else:
                unmatched_candidates, matched_objects, unmatched_objects = (
                    candidates,
                    [],
                    objects,
                )
        else:
            unmatched_candidates, matched_objects, unmatched_objects = [], [], objects

        return unmatched_candidates, matched_objects, unmatched_objects

    def match_dets_and_objs(self, distance_matrix: np.array, distance_threshold):
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

            while current_min < distance_threshold:
                flattened_arg_min = distance_matrix.argmin()
                det_idx = flattened_arg_min // distance_matrix.shape[1]
                obj_idx = flattened_arg_min % distance_matrix.shape[1]
                det_idxs.append(det_idx)
                obj_idxs.append(obj_idx)
                distance_matrix[det_idx, :] = distance_threshold + 1
                distance_matrix[:, obj_idx] = distance_threshold + 1
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
        hit_counter_max: int,
        initialization_delay: int,
        pointwise_hit_counter_max: int,
        detection_threshold: float,
        period: int,
        filter_factory: "FilterFactory",
        past_detections_length: int,
        reid_hit_counter_max: Optional[int],
        abs_to_rel: Callable[[np.array], np.array],
    ):
        try:
            initial_detection_points = validate_points(
                initial_detection.absolute_points
            )
        except AttributeError:
            print(
                f"\n[red]ERROR[/red]: The detection list fed into `tracker.update()` should be composed of {Detection} objects not {type(initial_detection)}.\n"
            )
            exit()

        self.dim_points = initial_detection_points.shape[1]
        self.num_points = initial_detection_points.shape[0]
        self.hit_counter_max: int = hit_counter_max
        self.pointwise_hit_counter_max: int = pointwise_hit_counter_max
        self.initialization_delay = initialization_delay
        if self.pointwise_hit_counter_max < period:
            self.pointwise_hit_counter_max = period
        self.detection_threshold: float = detection_threshold
        self.initial_period: int = period
        self.hit_counter: int = period
        self.reid_hit_counter_max = reid_hit_counter_max
        self.reid_hit_counter: Optional[int] = None
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
        if initial_detection.scores is None:
            self.detected_at_least_once_points = np.array([True] * self.num_points)
        else:
            self.detected_at_least_once_points = (
                initial_detection.scores > self.detection_threshold
            )
        self.point_hit_counter: np.ndarray = self.detected_at_least_once_points.astype(
            int
        )
        initial_detection.age = self.age
        self.past_detections_length = past_detections_length
        if past_detections_length > 0:
            self.past_detections: Sequence["Detection"] = [initial_detection]
        else:
            self.past_detections: Sequence["Detection"] = []

        # Create Kalman Filter
        self.filter = filter_factory.create_filter(initial_detection_points)
        self.dim_z = self.dim_points * self.num_points
        self.label = initial_detection.label
        self.abs_to_rel = abs_to_rel

    def tracker_step(self):
        self.hit_counter -= 1
        if self.reid_hit_counter is None:
            if self.hit_counter <= 0:
                self.reid_hit_counter = self.reid_hit_counter_max
        else:
            self.reid_hit_counter -= 1
        self.point_hit_counter -= 1
        self.age += 1
        # Advances the tracker's state
        self.filter.predict()

    @property
    def is_initializing(self):
        if self.is_initializing_flag and self.hit_counter > self.initialization_delay:
            self.is_initializing_flag = False
            TrackedObject.count += 1
            self.id = TrackedObject.count
        return self.is_initializing_flag

    @property
    def hit_counter_is_positive(self):
        return self.hit_counter >= 0

    @property
    def reid_hit_counter_is_positive(self):
        return self.reid_hit_counter is None or self.reid_hit_counter >= 0

    @property
    def estimate(self):
        positions = self.filter.x.T.flatten()[: self.dim_z].reshape(-1, self.dim_points)
        velocities = self.filter.x.T.flatten()[self.dim_z :].reshape(
            -1, self.dim_points
        )
        if self.abs_to_rel is not None:
            return self.abs_to_rel(positions)
        return positions

    def get_estimate(self, absolute=False):
        positions = self.filter.x.T.flatten()[: self.dim_z].reshape(-1, 2)
        if self.abs_to_rel is None:
            if not absolute:
                return positions
            else:
                raise ValueError(
                    "You must provide 'coord_transformations' to the tracker to get absolute coordinates"
                )
        else:
            if absolute:
                return positions
            else:
                return self.abs_to_rel(positions)

    @property
    def live_points(self):
        return self.point_hit_counter > 0

    def hit(self, detection: "Detection", period: int = 1):
        points = validate_points(detection.absolute_points)
        self.conditionally_add_to_past_detections(detection)

        self.last_detection = detection
        if self.hit_counter < self.hit_counter_max:
            self.hit_counter += 2 * period

        # We use a kalman filter in which we consider each coordinate on each point as a sensor.
        # This is a hacky way to update only certain sensors (only x, y coordinates for
        # points which were detected).
        # TODO: Use keypoint confidence information to change R on each sensor instead?
        if detection.scores is not None:
            assert len(detection.scores.shape) == 1
            points_over_threshold_mask = detection.scores > self.detection_threshold
            matched_sensors_mask = np.array(
                [(m,) * self.dim_points for m in points_over_threshold_mask]
            ).flatten()
            H_pos = np.diag(matched_sensors_mask).astype(
                float
            )  # We measure x, y positions
            self.point_hit_counter[points_over_threshold_mask] += 2 * period
        else:
            points_over_threshold_mask = np.array([True] * self.num_points)
            H_pos = np.identity(self.num_points * self.dim_points)
            self.point_hit_counter += 2 * period
        self.point_hit_counter[
            self.point_hit_counter >= self.pointwise_hit_counter_max
        ] = self.pointwise_hit_counter_max
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
            [(m,) * self.dim_points for m in self.detected_at_least_once_points]
        ).flatten()
        now_detected_mask = np.hstack(
            (points_over_threshold_mask,) * self.dim_points
        ).flatten()
        first_detection_mask = np.logical_and(
            now_detected_mask, np.logical_not(detected_at_least_once_mask)
        )

        self.filter.x[: self.dim_z][first_detection_mask] = np.expand_dims(
            points.flatten(), 0
        ).T[first_detection_mask]

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
        if self.past_detections_length == 0:
            return
        if len(self.past_detections) < self.past_detections_length:
            detection.age = self.age
            self.past_detections.append(detection)
        elif self.age >= self.past_detections[0].age * self.past_detections_length:
            self.past_detections.pop(0)
            detection.age = self.age
            self.past_detections.append(detection)

    def merge(self, tracked_object):
        """Merge with a not yet initialized TrackedObject instance"""
        self.reid_hit_counter = None
        self.hit_counter = self.initial_period * 2
        self.point_hit_counter = tracked_object.point_hit_counter
        self.last_distance = tracked_object.last_distance
        self.current_min_distance = tracked_object.current_min_distance
        self.last_detection = tracked_object.last_detection
        self.detected_at_least_once_points = (
            tracked_object.detected_at_least_once_points
        )
        self.filter = tracked_object.filter

        for past_detection in tracked_object.past_detections:
            self.conditionally_add_to_past_detections(past_detection)


class Detection:
    def __init__(
        self, points: np.array, scores=None, data=None, label=None, embedding=None
    ):
        self.points = points
        self.scores = scores
        self.data = data
        self.label = label
        self.absolute_points = points.copy()
        self.embedding = embedding
