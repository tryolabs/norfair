from typing import Optional, Sequence, Tuple

from .camera_motion import TranslationTransformation

try:
    import cv2
except ImportError:
    from .utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()
import random

import numpy as np

from .utils import validate_points, warn_once


def draw_points(
    frame: np.array,
    detections: Sequence["Detection"],
    radius: Optional[int] = None,
    thickness: Optional[int] = None,
    color: Optional[Tuple[int, int, int]] = None,
    color_by_label: bool = False,
    draw_labels: bool = False,
    label_size: Optional[int] = None,
):
    if detections is None:
        return
    frame_scale = frame.shape[0] / 100
    if radius is None:
        radius = int(max(frame_scale * 0.7, 1))
    if thickness is None:
        thickness = int(max(frame_scale / 7, 1))
    if label_size is None:
        label_size = int(max(frame_scale / 100, 1))
    if color is None:
        color = Color.red
    for d in detections:
        if color_by_label:
            color = Color.random(abs(hash(d.label)))
        points = d.points
        points = validate_points(points)
        for point in points:
            cv2.circle(
                frame,
                tuple(point.astype(int)),
                radius=radius,
                color=color,
                thickness=thickness,
            )

        if draw_labels:
            label_draw_position = np.array([min(points[:, 0]), min(points[:, 1])])
            label_draw_position -= radius
            cv2.putText(
                frame,
                f"L: {d.label}",
                tuple(label_draw_position.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                color,
                thickness,
                cv2.LINE_AA,
            )


def draw_tracked_objects(
    frame: np.array,
    objects: Sequence["TrackedObject"],
    radius: Optional[int] = None,
    color: Optional[Tuple[int, int, int]] = None,
    id_size: Optional[float] = None,
    id_thickness: Optional[int] = None,
    draw_points: bool = True,
    color_by_label: bool = False,
    draw_labels: bool = False,
    label_size: Optional[int] = None,
):
    frame_scale = frame.shape[0] / 100
    if radius is None:
        radius = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)
    if label_size is None:
        label_size = int(max(frame_scale / 100, 1))

    for obj in objects:
        if not obj.live_points.any():
            continue
        if color_by_label:
            point_color = Color.random(abs(hash(obj.label)))
            id_color = point_color
        elif color is None:
            object_id = obj.id if obj.id is not None else random.randint(0, 999)
            point_color = Color.random(object_id)
            id_color = point_color
        else:
            point_color = color
            id_color = color

        if draw_points:
            for point, live in zip(obj.estimate, obj.live_points):
                if live:
                    cv2.circle(
                        frame,
                        tuple(point.astype(int)),
                        radius=radius,
                        color=point_color,
                        thickness=-1,
                    )

            if draw_labels:
                points = obj.estimate[obj.live_points]
                points = points.astype(int)
                label_draw_position = np.array([min(points[:, 0]), min(points[:, 1])])
                label_draw_position -= radius
                cv2.putText(
                    frame,
                    f"L: {obj.label}",
                    tuple(label_draw_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_size,
                    point_color,
                    id_thickness,
                    cv2.LINE_AA,
                )

        if id_size > 0:
            id_draw_position = centroid(obj.estimate[obj.live_points])
            cv2.putText(
                frame,
                str(obj.id),
                id_draw_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                id_size,
                id_color,
                id_thickness,
                cv2.LINE_AA,
            )


def draw_debug_metrics(
    frame: np.array,
    objects: Sequence["TrackedObject"],
    text_size: Optional[float] = None,
    text_thickness: Optional[int] = None,
    color: Optional[Tuple[int, int, int]] = None,
    only_ids=None,
    only_initializing_ids=None,
    draw_score_threshold: float = 0,
    color_by_label: bool = False,
    draw_labels: bool = False,
):
    """Draw objects with their debug information

    It is recommended to set the input variable `objects` to `your_tracker_object.objects`
    so you can also debug objects wich haven't finished initializing, and you get a more
    complete view of what your tracker is doing on each step.
    """
    frame_scale = frame.shape[0] / 100
    if text_size is None:
        text_size = frame_scale / 10
    if text_thickness is None:
        text_thickness = int(frame_scale / 5)
    radius = int(frame_scale * 0.5)

    for obj in objects:
        if (
            not (obj.last_detection.scores is None)
            and not (obj.last_detection.scores > draw_score_threshold).any()
        ):
            continue
        if only_ids is not None:
            if obj.id not in only_ids:
                continue
        if only_initializing_ids is not None:
            if obj.initializing_id not in only_initializing_ids:
                continue
        if color_by_label:
            text_color = Color.random(abs(hash(obj.label)))
        elif color is None:
            text_color = Color.random(obj.initializing_id)
        else:
            text_color = color
        draw_position = centroid(
            obj.estimate[obj.last_detection.scores > draw_score_threshold]
            if obj.last_detection.scores is not None
            else obj.estimate
        )

        for point in obj.estimate:
            cv2.circle(
                frame,
                tuple(point.astype(int)),
                radius=radius,
                color=text_color,
                thickness=-1,
            )

        # Distance to last matched detection
        if obj.last_distance is None:
            last_dist = "-"
        elif obj.last_distance > 999:
            last_dist = ">"
        else:
            last_dist = "{:.2f}".format(obj.last_distance)

        # Distance to currently closest detection
        if obj.current_min_distance is None:
            current_min_dist = "-"
        else:
            current_min_dist = "{:.2f}".format(obj.current_min_distance)

        # No support for multiline text in opencv :facepalm:
        lines_to_draw = [
            "{}|{}".format(obj.id, obj.initializing_id),
            "a:{}".format(obj.age),
            "h:{}".format(obj.hit_counter),
            "ld:{}".format(last_dist),
            "cd:{}".format(current_min_dist),
        ]
        if draw_labels:
            lines_to_draw.append("l:{}".format(obj.label))

        for i, line in enumerate(lines_to_draw):
            draw_position = (
                int(draw_position[0]),
                int(draw_position[1] + i * text_size * 7 + 15),
            )
            cv2.putText(
                frame,
                line,
                draw_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )


def centroid(tracked_points: np.array) -> Tuple[int, int]:
    num_points = tracked_points.shape[0]
    sum_x = np.sum(tracked_points[:, 0])
    sum_y = np.sum(tracked_points[:, 1])
    return int(sum_x / num_points), int(sum_y / num_points)


def draw_boxes(
    frame,
    detections,
    line_color=None,
    line_width=None,
    random_color=False,
    color_by_label=False,
    draw_labels=False,
    label_size=None,
):
    frame_scale = frame.shape[0] / 100
    if detections is None:
        return frame
    if line_width is None:
        line_width = int(max(frame_scale / 7, 1))
    if line_color is None:
        line_color = Color.red
    if label_size is None:
        label_size = int(max(frame_scale / 100, 1))
    for d in detections:
        if color_by_label:
            line_color = Color.random(abs(hash(d.label)))
        elif random_color:
            line_color = Color.random(random.randint(0, 20))
        points = d.points
        points = validate_points(points)
        points = points.astype(int)
        cv2.rectangle(
            frame,
            tuple(points[0, :]),
            tuple(points[1, :]),
            color=line_color,
            thickness=line_width,
        )

        if draw_labels:
            label_draw_position = np.array(points[0, :])
            cv2.putText(
                frame,
                f"L: {d.label}",
                tuple(label_draw_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                line_color,
                line_width,
                cv2.LINE_AA,
            )

    return frame


def draw_tracked_boxes(
    frame,
    objects,
    border_colors=None,
    border_width=None,
    id_size=None,
    id_thickness=None,
    draw_box=True,
    color_by_label=False,
    draw_labels=False,
    label_size=None,
    label_width=None,
):
    frame_scale = frame.shape[0] / 100
    if border_width is None:
        border_width = int(frame_scale * 0.5)
    if label_width is None:
        label_width = int(max(frame_scale / 7, 2))
    if label_size is None:
        label_size = int(max(frame_scale / 100, 1))
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)
    if isinstance(border_colors, tuple):
        border_colors = [border_colors]

    for n, obj in enumerate(objects):
        if not obj.live_points.any():
            continue
        if color_by_label:
            color = Color.random(abs(hash(obj.label)))
        elif border_colors is None:
            color = Color.random(obj.id)
        else:
            color = border_colors[n % len(border_colors)]

        points = obj.estimate
        if draw_box:
            points = points.astype(int)
            cv2.rectangle(
                frame,
                tuple(points[0, :]),
                tuple(points[1, :]),
                color=color,
                thickness=border_width,
            )

            if draw_labels:
                label_draw_position = np.array(points[0, :])
                cv2.putText(
                    frame,
                    f"L: {obj.label}",
                    tuple(label_draw_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_size,
                    color,
                    label_width,
                    cv2.LINE_AA,
                )

        if id_size > 0:
            id_draw_position = np.mean(points, axis=0)
            id_draw_position = id_draw_position.astype(int)
            cv2.putText(
                frame,
                str(obj.id),
                tuple(id_draw_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                id_size,
                color,
                id_thickness,
                cv2.LINE_AA,
            )
    return frame


class Paths:
    def __init__(self, get_points_to_draw=None, thickness=None, color=None, radius=None, attenuation=0.01):
        if get_points_to_draw is None:
            def get_points_to_draw(points):
                return [np.mean(np.array(points), axis=0)]
        
        self.get_points_to_draw = get_points_to_draw

        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.mask = None
        self.attenuation_factor = 1 - attenuation

    def draw(self, frame, tracked_objects):
        if self.mask is None:
            frame_scale = frame.shape[0] / 100

            if self.radius is None:
                self.radius = int(max(frame_scale * 0.7, 1))
            if self.thickness is None:
                self.thickness = int(max(frame_scale / 7, 1))

            self.mask = np.zeros(frame.shape, np.uint8)
        
        self.mask = (self.mask*self.attenuation_factor).astype('uint8') 

        for obj in tracked_objects:
            if obj.abs_to_rel is not None:
                warn_once("It seems that your using the Path drawer together with MotionEstimator. This is not fully supported and the results will not be what's expected")

            if self.color is None:
                color = Color.random(obj.id)
            else:
                color = self.color

            points_to_draw = self.get_points_to_draw(obj.estimate)

            for point in points_to_draw:
                cv2.circle(
                    self.mask,
                    tuple(point.astype(int)),
                    radius=self.radius,
                    color=color,
                    thickness=self.thickness,
                )

        return cv2.addWeighted(self.mask, 1, frame, 1, 0, frame)


def _draw_cross(frame, center, radius, color, thickness):
    middle_x, middle_y = center
    left, top = center - radius
    right, bottom = center + radius
    cv2.line(
        frame, (middle_x, top), (middle_x, bottom), color=color, thickness=thickness
    )
    cv2.line(
        frame, (left, middle_y), (right, middle_y), color=color, thickness=thickness
    )


class Color:
    green = (0, 128, 0)
    white = (255, 255, 255)
    olive = (0, 128, 128)
    black = (0, 0, 0)
    navy = (128, 0, 0)
    red = (0, 0, 255)
    maroon = (0, 0, 128)
    grey = (128, 128, 128)
    purple = (128, 0, 128)
    yellow = (0, 255, 255)
    lime = (0, 255, 0)
    fuchsia = (255, 0, 255)
    aqua = (255, 255, 0)
    blue = (255, 0, 0)
    teal = (128, 128, 0)
    silver = (192, 192, 192)

    @staticmethod
    def random(obj_id: int) -> Tuple[int, int, int]:
        color_list = [
            c
            for c in Color.__dict__.keys()
            if c[:2] != "__"
            and c not in ("random", "red", "white", "grey", "black", "silver")
        ]
        return getattr(Color, color_list[obj_id % len(color_list)])


from functools import lru_cache


@lru_cache(maxsize=4)
def _get_grid(size, w, h, polar=False):
    """
    Construct the grid of points.

    Points are choosen
    Results are cached since the grid in absolute coordinates doesn't change.
    """
    # We need to get points on a semi-sphere of radious 1 centered around (0, 0)

    # First step is to get a grid of angles, theta and phi ∈ (-pi/2, pi/2)
    step = np.pi / size
    start = -np.pi / 2 + step
    end = np.pi / 2 - step
    theta, fi = np.mgrid[start:end:step, start:end:step]

    if polar:
        # if polar=True the first frame will show points as if
        # you are on the center of the earth looking at one of the poles.
        # Points on the sphere are defined as [sin(theta) * cos(fi), sin(theta) * sin(fi), cos(theta)]
        # Then we need to intersect the line defined by the point above with the
        # plane z=1 which is the "absolute plane", we do so by dividing by cos(theta), the result becomes
        # [tan(theta) * cos(fi), tan(theta) * sin(phi), 1]
        # note that the z=1 is implied by the coord_transformation so there is no need to add it.
        tan_theta = np.tan(theta)

        X = tan_theta * np.cos(fi)
        Y = tan_theta * np.sin(fi)
    else:
        # otherwhise will show as if you were looking at the ecuator
        X = np.tan(fi)
        Y = np.divide(np.tan(theta), np.cos(fi))
    # construct the points as x, y coordinates
    points = np.vstack((X.flatten(), Y.flatten())).T
    # scale and center the points
    return  points * max(h, w) + np.array([w // 2, h // 2])


def draw_absolute_grid(frame, coord_transformations, grid_size=20, radius=2, thickness=1, color=Color.black, polar=False):
    h, w, _ = frame.shape

    # get absolute points grid
    points = _get_grid(grid_size, w, h, polar=polar)

    # transform the points to relative coordinates
    if coord_transformations is None:
        points_transformed = points
    else:
        points_transformed = coord_transformations.abs_to_rel(points)

    # filter points that are not visible
    visible_points = points_transformed[
        (points_transformed <= np.array([w, h])).all(axis=1)
        & (points_transformed >= 0).all(axis=1)
    ]
    for point in visible_points:
        _draw_cross(
            frame, point.astype(int), radius=radius, thickness=thickness, color=color
        )


class FixedCamera:
    def __init__(self, scale: float=2, attenuation: float=0.05):
        self.scale = scale
        self._background = None
        self._attenuation_factor = 1 - attenuation

    def adjust_frame(self, frame, coord_transformation: TranslationTransformation):
        # initialize background if necessary
        if self._background is None:
            original_size = (frame.shape[1], frame.shape[0])  # OpenCV format is (width, height)

            scaled_size = tuple(
                (np.array(original_size) * np.array(self.scale))
                .round()
                .astype(int)
            )
            self._background = np.zeros(
                [scaled_size[1], scaled_size[0], frame.shape[-1]],
                frame.dtype,
            )
        else:
            self._background = (self._background * self._attenuation_factor).astype(frame.dtype)

        # top_left is the anchor coordinate from where we start drawing the fame on top of the background
        # aim to draw it in the center of the background but transformations will move this point
        top_left = (
            np.array(self._background.shape[:2]) // 2
            - np.array(frame.shape[:2]) // 2
        )
        top_left = (
            coord_transformation.rel_to_abs(top_left[::-1]).round().astype(int)[::-1]
        )
        # box of the background that will be updated and the limits of it
        background_y0, background_y1 = (top_left[0], top_left[0] + frame.shape[0])
        background_x0, background_x1 = (top_left[1], top_left[1] + frame.shape[1])
        background_size_y, background_size_x = self._background.shape[:2]

        # define box of the frame that will be used
        # if the scale is not enough to support the movement, warn the user but keep drawing
        # cropping the frame so that the operation doesn't fail
        frame_y0, frame_y1, frame_x0, frame_x1 = (0, frame.shape[0], 0, frame.shape[1])
        if (
            background_y0 < 0
            or background_x0 < 0
            or background_y1 > background_size_y
            or background_x1 > background_size_x
        ):
            warn_once(
                "moving_camera_scale is not enough to cover the range of camera movement, frame will be cropped"
            )
            # crop left or top of the frame if necessary
            frame_y0 = max(-background_y0, 0)
            frame_x0 = max(-background_x0, 0)
            # crop right or bottom of the frame if necessary
            frame_y1 = max(min(background_size_y - background_y0, background_y1 - background_y0), 0)
            frame_x1 = max(min(background_size_x - background_x0, background_x1 - background_x0), 0)
            # handle cases where the limits of the background become negative which numpy will interpret incorrectly
            background_y0 = max(background_y0, 0)
            background_x0 = max(background_x0, 0)
            background_y1 = max(background_y1, 0)
            background_x1 = max(background_x1, 0)
        self._background[background_y0:background_y1, background_x0:background_x1, :] = frame[frame_y0:frame_y1, frame_x0:frame_x1, :]
        return self._background
