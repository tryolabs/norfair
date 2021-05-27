from typing import Optional, Sequence, Tuple

try:
    import cv2
except ImportError:
    from .utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()
import random

import numpy as np

from .utils import validate_points


def draw_points(
    frame: np.array,
    detections: Sequence["Detection"],
    radius: Optional[int] = None,
    thickness: Optional[int] = None,
    color: Optional[Tuple[int, int, int]] = None,
):
    if detections is None:
        return
    frame_scale = frame.shape[0] / 100
    if radius is None:
        radius = int(max(frame_scale * 0.7, 1))
    if thickness is None:
        thickness = int(max(frame_scale / 7, 1))
    if color is None:
        color = Color.red
    for d in detections:
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


def draw_tracked_objects(
    frame: np.array,
    objects: Sequence["TrackedObject"],
    radius: Optional[int] = None,
    color: Optional[Tuple[int, int, int]] = None,
    id_size: Optional[float] = None,
    id_thickness: Optional[int] = None,
    draw_points: bool = True,
):
    frame_scale = frame.shape[0] / 100
    if radius is None:
        radius = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)

    for obj in objects:
        if not obj.live_points.any():
            continue
        if color is None:
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
        if color is None:
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
        lines_to_draw = (
            "{}|{}".format(obj.id, obj.initializing_id),
            "a:{}".format(obj.age),
            "h:{}".format(obj.hit_counter),
            "ld:{}".format(last_dist),
            "cd:{}".format(current_min_dist),
        )
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


def draw_boxes(frame, detections, line_color=None, line_width=None, random_color=False):
    frame_scale = frame.shape[0] / 100
    if detections is None:
        return frame
    frame_scale = frame_scale / 100
    if line_width is None:
        line_width = int(max(frame_scale / 7, 1))
    if line_color is None:
        line_color = Color.red
    for d in detections:
        if random_color:
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
    return frame


def draw_tracked_boxes(
    frame,
    objects,
    border_colors=None,
    border_width=None,
    id_size=None,
    id_thickness=None,
    draw_box=True,
):
    frame_scale = frame.shape[0] / 100
    if border_width is None:
        border_width = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)
    if isinstance(border_colors, tuple):
        border_colors = [border_colors]

    for n, obj in enumerate(objects):
        if not obj.live_points.any():
            continue
        if border_colors is None:
            color = Color.random(obj.id)
        else:
            color = border_colors[n % len(border_colors)]

        if draw_box:
            points = obj.estimate
            points = points.astype(int)
            cv2.rectangle(
                frame,
                tuple(points[0, :]),
                tuple(points[1, :]),
                color=color,
                thickness=border_width,
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
