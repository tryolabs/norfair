import cv2
import numpy as np
from .utils import validate_points

def draw_points(frame, detections, radius=None, thickness=None, color=None):
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
            cv2.circle(frame, tuple(point.astype(int)), radius=radius, color=color, thickness=thickness)

def draw_tracked_objects(frame, objects, radius=None, color=None, id_size=None, id_thickness=None, draw_points=True):
    frame_scale = frame.shape[0] / 100
    if radius is None:
        radius = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)

    for obj in objects:
        if color is None:
            point_color = Color.random(obj.id)
            id_color = point_color
        else:
            point_color = color
            id_color = color

        if draw_points:
            for point in obj.estimate:
                cv2.circle(frame, tuple(point.astype(int)), radius=radius, color=point_color, thickness=-1)

        if id_size > 0:
            id_draw_position = centroid(obj.estimate)
            cv2.putText(
                frame, str(obj.id), id_draw_position, cv2.FONT_HERSHEY_SIMPLEX, id_size,
                id_color, id_thickness, cv2.LINE_AA
            )

def draw_debug_metrics(frame, objects, text_size=None, text_thickness=None, color=None,
                       only_ids=None, only_initializing_ids=None):
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
        if only_ids is not None:
            if obj.id not in only_ids: continue
        if only_initializing_ids is not None:
            if obj.initializing_id not in only_initializing_ids: continue
        if color is None:
            text_color = Color.random(obj.initializing_id)
        else:
            text_color = color
        draw_position = centroid(obj.estimate)

        for point in obj.estimate:
            cv2.circle(frame, tuple(point.astype(int)), radius=radius, color=text_color, thickness=-1)

        # Interframe distance
        dist = obj.last_distance
        if dist is None:
            dist = "-"
        elif dist > 1000:
            dist = ">"
        else:
            dist = "{:.1f}".format(dist)

        # No support for multiline text in opencv :facepalm:
        lines_to_draw = (
            "{}|{}".format(obj.id, obj.initializing_id),
            "d:{}".format(dist),
            "a:{}".format(obj.age),
            "h:{}".format(obj.hit_counter),
        )
        for i, line in enumerate(lines_to_draw):
            draw_position = (
                int(draw_position[0]),
                int(draw_position[1] + i * text_size * 7 + 15)
            )
            cv2.putText(
                frame, line, draw_position, cv2.FONT_HERSHEY_SIMPLEX, text_size,
                text_color, text_thickness, cv2.LINE_AA
            )

def centroid(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return int(sum_x/length), int(sum_y/length)

class Color():
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
    def random(obj_id):
        color_list = [
            c for c in Color.__dict__.keys()
            if c[:2] != "__" and c not in ("random", "red", "white", "grey", "black")
        ]
        return getattr(Color, color_list[obj_id % len(color_list)])
