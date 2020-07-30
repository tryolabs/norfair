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
        color_list = [c for c in Color.__dict__.keys() if c[:2] != "__" and c != "random"]
        return getattr(Color, color_list[obj_id % len(color_list)])
