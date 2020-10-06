import os
import time
import numpy as np
import random
import cv2
from rich import print
from .utils import validate_points
from rich.progress import BarColumn, Progress, TimeRemainingColumn


def draw_boxes(frame, detections, line_color=None, line_width=None):
    frame_scale = frame.shape[0] / 100
    if detections is None:
        return frame
    frame_scale = frame_scale / 100
    if line_width is None:
        line_width = int(max(frame_scale / 7, 1))
    if line_color is None:
        line_color = Color.red
    color_is_rand = line_color == Color.rand
    for d in detections:
        if color_is_rand:
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
    line_color=None,
    line_width=None,
    id_size=None,
    id_thickness=None,
    draw_box=True,
):
    frame_scale = frame.shape[0] / 100
    if line_width is None:
        line_width = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)
    color_is_None = line_color == None
    for obj in objects:
        if not obj.live_points.any():
            continue
        if color_is_None:
            line_color = Color.random(obj.id)
        id_color = line_color

        if draw_box:
            points = obj.estimate
            points = points.astype(int)
            cv2.rectangle(
                frame,
                tuple(points[0, :]),
                tuple(points[1, :]),
                color=line_color,
                thickness=line_width,
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
                id_color,
                id_thickness,
                cv2.LINE_AA,
            )
    return frame


def write_video(output_video, frame_path, detections=None, tracked_objects=None):
    frame = cv2.imread(frame_path)
    frame = draw_boxes(frame, detections, line_color=None, line_width=None)
    frame = draw_tracked_boxes(
        frame,
        tracked_objects,
        line_color=None,
        line_width=None,
        id_size=None,
        id_thickness=None,
        draw_box=True,
    )
    output_video.write(frame)
    cv2.waitKey(1)


def search_value_on_document(file_path, value_name):
    with open(file_path, "r") as myfile:
        seqinfo = myfile.read()
    position = seqinfo.find(value_name)
    position = position + len(value_name)
    while not seqinfo[position].isdigit():
        position += 1
    value_str = ""
    while seqinfo[position].isdigit():
        value_str = value_str + seqinfo[position]
        position += 1
    return int(value_str)


def write_predictions(frame_number, objects=None, out_file=None):
    # write tracked objects information in the output file
    for t in range(len(objects)):
        frame_str = str(int(frame_number))
        id_str = str(int(objects[t].id))
        bb_left_str = str((objects[t].estimate[0, 0]))
        bb_top_str = str((objects[t].estimate[0, 1]))  # [0,1]
        bb_width_str = str((objects[t].estimate[1, 0] - objects[t].estimate[0, 0]))
        bb_height_str = str((objects[t].estimate[1, 1] - objects[t].estimate[0, 1]))
        row_text_out = (
            frame_str
            + ","
            + id_str
            + ","
            + bb_left_str
            + ","
            + bb_top_str
            + ","
            + bb_width_str
            + ","
            + bb_height_str
            + ",-1,-1,-1,-1"
        )
        out_file.write(row_text_out)
        out_file.write("\n")


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
    rand = (-1, -1, -1)  # random color for each detection

    @staticmethod
    def random(obj_id):
        color_list = [
            c
            for c in Color.__dict__.keys()
            if c[:2] != "__"
            and c not in ("random", "red", "white", "grey", "black", "silver", "rand")
        ]
        return getattr(Color, color_list[obj_id % len(color_list)])
