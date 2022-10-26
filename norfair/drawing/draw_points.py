from typing import Optional, Sequence, Union

import numpy as np

from norfair.tracker import Detection, TrackedObject

from .color import ColorLike, Palette, parse_color
from .drawer import Drawable, Drawer
from .utils import _build_text


def draw_points(
    frame: np.ndarray,
    drawables: Sequence[Union[Detection, TrackedObject]],
    color: ColorLike = "by_id",
    radius: Optional[int] = None,
    thickness: Optional[int] = None,
    draw_labels: bool = True,
    draw_ids: bool = True,
    draw_points: bool = True,
    text_size: Optional[int] = None,
    text_color: Optional[ColorLike] = None,
    hide_dead_points: bool = True,
):
    """
    Draw the points included in a list of Detections or TrackedObjects.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    drawables : Sequence[Union[Detection, TrackedObject]]
        List of objects to draw, Detections and TrackedObjects are accepted.
    color : ColorLike, optional
        This parameter can take:
        1. A color as a tuple of ints describing the BGR `(0, 0, 255)`
        2. A 6-digit hex string `"#FF0000"`
        3. One of the defined color names `"red"`
        4. A string defining the strategy to choose colors from the Palette:
            1. based on the id of the objects `"by_id"`
            2. based on the label of the objects `"by_label"`
        Note that if your objects don't have labels or ids (Detections never have ids)
        the selected color will be the same for all objects.
    radius : Optional[int], optional
        Radius of the circles representing each point.
        By default a sensible value is picked considering the frame size.
    thickness : Optional[int], optional
        Thickness or width of the line.
    draw_labels : bool, optional
        If set to True, the label is added to a title that is drawn on top of the box.
        If an object doesn't have a label this parameter is ignored.
    draw_ids : bool, optional
        If set to True, the id is added to a title that is drawn on top of the box.
        If an object doesn't have an id this parameter is ignored.
    text_size : Optional[float], optional
        Size of the title, the value is used as a multiplier of the base size of the font.
        By default the size is scaled automatically based on the frame size.
    text_color : Optional[int], optional
        Color of the text. By default the same color as the box is used.
    text_thickness : Optional[int], optional
        Thickness of the font. By default it's scaled with the `text_size`.
    hide_dead_points : bool, optional
        By default the dead points of the TrackedObject are hidden.
        A point is considered dead if the corresponding value of `TrackedObject.live_points` is set to False.
        If all objects are dead the object is not drawn.
        All points of a detection are considered to be alive.
        Set this param to False to always draw all points.

    Returns
    -------
    np.ndarray
        The resulting frame.
    """
    if drawables is None:
        return
    if color is None:
        color = "by_id"
    if thickness is None:
        thickness = -1
    if radius is None:
        radius = int(round(max(max(frame.shape) * 0.002, 1)))

    for o in drawables:
        d = Drawable(o)

        if hide_dead_points and not d.live_points.any():
            continue

        if color == "by_id":
            obj_color = Palette.choose_color(d.id)
        elif color == "by_label":
            obj_color = Palette.choose_color(d.label)
        else:
            obj_color = parse_color(color)

        if text_color is None:
            obj_text_color = obj_color
        else:
            obj_text_color = color

        if draw_points:
            for point, live in zip(d.points, d.live_points):
                if live or not hide_dead_points:
                    Drawer.circle(
                        frame,
                        tuple(point.astype(int)),
                        radius=radius,
                        color=obj_color,
                        thickness=thickness,
                    )

        if draw_labels or draw_ids:
            position = d.points[d.live_points].mean(axis=0)
            position -= radius
            text = _build_text(d, draw_labels=draw_labels, draw_ids=draw_ids)

            Drawer.text(
                frame,
                text,
                tuple(position.astype(int)),
                size=text_size,
                color=obj_text_color,
            )
    return frame


# TODO: We used to have this function to debug
# migrate it to use Drawer and clean it up
# if possible maybe merge this functionality to the function above

# def draw_debug_metrics(
#     frame: np.ndarray,
#     objects: Sequence["TrackedObject"],
#     text_size: Optional[float] = None,
#     text_thickness: Optional[int] = None,
#     color: Optional[Tuple[int, int, int]] = None,
#     only_ids=None,
#     only_initializing_ids=None,
#     draw_score_threshold: float = 0,
#     color_by_label: bool = False,
#     draw_labels: bool = False,
# ):
#     """Draw objects with their debug information

#     It is recommended to set the input variable `objects` to `your_tracker_object.objects`
#     so you can also debug objects wich haven't finished initializing, and you get a more
#     complete view of what your tracker is doing on each step.
#     """
#     frame_scale = frame.shape[0] / 100
#     if text_size is None:
#         text_size = frame_scale / 10
#     if text_thickness is None:
#         text_thickness = int(frame_scale / 5)
#     radius = int(frame_scale * 0.5)

#     for obj in objects:
#         if (
#             not (obj.last_detection.scores is None)
#             and not (obj.last_detection.scores > draw_score_threshold).any()
#         ):
#             continue
#         if only_ids is not None:
#             if obj.id not in only_ids:
#                 continue
#         if only_initializing_ids is not None:
#             if obj.initializing_id not in only_initializing_ids:
#                 continue
#         if color_by_label:
#             text_color = Color.random(abs(hash(obj.label)))
#         elif color is None:
#             text_color = Color.random(obj.initializing_id)
#         else:
#             text_color = color
#         draw_position = _centroid(
#             obj.estimate[obj.last_detection.scores > draw_score_threshold]
#             if obj.last_detection.scores is not None
#             else obj.estimate
#         )

#         for point in obj.estimate:
#             cv2.circle(
#                 frame,
#                 tuple(point.astype(int)),
#                 radius=radius,
#                 color=text_color,
#                 thickness=-1,
#             )

#         # Distance to last matched detection
#         if obj.last_distance is None:
#             last_dist = "-"
#         elif obj.last_distance > 999:
#             last_dist = ">"
#         else:
#             last_dist = "{:.2f}".format(obj.last_distance)

#         # Distance to currently closest detection
#         if obj.current_min_distance is None:
#             current_min_dist = "-"
#         else:
#             current_min_dist = "{:.2f}".format(obj.current_min_distance)

#         # No support for multiline text in opencv :facepalm:
#         lines_to_draw = [
#             "{}|{}".format(obj.id, obj.initializing_id),
#             "a:{}".format(obj.age),
#             "h:{}".format(obj.hit_counter),
#             "ld:{}".format(last_dist),
#             "cd:{}".format(current_min_dist),
#         ]
#         if draw_labels:
#             lines_to_draw.append("l:{}".format(obj.label))

#         for i, line in enumerate(lines_to_draw):
#             draw_position = (
#                 int(draw_position[0]),
#                 int(draw_position[1] + i * text_size * 7 + 15),
#             )
#             cv2.putText(
#                 frame,
#                 line,
#                 draw_position,
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 text_size,
#                 text_color,
#                 text_thickness,
#                 cv2.LINE_AA,
#             )
