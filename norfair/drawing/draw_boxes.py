from typing import Optional, Sequence, Tuple, Union

import numpy as np

from norfair.tracker import Detection, TrackedObject

from .color import ColorLike, Palette, parse_color
from .drawer import Drawable, Drawer
from .utils import _build_text


def draw_boxes(
    frame: np.ndarray,
    drawables: Sequence[Union[Detection, TrackedObject]],
    color: ColorLike = "by_id",
    thickness: Optional[int] = None,
    draw_labels: bool = False,
    draw_ids: bool = False,
    text_size: Optional[float] = None,
    text_color: Optional[ColorLike] = None,
    text_thickness: Optional[int] = None,
) -> np.ndarray:
    """
    Draw bounding boxes corresponding to Detections or TrackedObjects.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    drawables : Sequence[Union[Detection, TrackedObject]]
        List of objects to draw, Detections and TrackedObjects are accepted.
        This objects are assumed to contain 2 bi-dimensional points defining
        the bounding box as `[[x0, y0], [x1, y1]]`.
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

    Returns
    -------
    np.ndarray
        The resulting frame.
    """
    if color is None:
        color = "by_id"
    if thickness is None:
        thickness = int(max(frame.shape) / 500)

    if drawables is None:
        return frame

    for obj in drawables:
        d = Drawable(obj)

        if color == "by_id":
            obj_color = Palette.choose_color(d.id)
        elif color == "by_label":
            obj_color = Palette.choose_color(d.label)
        else:
            obj_color = parse_color(color)

        points = d.points.astype(int)

        Drawer.rectangle(
            frame,
            tuple(points),
            color=obj_color,
            thickness=thickness,
        )

        text = _build_text(d, draw_labels=draw_labels, draw_ids=draw_ids)
        if text:
            if text_color is None:
                obj_text_color = obj_color
            else:
                obj_text_color = color
            # the anchor will become the bottom-left of the text,
            # we select-top left of the bbox compensating for the thickness of the box
            text_anchor = (
                points[0, 0] - thickness // 2,
                points[0, 1] - thickness // 2 - 1,
            )
            frame = Drawer.text(
                frame,
                text,
                position=text_anchor,
                size=text_size,
                color=obj_text_color,
                thickness=text_thickness,
            )

    return frame
