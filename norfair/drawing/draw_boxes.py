from typing import Optional, Sequence, Tuple, Union

import numpy as np

from norfair.tracker import Detection, TrackedObject
from norfair.utils import warn_once

from .color import ColorLike, Palette, parse_color
from .drawer import Drawable, Drawer
from .utils import _build_text


def draw_boxes(
    frame: np.ndarray,
    drawables: Union[Sequence[Detection], Sequence[TrackedObject]] = None,
    color: ColorLike = "by_id",
    thickness: Optional[int] = None,
    random_color: bool = None,  # Deprecated
    color_by_label: bool = None,  # Deprecated
    draw_labels: bool = False,
    text_size: Optional[float] = None,
    draw_ids: bool = True,
    text_color: Optional[ColorLike] = None,
    text_thickness: Optional[int] = None,
    draw_box: bool = True,
    detections: Sequence["Detection"] = None,  # Deprecated
    line_color: Optional[ColorLike] = None,  # Deprecated
    line_width: Optional[int] = None,  # Deprecated
    label_size: Optional[int] = None,  # Deprecated´
    draw_scores: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes corresponding to Detections or TrackedObjects.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    drawables : Union[Sequence[Detection], Sequence[TrackedObject]], optional
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
            3. random choice `"random"`

        If using `by_id` or `by_label` strategy but your objects don't
        have that field defined (Detections never have ids) the
        selected color will be the same for all objects (Palette's default Color).
    thickness : Optional[int], optional
        Thickness or width of the line.
    random_color : bool, optional
        **Deprecated**. Set color="random".
    color_by_label : bool, optional
        **Deprecated**. Set color="by_label".
    draw_labels : bool, optional
        If set to True, the label is added to a title that is drawn on top of the box.
        If an object doesn't have a label this parameter is ignored.
    draw_scores : bool, optional
        If set to True, the score is added to a title that is drawn on top of the box.
        If an object doesn't have a label this parameter is ignored.
    text_size : Optional[float], optional
        Size of the title, the value is used as a multiplier of the base size of the font.
        By default the size is scaled automatically based on the frame size.
    draw_ids : bool, optional
        If set to True, the id is added to a title that is drawn on top of the box.
        If an object doesn't have an id this parameter is ignored.
    text_color : Optional[ColorLike], optional
        Color of the text. By default the same color as the box is used.
    text_thickness : Optional[int], optional
        Thickness of the font. By default it's scaled with the `text_size`.
    draw_box : bool, optional
        Set to False to hide the box and just draw the text.
    detections : Sequence[Detection], optional
        **Deprecated**. Use drawables.
    line_color: Optional[ColorLike], optional
        **Deprecated**. Use color.
    line_width: Optional[int], optional
        **Deprecated**. Use thickness.
    label_size: Optional[int], optional
        **Deprecated**. Use text_size.

    Returns
    -------
    np.ndarray
        The resulting frame.
    """
    #
    # handle deprecated parameters
    #
    if random_color is not None:
        warn_once(
            'Parameter "random_color" is deprecated, set `color="random"` instead'
        )
        color = "random"
    if color_by_label is not None:
        warn_once(
            'Parameter "color_by_label" is deprecated, set `color="by_label"` instead'
        )
        color = "by_label"
    if detections is not None:
        warn_once('Parameter "detections" is deprecated, use "drawables" instead')
        drawables = detections
    if line_color is not None:
        warn_once('Parameter "line_color" is deprecated, use "color" instead')
        color = line_color
    if line_width is not None:
        warn_once('Parameter "line_width" is deprecated, use "thickness" instead')
        thickness = line_width
    if label_size is not None:
        warn_once('Parameter "label_size" is deprecated, use "text_size" instead')
        text_size = label_size
    # end

    if color is None:
        color = "by_id"
    if thickness is None:
        thickness = int(max(frame.shape) / 500)

    if drawables is None:
        return frame

    if text_color is not None:
        text_color = parse_color(text_color)

    for obj in drawables:
        if not isinstance(obj, Drawable):
            d = Drawable(obj)
        else:
            d = obj

        if color == "by_id":
            obj_color = Palette.choose_color(d.id)
        elif color == "by_label":
            obj_color = Palette.choose_color(d.label)
        elif color == "random":
            obj_color = Palette.choose_color(np.random.rand())
        else:
            obj_color = parse_color(color)

        points = d.points.astype(int)
        if draw_box:
            Drawer.rectangle(
                frame,
                tuple(points),
                color=obj_color,
                thickness=thickness,
            )

        text = _build_text(
            d, draw_labels=draw_labels, draw_ids=draw_ids, draw_scores=draw_scores
        )
        if text:
            if text_color is None:
                obj_text_color = obj_color
            else:
                obj_text_color = text_color
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


def draw_tracked_boxes(
    frame: np.ndarray,
    objects: Sequence["TrackedObject"],
    border_colors: Optional[Tuple[int, int, int]] = None,
    border_width: Optional[int] = None,
    id_size: Optional[int] = None,
    id_thickness: Optional[int] = None,
    draw_box: bool = True,
    color_by_label: bool = False,
    draw_labels: bool = False,
    label_size: Optional[int] = None,
    label_width: Optional[int] = None,
) -> np.array:
    "**Deprecated**. Use [`draw_box`][norfair.drawing.draw_boxes.draw_boxes]"
    warn_once("draw_tracked_boxes is deprecated, use draw_box instead")
    return draw_boxes(
        frame=frame,
        drawables=objects,
        color="by_label" if color_by_label else border_colors,
        thickness=border_width,
        text_size=label_size or id_size,
        text_thickness=id_thickness or label_width,
        draw_labels=draw_labels,
        draw_ids=id_size is not None and id_size > 0,
        draw_box=draw_box,
    )
