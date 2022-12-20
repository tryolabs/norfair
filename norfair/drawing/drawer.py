from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from norfair.drawing.color import Color, ColorType
from norfair.tracker import Detection, TrackedObject

try:
    import cv2
except ImportError:
    from norfair.utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()


class Drawer:
    """
    Basic drawing functionality.

    This class encapsulates opencv drawing functions allowing for
    different backends to be implemented following the same interface.
    """

    @classmethod
    def circle(
        cls,
        frame: np.ndarray,
        position: Tuple[int, int],
        radius: Optional[int] = None,
        thickness: Optional[int] = None,
        color: ColorType = None,
    ) -> np.ndarray:
        """
        Draw a circle.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        position : Tuple[int, int]
            Position of the point. This will become the center of the circle.
        radius : Optional[int], optional
            Radius of the circle.
        thickness : Optional[int], optional
            Thickness or width of the line.
        color : Color, optional
            A tuple of ints describing the BGR color `(0, 0, 255)`.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
        if radius is None:
            radius = int(max(max(frame.shape) * 0.005, 1))
        if thickness is None:
            thickness = radius - 1

        return cv2.circle(
            frame,
            position,
            radius=radius,
            color=color,
            thickness=thickness,
        )

    @classmethod
    def text(
        cls,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        size: Optional[float] = None,
        color: Optional[ColorType] = None,
        thickness: Optional[int] = None,
        shadow: bool = True,
        shadow_color: ColorType = Color.black,
        shadow_offset: int = 1,
    ) -> np.ndarray:
        """
        Draw text

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        text : str
            The text to be written.
        position : Tuple[int, int]
            Position of the bottom-left corner of the text.
            This value is adjusted considering the thickness automatically.
        size : Optional[float], optional
            Scale of the font, by default chooses a sensible value is picked based on the size of the frame.
        color : Optional[ColorType], optional
            Color of the text, by default is black.
        thickness : Optional[int], optional
            Thickness of the lines, by default a sensible value is picked based on the size.
        shadow : bool, optional
            If True, a shadow of the text is added which improves legibility.
        shadow_color : Color, optional
            Color of the shadow.
        shadow_offset : int, optional
            Offset of the shadow.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
        if size is None:
            size = min(max(max(frame.shape) / 4000, 0.5), 1.5)
        if thickness is None:
            thickness = int(round(size) + 1)

        if thickness is None and size is not None:
            thickness = int(round(size) + 1)
        # adjust position based on the thickness
        anchor = (position[0] + thickness // 2, position[1] - thickness // 2)
        if shadow:
            frame = cv2.putText(
                frame,
                text,
                (anchor[0] + shadow_offset, anchor[1] + shadow_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                shadow_color,
                thickness,
                cv2.LINE_AA,
            )
        return cv2.putText(
            frame,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @classmethod
    def rectangle(
        cls,
        frame: np.ndarray,
        points: Sequence[Tuple[int, int]],
        color: Optional[ColorType] = None,
        thickness: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw a rectangle

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        points : Sequence[Tuple[int, int]]
            Points describing the rectangle in the format `[[x0, y0], [x1, y1]]`.
        color : Optional[ColorType], optional
            Color of the lines, by default Black.
        thickness : Optional[int], optional
            Thickness of the lines, by default 1.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
        frame = cv2.rectangle(
            frame,
            tuple(points[0]),
            tuple(points[1]),
            color=color,
            thickness=thickness,
        )
        return frame

    @classmethod
    def cross(
        cls,
        frame: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: ColorType,
        thickness: int,
    ) -> np.ndarray:
        """
        Draw a cross

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        center : Tuple[int, int]
            Center of the cross.
        radius : int
            Size or radius of the cross.
        color : Color
            Color of the lines.
        thickness : int
            Thickness of the lines.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
        middle_x, middle_y = center
        left, top = center - radius
        right, bottom = center + radius
        frame = cls.line(
            frame,
            start=(middle_x, top),
            end=(middle_x, bottom),
            color=color,
            thickness=thickness,
        )
        frame = cls.line(
            frame,
            start=(left, middle_y),
            end=(right, middle_y),
            color=color,
            thickness=thickness,
        )
        return frame

    @classmethod
    def line(
        cls,
        frame: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: ColorType = Color.black,
        thickness: int = 1,
    ) -> np.ndarray:
        """
        Draw a line.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        start : Tuple[int, int]
            Starting point.
        end : Tuple[int, int]
            End point.
        color : ColorType, optional
            Line color.
        thickness : int, optional
            Line width.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
        return cv2.line(
            frame,
            pt1=start,
            pt2=end,
            color=color,
            thickness=thickness,
        )

    @classmethod
    def alpha_blend(
        cls,
        frame1: np.ndarray,
        frame2: np.ndarray,
        alpha: float = 0.5,
        beta: Optional[float] = None,
        gamma: float = 0,
    ) -> np.ndarray:
        """
        Blend 2 frame as a wheigthted sum.

        Parameters
        ----------
        frame1 : np.ndarray
            An OpenCV frame.
        frame2 : np.ndarray
            An OpenCV frame.
        alpha : float, optional
            Weight of frame1.
        beta : Optional[float], optional
            Weight of frame2, by default `1 - alpha`
        gamma : float, optional
            Scalar to add to the sum.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
        if beta is None:
            beta = 1 - alpha
        return cv2.addWeighted(
            src1=frame1, src2=frame2, alpha=alpha, beta=beta, gamma=gamma
        )


class Drawable:
    """
    Class to standardize Drawable objects like Detections and TrackedObjects

    Parameters
    ----------
    obj : Union[Detection, TrackedObject], optional
        A [Detection][norfair.tracker.Detection] or a [TrackedObject][norfair.tracker.TrackedObject]
        that will be used to initialized the drawable.
        If this parameter is passed, all other arguments are ignored
    points : np.ndarray, optional
        Points included in the drawable, shape is `(N_points, N_dimensions)`. Ignored if `obj` is passed
    id : Any, optional
        Id of this object. Ignored if `obj` is passed
    label : Any, optional
        Label specifying the class of the object. Ignored if `obj` is passed
    scores : np.ndarray, optional
        Confidence scores of each point, shape is `(N_points,)`. Ignored if `obj` is passed
    live_points : np.ndarray, optional
        Bolean array indicating which points are alive, shape is `(N_points,)`. Ignored if `obj` is passed

    Raises
    ------
    ValueError
        If obj is not an instance of the supported classes.
    """

    def __init__(
        self,
        obj: Union[Detection, TrackedObject] = None,
        points: np.ndarray = None,
        id: Any = None,
        label: Any = None,
        scores: np.ndarray = None,
        live_points: np.ndarray = None,
    ) -> None:
        if isinstance(obj, Detection):
            self.points = obj.points
            self.id = None
            self.label = obj.label
            self.scores = obj.scores
            # TODO: alive points for detections could be the ones over the threshold
            # but that info is not available here
            self.live_points = np.ones(obj.points.shape[0]).astype(bool)

        elif isinstance(obj, TrackedObject):
            self.points = obj.estimate
            self.id = obj.id
            self.label = obj.label
            # TODO: TrackedObject.scores could be an interesting thing to have
            # it could be the scores of the last detection or some kind of moving average
            self.scores = None
            self.live_points = obj.live_points
        elif obj is None:
            self.points = points
            self.id = id
            self.label = label
            self.scores = scores
            self.live_points = live_points
        else:
            raise ValueError(
                f"Extecting a Detection or a TrackedObject but received {type(obj)}"
            )
