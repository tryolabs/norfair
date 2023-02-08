from collections import defaultdict
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from norfair.drawing.color import Palette
from norfair.drawing.drawer import Drawer
from norfair.tracker import TrackedObject
from norfair.utils import warn_once


class Paths:
    """
    Class that draws the paths taken by a set of points of interest defined from the coordinates of each tracker estimation.

    Parameters
    ----------
    get_points_to_draw : Optional[Callable[[np.array], np.array]], optional
        Function that takes a list of points (the `.estimate` attribute of a [`TrackedObject`][norfair.tracker.TrackedObject])
        and returns a list of points for which we want to draw their paths.

        By default it is the mean point of all the points in the tracker.
    thickness : Optional[int], optional
        Thickness of the circles representing the paths of interest.
    color : Optional[Tuple[int, int, int]], optional
        [Color][norfair.drawing.Color] of the circles representing the paths of interest.
    radius : Optional[int], optional
        Radius of the circles representing the paths of interest.
    attenuation : float, optional
        A float number in [0, 1] that dictates the speed at which the path is erased.
        if it is `0` then the path is never erased.

    Examples
    --------
    >>> from norfair import Tracker, Video, Path
    >>> video = Video("video.mp4")
    >>> tracker = Tracker(...)
    >>> path_drawer = Path()
    >>> for frame in video:
    >>>    detections = get_detections(frame)  # runs detector and returns Detections
    >>>    tracked_objects = tracker.update(detections)
    >>>    frame = path_drawer.draw(frame, tracked_objects)
    >>>    video.write(frame)
    """

    def __init__(
        self,
        get_points_to_draw: Optional[Callable[[np.array], np.array]] = None,
        thickness: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None,
        radius: Optional[int] = None,
        attenuation: float = 0.01,
    ):
        if get_points_to_draw is None:

            def get_points_to_draw(points):
                return [np.mean(np.array(points), axis=0)]

        self.get_points_to_draw = get_points_to_draw

        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.mask = None
        self.attenuation_factor = 1 - attenuation

    def draw(
        self, frame: np.ndarray, tracked_objects: Sequence[TrackedObject]
    ) -> np.array:
        """
        Draw the paths of the points interest on a frame.

        !!! warning
            This method does **not** draw frames in place as other drawers do, the resulting frame is returned.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on.
        tracked_objects : Sequence[TrackedObject]
            List of [`TrackedObject`][norfair.tracker.TrackedObject] to get the points of interest in order to update the paths.

        Returns
        -------
        np.array
            The resulting frame.
        """
        if self.mask is None:
            frame_scale = frame.shape[0] / 100

            if self.radius is None:
                self.radius = int(max(frame_scale * 0.7, 1))
            if self.thickness is None:
                self.thickness = int(max(frame_scale / 7, 1))

            self.mask = np.zeros(frame.shape, np.uint8)

        self.mask = (self.mask * self.attenuation_factor).astype("uint8")

        for obj in tracked_objects:
            if obj.abs_to_rel is not None:
                warn_once(
                    "It seems that your using the Path drawer together with MotionEstimator. This is not fully supported and the results will not be what's expected"
                )

            if self.color is None:
                color = Palette.choose_color(obj.id)
            else:
                color = self.color

            points_to_draw = self.get_points_to_draw(obj.estimate)

            for point in points_to_draw:
                self.mask = Drawer.circle(
                    self.mask,
                    position=tuple(point.astype(int)),
                    radius=self.radius,
                    color=color,
                    thickness=self.thickness,
                )

        return Drawer.alpha_blend(self.mask, frame, alpha=1, beta=1)


class AbsolutePaths:
    """
    Class that draws the absolute paths taken by a set of points.

    Works just like [`Paths`][norfair.drawing.Paths] but supports camera motion.

    !!! warning
        This drawer is not optimized so it can be stremely slow. Performance degrades linearly with
        `max_history * number_of_tracked_objects`.

    Parameters
    ----------
    get_points_to_draw : Optional[Callable[[np.array], np.array]], optional
        Function that takes a list of points (the `.estimate` attribute of a [`TrackedObject`][norfair.tracker.TrackedObject])
        and returns a list of points for which we want to draw their paths.

        By default it is the mean point of all the points in the tracker.
    thickness : Optional[int], optional
        Thickness of the circles representing the paths of interest.
    color : Optional[Tuple[int, int, int]], optional
        [Color][norfair.drawing.Color] of the circles representing the paths of interest.
    radius : Optional[int], optional
        Radius of the circles representing the paths of interest.
    max_history : int, optional
        Number of past points to include in the path. High values make the drawing slower

    Examples
    --------
    >>> from norfair import Tracker, Video, Path
    >>> video = Video("video.mp4")
    >>> tracker = Tracker(...)
    >>> path_drawer = Path()
    >>> for frame in video:
    >>>    detections = get_detections(frame)  # runs detector and returns Detections
    >>>    tracked_objects = tracker.update(detections)
    >>>    frame = path_drawer.draw(frame, tracked_objects)
    >>>    video.write(frame)
    """

    def __init__(
        self,
        get_points_to_draw: Optional[Callable[[np.array], np.array]] = None,
        thickness: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None,
        radius: Optional[int] = None,
        max_history=20,
    ):

        if get_points_to_draw is None:

            def get_points_to_draw(points):
                return [np.mean(np.array(points), axis=0)]

        self.get_points_to_draw = get_points_to_draw

        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.past_points = defaultdict(lambda: [])
        self.max_history = max_history
        self.alphas = np.linspace(0.99, 0.01, max_history)

    def draw(self, frame, tracked_objects, coord_transform=None):
        frame_scale = frame.shape[0] / 100

        if self.radius is None:
            self.radius = int(max(frame_scale * 0.7, 1))
        if self.thickness is None:
            self.thickness = int(max(frame_scale / 7, 1))
        for obj in tracked_objects:
            if not obj.live_points.any():
                continue

            if self.color is None:
                color = Palette.choose_color(obj.id)
            else:
                color = self.color

            points_to_draw = self.get_points_to_draw(obj.get_estimate(absolute=True))

            for point in coord_transform.abs_to_rel(points_to_draw):
                Drawer.circle(
                    frame,
                    position=tuple(point.astype(int)),
                    radius=self.radius,
                    color=color,
                    thickness=self.thickness,
                )

            last = points_to_draw
            for i, past_points in enumerate(self.past_points[obj.id]):
                overlay = frame.copy()
                last = coord_transform.abs_to_rel(last)
                for j, point in enumerate(coord_transform.abs_to_rel(past_points)):
                    Drawer.line(
                        overlay,
                        tuple(last[j].astype(int)),
                        tuple(point.astype(int)),
                        color=color,
                        thickness=self.thickness,
                    )
                last = past_points

                alpha = self.alphas[i]
                frame = Drawer.alpha_blend(overlay, frame, alpha=alpha)
            self.past_points[obj.id].insert(0, points_to_draw)
            self.past_points[obj.id] = self.past_points[obj.id][: self.max_history]
        return frame
