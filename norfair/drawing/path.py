from collections import defaultdict
from typing import Callable, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:
    from norfair.utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()
import numpy as np

from norfair.camera_motion import HomographyTransformation, TranslationTransformation
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

    Parameters
    ----------
    get_points_to_draw : Optional[Callable[[np.array], np.array]], optional
        Function that takes a [`TrackedObject`][norfair.tracker.TrackedObject], and returns a list of points
        (in the absolute coordinate frame) for which we want to draw their paths.

        By default we just average the points with greatest height ('feet') if the object has live points.
    scale : Optional[float], optional
        Norfair will draw over a background canvas in the absolute coordinates. This determines how
        relatively bigger is this canvas with respect to the original frame.
        After the camera moves, part of the frame might get outside the canvas if scale is not large enough.
    attenuation : Optional[float], optional
        How fast we forget old points in the path. (0=Draw all points, 1=Draw only most current point)
    thickness : Optional[int], optional
        Thickness of the circles representing the paths of interest.
    color : Optional[Tuple[int, int, int]], optional
        [Color][norfair.drawing.Color] of the circles representing the paths of interest.
    radius : Optional[int], optional
        Radius of the circles representing the paths of interest.
    max_history : int, optional
        Number of past points to include in the path. High values make the drawing slower
    path_blend_factor: Optional[float], optional
        When blending the frame and the canvas (with the paths overdrawn), we do:
        frame = path_blend_factor * canvas + frame_blend_factor * frame
    frame_blend_factor:
        When blending the frame and the canvas (with the paths overdrawn), we do:
        frame = path_blend_factor * canvas + frame_blend_factor * frame

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
        scale: float = None,
        attenuation: float = 0.05,
        get_points_to_draw: Optional[Callable[[np.array], np.array]] = None,
        thickness: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None,
        radius: Optional[int] = None,
        path_blend_factor=2,
        frame_blend_factor=1,
    ):
        self.scale = scale
        self._background = None
        self._attenuation_factor = 1 - attenuation

        if get_points_to_draw is None:

            def get_points_to_draw(obj):
                # don't draw the object if we haven't seen it recently
                if not obj.live_points.any():
                    return []

                # obtain point with greatest height (feet)
                points_height = obj.estimate[:, 1]
                feet_indices = np.argwhere(points_height == points_height.max())
                # average their absolute positions
                try:
                    return np.mean(
                        obj.get_estimate(absolute=True)[feet_indices], axis=0
                    )
                except:
                    return np.mean(obj.estimate[feet_indices], axis=0)

        self.get_points_to_draw = get_points_to_draw
        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.path_blend_factor = path_blend_factor
        self.frame_blend_factor = frame_blend_factor

    def draw(self, frame, tracked_objects, coord_transformations=None):
        """
        the objects have a relative frame: frame_det
        the objects have an absolute frame: frame_one
        the frame passed could be either frame_det, or a new perspective where you want to draw the paths

        initialization:
         1. top_left is an arbitrary coordinate of some pixel inside background
        logic:
         1. draw track.get_estimate(absolute=True) + top_left, in background
         2. transform background with the composition (coord_transformations.abs_to_rel o minus_top_left_translation). If coord_transformations is None, only use minus_top_left_translation.
         3. crop [:frame.width, :frame.height] from the result
         4. overlay that over frame

        Remark:
        In any case, coord_transformations should be the coordinate transformation between the tracker absolute coords (as abs) and frame coords (as rel)
        """

        # initialize background if necessary
        if self._background is None:
            if self.scale is None:
                # set the default scale, depending if coord_transformations is provided or not
                if coord_transformations is None:
                    self.scale = 1
                else:
                    self.scale = 3

            original_size = (
                frame.shape[1],
                frame.shape[0],
            )  # OpenCV format is (width, height)

            scaled_size = tuple(
                (np.array(original_size) * np.array(self.scale)).round().astype(int)
            )
            self._background = np.zeros(
                [scaled_size[1], scaled_size[0], frame.shape[-1]],
                frame.dtype,
            )

            # this is the corner of the first passed frame (inside the background)
            self.top_left = (
                np.array(self._background.shape[:2]) // 2
                - np.array(frame.shape[:2]) // 2
            )
            self.top_left = self.top_left[::-1]
        else:
            self._background = (self._background * self._attenuation_factor).astype(
                frame.dtype
            )

        frame_scale = frame.shape[0] / 100

        if self.radius is None:
            self.radius = int(max(frame_scale * 0.7, 1))
        if self.thickness is None:
            self.thickness = int(max(frame_scale / 7, 1))

        # draw in background (each point in top_left_translation(abs_coordinate))
        for obj in tracked_objects:
            if self.color is None:
                color = Palette.choose_color(obj.id)
            else:
                color = self.color

            points_to_draw = self.get_points_to_draw(obj)

            for point in points_to_draw:
                Drawer.circle(
                    self._background,
                    position=tuple((point + self.top_left).astype(int)),
                    radius=self.radius,
                    color=color,
                    thickness=self.thickness,
                )

        # apply warp to self._background with composition abs_to_rel o -top_left_translation to background, and crop [:width, :height] to get frame overdrawn
        if isinstance(coord_transformations, HomographyTransformation):
            minus_top_left_translation = np.array(
                [[1, 0, -self.top_left[0]], [0, 1, -self.top_left[1]], [0, 0, 1]]
            )
            full_transformation = (
                coord_transformations.homography_matrix @ minus_top_left_translation
            )
            background_size_frame = cv2.warpPerspective(
                self._background,
                full_transformation,
                tuple(frame.shape[:2][::-1]),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        elif isinstance(coord_transformations, TranslationTransformation):
            full_transformation = np.array(
                [
                    [1, 0, coord_transformations.movement_vector[0] - self.top_left[0]],
                    [0, 1, coord_transformations.movement_vector[1] - self.top_left[1]],
                ]
            )
            background_size_frame = cv2.warpAffine(
                self._background,
                full_transformation,
                tuple(frame.shape[:2][::-1]),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        else:
            background_size_frame = self._background[
                self.top_left[1] : self.top_left[1] + frame.shape[0],
                self.top_left[0] : self.top_left[0] + frame.shape[1],
            ]

        frame = cv2.addWeighted(
            frame,
            self.frame_blend_factor,
            background_size_frame,
            self.path_blend_factor,
            0.0,
        )
        return frame
