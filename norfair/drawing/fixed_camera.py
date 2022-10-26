import numpy as np

from norfair.camera_motion import TranslationTransformation
from norfair.utils import warn_once


class FixedCamera:
    """
    Class used to stabilize video based on the camera motion.

    Starts with a larger frame, where the original frame is drawn on top of a black background.
    As the camera moves, the smaller frame moves in the opposite direction, stabilizing the objects in it.

    Useful for debugging or demoing the camera motion.
    ![Example GIF](../../videos/camera_stabilization.gif)

    !!! Warning
        This only works with [`TranslationTransformation`][norfair.camera_motion.TranslationTransformation],
        using [`HomographyTransformation`][norfair.camera_motion.HomographyTransformation] will result in
        unexpected behaviour.

    !!! Warning
        If using other drawers, always apply this one last. Using other drawers on the scaled up frame will not work as expected.

    !!! Note
        Sometimes the camera moves so far from the original point that the result won't fit in the scaled-up frame.
        In this case, a warning will be logged and the frames will be cropped to avoid errors.

    Parameters
    ----------
    scale : float, optional
        The resulting video will have a resolution of `scale * (H, W)` where HxW is the resolution of the original video.
        Use a bigger scale if the camera is moving too much.
    attenuation : float, optional
        Controls how fast the older frames fade to black.

    Examples
    --------
    >>> # setup
    >>> tracker = Tracker("frobenious", 100)
    >>> motion_estimator = MotionEstimator()
    >>> video = Video(input_path="video.mp4")
    >>> fixed_camera = FixedCamera()
    >>> # process video
    >>> for frame in video:
    >>>     coord_transformations = motion_estimator.update(frame)
    >>>     detections = get_detections(frame)
    >>>     tracked_objects = tracker.update(detections, coord_transformations)
    >>>     draw_tracked_objects(frame, tracked_objects)  # fixed_camera should always be the last drawer
    >>>     bigger_frame = fixed_camera.adjust_frame(frame, coord_transformations)
    >>>     video.write(bigger_frame)
    """

    def __init__(self, scale: float = 2, attenuation: float = 0.05):
        self.scale = scale
        self._background = None
        self._attenuation_factor = 1 - attenuation

    def adjust_frame(
        self, frame: np.ndarray, coord_transformation: TranslationTransformation
    ) -> np.ndarray:
        """
        Render scaled up frame.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame.
        coord_transformation : TranslationTransformation
            The coordinate transformation as returned by the [`MotionEstimator`][norfair.camera_motion.MotionEstimator]

        Returns
        -------
        np.ndarray
            The new bigger frame with the original frame drawn on it.
        """

        # initialize background if necessary
        if self._background is None:
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
        else:
            self._background = (self._background * self._attenuation_factor).astype(
                frame.dtype
            )

        # top_left is the anchor coordinate from where we start drawing the fame on top of the background
        # aim to draw it in the center of the background but transformations will move this point
        top_left = (
            np.array(self._background.shape[:2]) // 2 - np.array(frame.shape[:2]) // 2
        )
        top_left = (
            coord_transformation.rel_to_abs(top_left[::-1]).round().astype(int)[::-1]
        )
        # box of the background that will be updated and the limits of it
        background_y0, background_y1 = (top_left[0], top_left[0] + frame.shape[0])
        background_x0, background_x1 = (top_left[1], top_left[1] + frame.shape[1])
        background_size_y, background_size_x = self._background.shape[:2]

        # define box of the frame that will be used
        # if the scale is not enough to support the movement, warn the user but keep drawing
        # cropping the frame so that the operation doesn't fail
        frame_y0, frame_y1, frame_x0, frame_x1 = (0, frame.shape[0], 0, frame.shape[1])
        if (
            background_y0 < 0
            or background_x0 < 0
            or background_y1 > background_size_y
            or background_x1 > background_size_x
        ):
            warn_once(
                "moving_camera_scale is not enough to cover the range of camera movement, frame will be cropped"
            )
            # crop left or top of the frame if necessary
            frame_y0 = max(-background_y0, 0)
            frame_x0 = max(-background_x0, 0)
            # crop right or bottom of the frame if necessary
            frame_y1 = max(
                min(background_size_y - background_y0, background_y1 - background_y0), 0
            )
            frame_x1 = max(
                min(background_size_x - background_x0, background_x1 - background_x0), 0
            )
            # handle cases where the limits of the background become negative which numpy will interpret incorrectly
            background_y0 = max(background_y0, 0)
            background_x0 = max(background_x0, 0)
            background_y1 = max(background_y1, 0)
            background_x1 = max(background_x1, 0)
        self._background[
            background_y0:background_y1, background_x0:background_x1, :
        ] = frame[frame_y0:frame_y1, frame_x0:frame_x1, :]
        return self._background
