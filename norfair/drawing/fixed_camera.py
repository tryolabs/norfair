from typing import Union

import cv2
import numpy as np

from norfair.camera_motion import HomographyTransformation, TranslationTransformation


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
        self,
        frame: np.ndarray,
        coord_transformation: Union[
            HomographyTransformation, TranslationTransformation
        ],
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
            # top_left is the anchor coordinate from where we start drawing the fame on top of the background
            self.top_left = (
                np.array(self._background.shape[:2]) // 2
                - np.array(frame.shape[:2]) // 2
            )
        else:
            self._background = (self._background * self._attenuation_factor).astype(
                frame.dtype
            )

        # warp the frame with the following composition:
        # top_left_translation o rel_to_abs
        if isinstance(coord_transformation, HomographyTransformation):
            top_left_translation = np.array(
                [[1, 0, self.top_left[1]], [0, 1, self.top_left[0]], [0, 0, 1]]
            )
            full_transformation = (
                top_left_translation @ coord_transformation.inverse_homography_matrix
            )
            background_with_current_frame = cv2.warpPerspective(
                frame,
                full_transformation,
                tuple(self._background.shape[:2][::-1]),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        elif isinstance(coord_transformation, TranslationTransformation):

            full_transformation = np.array(
                [
                    [1, 0, self.top_left[1] - coord_transformation.movement_vector[0]],
                    [0, 1, self.top_left[0] - coord_transformation.movement_vector[1]],
                ]
            )
            background_with_current_frame = cv2.warpAffine(
                frame,
                full_transformation,
                tuple(self._background.shape[:2][::-1]),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        self._background = cv2.addWeighted(
            self._background,
            0.5,
            background_with_current_frame,
            0.5,
            0.0,
        )
        return self._background
