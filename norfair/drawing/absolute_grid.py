from functools import lru_cache

import numpy as np

from norfair.camera_motion import CoordinatesTransformation

from .color import Color, ColorType
from .drawer import Drawer


@lru_cache(maxsize=4)
def _get_grid(size, w, h, polar=False):
    """
    Construct the grid of points.

    Points are choosen
    Results are cached since the grid in absolute coordinates doesn't change.
    """
    # We need to get points on a semi-sphere of radious 1 centered around (0, 0)

    # First step is to get a grid of angles, theta and phi ∈ (-pi/2, pi/2)
    step = np.pi / size
    start = -np.pi / 2 + step / 2
    end = np.pi / 2
    theta, fi = np.mgrid[start:end:step, start:end:step]

    if polar:
        # if polar=True the first frame will show points as if
        # you are on the center of the earth looking at one of the poles.
        # Points on the sphere are defined as [sin(theta) * cos(fi), sin(theta) * sin(fi), cos(theta)]
        # Then we need to intersect the line defined by the point above with the
        # plane z=1 which is the "absolute plane", we do so by dividing by cos(theta), the result becomes
        # [tan(theta) * cos(fi), tan(theta) * sin(phi), 1]
        # note that the z=1 is implied by the coord_transformation so there is no need to add it.
        tan_theta = np.tan(theta)

        X = tan_theta * np.cos(fi)
        Y = tan_theta * np.sin(fi)
    else:
        # otherwhise will show as if you were looking at the equator
        X = np.tan(fi)
        Y = np.divide(np.tan(theta), np.cos(fi))
    # construct the points as x, y coordinates
    points = np.vstack((X.flatten(), Y.flatten())).T
    # scale and center the points
    return points * max(h, w) + np.array([w // 2, h // 2])


def draw_absolute_grid(
    frame: np.ndarray,
    coord_transformations: CoordinatesTransformation,
    grid_size: int = 20,
    radius: int = 2,
    thickness: int = 1,
    color: ColorType = Color.black,
    polar: bool = False,
):
    """
    Draw a grid of points in absolute coordinates.

    Useful for debugging camera motion.

    The points are drawn as if the camera were in the center of a sphere and points are drawn in the intersection
    of latitude and longitude lines over the surface of the sphere.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on.
    coord_transformations : CoordinatesTransformation
        The coordinate transformation as returned by the [`MotionEstimator`][norfair.camera_motion.MotionEstimator]
    grid_size : int, optional
        How many points to draw.
    radius : int, optional
        Size of each point.
    thickness : int, optional
        Thickness of each point
    color : ColorType, optional
        Color of the points.
    polar : Bool, optional
        If True, the points on the first frame are drawn as if the camera were pointing to a pole (viewed from the center of the earth).
        By default, False is used which means the points are drawn as if the camera were pointing to the Equator.
    """
    h, w, _ = frame.shape

    # get absolute points grid
    points = _get_grid(grid_size, w, h, polar=polar)

    # transform the points to relative coordinates
    if coord_transformations is None:
        points_transformed = points
    else:
        points_transformed = coord_transformations.abs_to_rel(points)

    # filter points that are not visible
    visible_points = points_transformed[
        (points_transformed <= np.array([w, h])).all(axis=1)
        & (points_transformed >= 0).all(axis=1)
    ]
    for point in visible_points:
        Drawer.cross(
            frame, point.astype(int), radius=radius, thickness=thickness, color=color
        )
