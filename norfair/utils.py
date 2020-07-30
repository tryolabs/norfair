import numpy as np


def validate_points(points):
    # If the user is tracking only a single point, reformat it slightly.
    if points.shape == (2,):
        points = points[np.newaxis, ...]
    else:
        if points.shape[1] != 2 or len(points.shape) > 2:
            print(f"The shape of `Detection.points` should be (num_of_points_to_track, 2), not {points.shape}.")
            print("Check your detection conversion code.")
            exit()
    return points
