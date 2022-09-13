import numpy as np

import norfair


def draw(
    paths_drawer,
    track_points,
    frame,
    detections,
    tracked_objects,
    coord_transformations,
    fix_paths,
):
    if track_points == "centroid":
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
    elif track_points == "bbox":
        norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_boxes(frame, tracked_objects)

    if fix_paths:
        frame = paths_drawer.draw(frame, tracked_objects, coord_transformations)
    elif paths_drawer is not None:
        frame = paths_drawer.draw(frame, tracked_objects)

    return frame


def center(points):
    return [np.mean(np.array(points), axis=0)]
