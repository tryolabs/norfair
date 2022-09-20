import argparse
from typing import List

import numpy as np
from draw import center, draw
from yolo import YOLO, yolo_detections_to_norfair_detections

from norfair import AbsolutePaths, Paths, Tracker, Video
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from norfair.distances import create_normalized_mean_euclidean_distance

DISTANCE_THRESHOLD_CENTROID: float = 0.08


def inference(
    input_video: str, model: str, track_points: str, model_threshold: str, classes: List
):
    coord_transformations = None
    paths_drawer = None
    fix_paths = True
    model = YOLO(model)
    video = Video(input_path=input_video)

    transformations_getter = HomographyTransformationGetter()

    motion_estimator = MotionEstimator(
        max_points=500, min_distance=7, transformations_getter=transformations_getter
    )

    distance_function = create_normalized_mean_euclidean_distance(
        video.input_height, video.input_width
    )
    distance_threshold = DISTANCE_THRESHOLD_CENTROID

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    paths_drawer = Paths(center, attenuation=0.01)

    if fix_paths:
        paths_drawer = AbsolutePaths(max_history=40, thickness=2)

    for frame in video:
        yolo_detections = model(
            frame,
            conf_threshold=model_threshold,
            iou_threshold=0.45,
            image_size=720,
            classes=classes,
        )

        mask = np.ones(frame.shape[:2], frame.dtype)

        coord_transformations = motion_estimator.update(frame, mask)

        detections = yolo_detections_to_norfair_detections(
            yolo_detections, track_points=track_points
        )

        tracked_objects = tracker.update(
            detections=detections, coord_transformations=coord_transformations
        )

        frame = draw(
            paths_drawer,
            track_points,
            frame,
            detections,
            tracked_objects,
            coord_transformations,
            fix_paths,
        )
        video.write(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, help="Video files to process")
    parser.add_argument(
        "--detector-path", type=str, default="yolov7.pt", help="YOLOv7 model path"
    )
    parser.add_argument(
        "--img-size", type=int, default="720", help="YOLOv7 inference size (pixels)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default="0.25",
        help="YOLOv7 object confidence threshold",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="Filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--track-points",
        type=str,
        default="bbox",
        help="Track points: 'centroid' or 'bbox'",
    )
    args = parser.parse_args()

    inference(
        args.files,
        args.detector_path,
        args.track_points,
        args.conf_threshold,
        args.classes,
    )
