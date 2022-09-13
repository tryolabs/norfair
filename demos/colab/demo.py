import argparse
import glob
import os
import tempfile

import numpy as np
from distance_function import euclidean_distance, iou
from draw import center, draw
from yolo import YOLO, yolo_detections_to_norfair_detections

from norfair import AbsolutePaths, Paths, Tracker, Video
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator

DISTANCE_THRESHOLD_BBOX = 3.33
DISTANCE_THRESHOLD_CENTROID = 30


def inference(
    input_video: str,
    model: str,
    track_points: str,
    model_threshold: str,
):
    temp_dir = tempfile.TemporaryDirectory()
    output_path = temp_dir.name

    coord_transformations = None
    paths_drawer = None
    fix_paths = False
    model = YOLO(model)
    video = Video(input_path=input_video, output_path=output_path)

    motion_estimation = True

    drawing_paths = True

    if motion_estimation and drawing_paths:
        fix_paths = True

    if motion_estimation:
        transformations_getter = HomographyTransformationGetter()

        motion_estimator = MotionEstimator(
            max_points=500, min_distance=7, transformations_getter=transformations_getter
        )

    distance_function = iou if track_points == "bbox" else euclidean_distance
    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX if track_points == "bbox" else DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    if drawing_paths:
        paths_drawer = Paths(center, attenuation=0.01)

    if fix_paths:
        paths_drawer = AbsolutePaths(max_history=5, thickness=2)

    for frame in video:
        yolo_detections = model(
            frame, conf_threshold=model_threshold, iou_threshold=0.45, image_size=720
        )

        mask = np.ones(frame.shape[:2], frame.dtype)

        if motion_estimation:
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

    base_file_name = input_video.split("/")[-1].split(".")[0]
    file_name = base_file_name + "_out.mp4"
    return os.path.join(output_path, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, help="Video files to process")
    parser.add_argument("--detector-path", type=str, default="yolov7.pt", help="YOLOv7 model path")
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
        "--iou-threshold", type=float, default="0.45", help="YOLOv7 IOU threshold for NMS"
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
        default="centroid",
        help="Track points: 'centroid' or 'bbox'",
    )
    args = parser.parse_args()

    inference(args.files, args.detector_path, args.track_points, args.conf_threshold)
