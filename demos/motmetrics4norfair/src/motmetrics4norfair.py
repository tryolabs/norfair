import argparse
import os.path

import numpy as np

from norfair import Tracker, drawing, metrics, video
from norfair.camera_motion import MotionEstimator
from norfair.filter import FilterPyKalmanFilterFactory

DETECTION_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 0.9
POINTWISE_HIT_COUNTER_MAX = 3
HIT_COUNTER_MAX = 8


def build_mask(frame, detections, tracked_objects):
    # create a mask of ones
    mask = np.ones(frame.shape[:2], frame.dtype)
    # set to 0 on detections and tracked_objects
    for det in detections:
        i = det.points.astype(int)
        mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0
    for obj in tracked_objects:
        i = obj.estimate.astype(int)
        mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0
    return mask


parser = argparse.ArgumentParser(
    description="Evaluate a basic tracker on MOTChallenge data. Display on terminal the MOTChallenge metrics results "
)
parser.add_argument(
    "dataset_path",
    type=str,
    nargs="?",
    help="Path to the MOT Challenge train dataset folder (test dataset doesn't provide labels)",
)
parser.add_argument(
    "--make-video",
    action="store_true",
    help="Generate an output video, using the frames provided by the MOTChallenge dataset.",
)
parser.add_argument(
    "--save-pred",
    action="store_true",
    help="Generate a text file with your predictions",
)
parser.add_argument(
    "--save-metrics",
    action="store_true",
    help="Generate a text file with your MOTChallenge metrics results",
)
parser.add_argument(
    "--output-path", type=str, nargs="?", default=".", help="Output path"
)
parser.add_argument(
    "--select-sequences",
    type=str,
    nargs="+",
    help="If you want to select a subset of sequences in your dataset path. Insert the names of the sequences you want to process.",
)

args = parser.parse_args()

output_path = args.output_path

if args.save_metrics:
    print("Saving metrics file at " + os.path.join(output_path, "metrics.txt"))
if args.save_pred:
    print("Saving predictions files at " + os.path.join(output_path, "predictions/"))
if args.make_video:
    print("Saving videos at " + os.path.join(output_path, "videos/"))

if args.select_sequences is None:
    sequences_paths = [f.path for f in os.scandir(args.dataset_path) if f.is_dir()]
else:
    sequences_paths = [
        os.path.join(args.dataset_path, f) for f in args.select_sequences
    ]

accumulator = metrics.Accumulators()

for input_path in sequences_paths:
    # Search vertical resolution in seqinfo.ini
    seqinfo_path = os.path.join(input_path, "seqinfo.ini")
    info_file = metrics.InformationFile(file_path=seqinfo_path)

    all_detections = metrics.DetectionFileParser(
        input_path=input_path, information_file=info_file
    )

    if args.save_pred:
        predictions_text_file = metrics.PredictionsTextFile(
            input_path=input_path, save_path=output_path, information_file=info_file
        )

    video_file = video.VideoFromFrames(
        input_path=input_path,
        save_path=output_path,
        information_file=info_file,
        make_video=args.make_video,
    )

    tracker = Tracker(
        distance_function="iou",
        distance_threshold=DISTANCE_THRESHOLD,
        detection_threshold=DETECTION_THRESHOLD,
        pointwise_hit_counter_max=POINTWISE_HIT_COUNTER_MAX,
        hit_counter_max=HIT_COUNTER_MAX,
        filter_factory=FilterPyKalmanFilterFactory(),
    )

    motion_estimator = MotionEstimator(max_points=500)

    # Initialize accumulator for this video
    accumulator.create_accumulator(input_path=input_path, information_file=info_file)

    tracked_objects = []
    for frame, detections in zip(video_file, all_detections):
        mask = build_mask(frame, detections, tracked_objects)
        coord_transformations = motion_estimator.update(frame, mask)

        tracked_objects = tracker.update(
            detections=detections,
            coord_transformations=coord_transformations,
        )

        # Draw detection and tracked object boxes on frame
        if args.make_video:
            frame = drawing.draw_boxes(frame, detections=detections)
            frame = drawing.draw_tracked_boxes(frame=frame, objects=tracked_objects)
            video_file.update(frame=frame)

        # Update output text file
        if args.save_pred:
            predictions_text_file.update(predictions=tracked_objects)

        accumulator.update(predictions=tracked_objects)

accumulator.compute_metrics()
accumulator.print_metrics()

if args.save_metrics:
    accumulator.save_metrics(save_path=output_path)
