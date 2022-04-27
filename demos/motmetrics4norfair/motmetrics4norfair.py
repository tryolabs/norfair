import argparse
import os.path

import numpy as np

from norfair import Tracker, drawing, metrics, video

frame_skip_period = 1
detection_threshold = 0.01
distance_threshold = 0.9
diagonal_proportion_threshold = 1 / 18
pointwise_hit_counter_max=3
hit_counter_max=2

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


def keypoints_distance(detected_pose, tracked_pose):
    norm_orders = [1, 2, np.inf]
    distances = 0
    diagonal = 0

    hor_min_pt = min(detected_pose.points[:, 0])
    hor_max_pt = max(detected_pose.points[:, 0])
    ver_min_pt = min(detected_pose.points[:, 1])
    ver_max_pt = max(detected_pose.points[:, 1])

    # Set keypoint_dist_threshold based on object size, and calculate
    # distance between detections and tracker estimations
    for p in norm_orders:
        distances += np.linalg.norm(
            detected_pose.points - tracked_pose.estimate, ord=p, axis=1
        )
        diagonal += np.linalg.norm(
            [hor_max_pt - hor_min_pt, ver_max_pt - ver_min_pt], ord=p
        )

    distances = distances / len(norm_orders)

    keypoint_dist_threshold = diagonal * diagonal_proportion_threshold

    match_num = np.count_nonzero(
        (distances < keypoint_dist_threshold)
        * (detected_pose.scores > detection_threshold)
        * (tracked_pose.last_detection.scores > detection_threshold)
    )
    return 1 / (1 + match_num)


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

    if args.make_video:
        video_file = video.VideoFromFrames(
            input_path=input_path, save_path=output_path, information_file=info_file
        )

    tracker = Tracker(
        distance_function=keypoints_distance,
        distance_threshold=distance_threshold,
        detection_threshold=detection_threshold,
        pointwise_hit_counter_max=pointwise_hit_counter_max,
        hit_counter_max=hit_counter_max,
    )

    # Initialize accumulator for this video
    accumulator.create_accumulator(input_path=input_path, information_file=info_file)

    for frame_number, detections in enumerate(all_detections):
        if frame_number % frame_skip_period == 0:
            tracked_objects = tracker.update(
                detections=detections, period=frame_skip_period
            )
        else:
            detections = []
            tracked_objects = tracker.update()

        # Draw detection and tracked object boxes on frame
        if args.make_video:
            frame = next(video_file)
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
