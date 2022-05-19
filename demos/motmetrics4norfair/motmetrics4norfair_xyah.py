import argparse
import os.path

import numpy as np

import sys
sys.path.append('../../norfair')
from tracker import Tracker
from norfair import drawing, metrics, video, Detection

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
    dest="make_video",
    action="store_true",
    help="Generate an output video, using the frames provided by the MOTChallenge dataset.",
)
parser.add_argument(
    "--save-pred",
    dest="save_pred",
    action="store_true",
    help="Generate a text file with your predictions",
)
parser.add_argument(
    "--save-metrics",
    dest="save_metrics",
    action="store_true",
    help="Generate a text file with your MOTChallenge metrics results",
)
parser.add_argument(
    "--output-path", dest="output_path", type=str, nargs="?", default=".", help="Output path"
)
parser.add_argument(
    "--select-sequences",
    dest="select_sequences",
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

class CartesianTrack:
    def __init__(self, estimate, obj_id, live_points):
        self.estimate = estimate
        self.id = obj_id
        self.live_points = live_points

# distance function for ByteTrack format
def iou_xyah(detected_pose, tracked_pose):
    height = tracked_pose.estimate[1, 1]
    width = tracked_pose.estimate[1, 0]*height
    area_tracked_pose = height*width
    half_size = np.array([width, height])/2

    top_left_corner_tracked_pose = tracked_pose.estimate[0] - half_size
    bottom_right_corner_tracked_pose = tracked_pose.estimate[0] + half_size


    height = detected_pose.points[1, 1]
    width = detected_pose.points[1, 0]*height
    area_detected_pose = height*width
    half_size = np.array([width, height])/2

    top_left_corner_detected_pose = detected_pose.points[0] - half_size
    bottom_right_corner_detected_pose = detected_pose.points[0] + half_size

    intersection = max(min(bottom_right_corner_detected_pose[1], bottom_right_corner_tracked_pose[1])-max(top_left_corner_tracked_pose[1], top_left_corner_detected_pose[1]), 0)*max(min(bottom_right_corner_detected_pose[0], bottom_right_corner_tracked_pose[0])-max(top_left_corner_tracked_pose[0], top_left_corner_detected_pose[0]), 0)
    union = area_detected_pose + area_tracked_pose - intersection

    return 1 - intersection/union


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
        distance_function=iou_xyah,
        distance_threshold=distance_threshold,
        detection_threshold=detection_threshold,
        pointwise_hit_counter_max=pointwise_hit_counter_max,
        hit_counter_max=hit_counter_max,
    )

    # Initialize accumulator for this video
    accumulator.create_accumulator(input_path=input_path, information_file=info_file)

    for frame_number, detections in enumerate(all_detections):
        
        # convert detections to ByteTrack format: (center_x, center_y, asp_ratio, height)
        xyah_detections = []
        for det in detections:
            center = np.mean(det.points, axis=0)
            width = det.points[1, 0] - det.points[0, 0]
            height = det.points[1, 1] - det.points[0, 1]
            xyah_state_det = np.vstack((center, [width/height, height]))
            xyah_detections.append(Detection(xyah_state_det, scores = det.scores))

        if frame_number % frame_skip_period == 0:
            tracked_objects = tracker.update(
                detections=xyah_detections, period=frame_skip_period
            )
        else:
            detections = []
            tracked_objects = tracker.update()

        x1y1x2y2_tracked_objects = []
        for n, obj in enumerate(tracked_objects):
            half_height = obj.estimate[1, 1]/2
            half_width = obj.estimate[1, 0]*half_height
            half_size = np.array([half_width, half_height])

            top_left_corner = obj.estimate[0] - half_size
            bottom_right_corner = obj.estimate[0] + half_size

            x1y1x2y2_tracked_objects.append(CartesianTrack(np.vstack((top_left_corner, bottom_right_corner)), obj.id, obj.live_points))

        # Draw detection and tracked object boxes on frame
        if args.make_video:
            frame = next(video_file)
            frame = drawing.draw_boxes(frame, detections=detections)
            frame = drawing.draw_tracked_boxes(frame=frame, objects=x1y1x2y2_tracked_objects)
            video_file.update(frame=frame)

        # Update output text file
        if args.save_pred:
            predictions_text_file.update(predictions=x1y1x2y2_tracked_objects)

        accumulator.update(predictions=x1y1x2y2_tracked_objects)

accumulator.compute_metrics()
accumulator.print_metrics()

if args.save_metrics:
    accumulator.save_metrics(save_path=output_path)
