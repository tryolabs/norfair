import sys
import os.path
import numpy as np
import norfair
from norfair import Detection, Tracker, metrics, video, drawing
import argparse

frame_skip_period = 1
detection_threshold = 0.01
distance_threshold = 0.4

parser = argparse.ArgumentParser(
    description="""
Evaluate a basic tracker on MOTChallenge data.
Display on terminal the MOTChallenge metrics results 
""", 
    #formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    "--make_video", action="store_true", help="""Generate an output video. 
    You need to have the following folder containing each frame of this video
    <file_to_proccess>/img1/"""
)
parser.add_argument(
    "--save_pred",
    action="store_true",
    help="Generate a txt file with your predictions",
)
parser.add_argument(
    "--save_metrics",
    action="store_true",
    help="Generate a txt file with your MOTChallenge metrics results",
)
parser.add_argument(
        "video_files",
        type=str,
        nargs="+",
    help="""
        Path to files you want to proccess.

        Be sure that for each path, you have the files:
        <path>/det/det.txt and 
        <path>/gt/gt.txt, containing your detections and ground truth data respectively""")

args = parser.parse_args()

output_path = "."

accumulator = metrics.Accumulators()

for testing_video in args.video_files:
    input_path = os.path.dirname(testing_video)

    # Search vertical resolution in seqinfo.ini
    seqinfo_path = os.path.join(input_path, "seqinfo.ini")
    info_file = metrics.InformationFile(file_path = seqinfo_path)
    vertical_resolution = info_file.search(variable_name = "imHeight")

    def keypoints_distance(detected_pose, tracked_pose):
        ps = [1, 2, np.inf]
        distances = 0
        diagonal = 0

        hor_min_pt = min(detected_pose.points[:, 0])
        hor_max_pt = max(detected_pose.points[:, 0])
        ver_min_pt = min(detected_pose.points[:, 1])
        ver_max_pt = max(detected_pose.points[:, 1])

        # Set keypoint_dist_threshold based on object size, and calculate
        # distance between detections and tracker estimations
        for p in ps:
            distances += np.linalg.norm(
                detected_pose.points - tracked_pose.estimate, ord=p, axis=1
            )
            diagonal += np.linalg.norm(
                [hor_max_pt - hor_min_pt, ver_max_pt - ver_min_pt], ord=p
            )

        distances = distances / len(ps)

        keypoint_dist_threshold = (
            vertical_resolution * (diagonal < vertical_resolution / 3) / 67
            + diagonal * (diagonal >= vertical_resolution / 3) / 17
        )

        match_num = np.count_nonzero(
            (distances < keypoint_dist_threshold)
            * (detected_pose.scores > detection_threshold)
            * (tracked_pose.last_detection.scores > detection_threshold)
        )
        return 1 / (1 + match_num)

    all_detections = metrics.get_detections(input_path=input_path, information_file = info_file)

    if args.save_pred:
        predictions_text_file = metrics.PredictionsTextFile(input_path=input_path, save_path=output_path, information_file = info_file)

    if args.make_video: 
        video_file = video.VideoFromFrames(input_path=input_path, save_path=output_path, information_file = info_file)

    tracker = Tracker(
        distance_function=keypoints_distance,
        distance_threshold=distance_threshold,
        detection_threshold=detection_threshold,
        point_transience=2,
    )

    # initialize accumulator for this video
    accumulator.create_accumulator(input_path=input_path, information_file = info_file)

    for frame_number, detections in enumerate(all_detections):
        if frame_number % frame_skip_period == 0:
            tracked_objects = tracker.update(
                detections=detections, period=frame_skip_period
            )
        else:
            tracked_objects = tracker.update()

        # save new frame on output video file
        if args.make_video:
            frame = video_file.get_frame()
            frame = drawing.draw_boxes(frame, detections=detections)
            frame = drawing.draw_tracked_boxes(frame=frame, objects=tracked_objects)
            video_file.update(frame=frame)

        # update output text file
        if args.save_pred:
            predictions_text_file.update_text_file(predictions=tracked_objects)

        accumulator.update(predictions=tracked_objects)

accumulator.compute_metrics()
accumulator.print_metrics()

if args.save_metrics:
    accumulator.save_metrics(save_path=output_path)
