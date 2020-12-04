import sys
import os.path
import numpy as np
import norfair
from norfair import Detection, Tracker, lib_metrics, video, drawing
import argparse

frame_skip_period = 1
detection_threshold = 0.01
distance_threshold = 0.4

parser = argparse.ArgumentParser(
    description="Generate trackers and compare them with groundtruth data"
)
parser.add_argument(
    "--make_video", action="store_true", help="To generate an output video"
)
parser.add_argument(
    "--save_pred",
    action="store_true",
    help="To generate a txt file with your predictions",
)
parser.add_argument(
    "--save_metrics",
    action="store_true",
    help="To generate a txt file with your metrics",
)
parser.add_argument(
        "files",
        type=str,
        nargs="+",
    help="files_to_process and optionally the output_path if any kind of output file (optional arguments) is specified",
    )
args = parser.parse_args()

make_video = args.make_video
save_pred = args.save_pred
save_metrics = args.save_metrics

if make_video or save_pred or save_metrics:
    save_path = args.files[-1]
    if not os.path.isdir(save_path):
        raise ValueError(
            "You must specify 'save_path' directory if you want to get an output file"
        )
    args.files.remove(save_path)

videos = args.files

Accumulator = lib_metrics.Accumulators()

# proccess every video
for testing_video in videos:
    input_path = os.path.dirname(testing_video)

    # Search vertical resolution in seqinfo.ini
    seqinfo_path = os.path.join(input_path, "seqinfo.ini")
    v_resolution = lib_metrics.search_value_on_document(seqinfo_path, "imHeight")

    def keypoints_distance(detected_pose, tracked_pose):
        ps = [1, 2, np.inf]
        distances = 0
        diagonal = 0

        hor_min_pt = min(detected_pose.points[:, 0])
        hor_max_pt = max(detected_pose.points[:, 0])
        ver_min_pt = min(detected_pose.points[:, 1])
        ver_max_pt = max(detected_pose.points[:, 1])

        # set keypoint_dist_threshold based on object size, and calculate
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
            v_resolution * (diagonal < v_resolution / 3) / 67
            + diagonal * (diagonal >= v_resolution / 3) / 17
        )

        match_num = np.count_nonzero(
            (distances < keypoint_dist_threshold)
            * (detected_pose.scores > detection_threshold)
            * (tracked_pose.last_detection.scores > detection_threshold)
        )
        return 1 / (1 + match_num)

    all_detections = lib_metrics.DetFromFile(input_path=input_path).ordered_by_frame

    # inirialize output text file
    if save_pred:
        Text_file = lib_metrics.TextFile(input_path=input_path, save_path=save_path)

    if make_video:  # initialize video()
        Video_file = video.VideoFromFrames(input_path=input_path, save_path=save_path)

    # initialize tracker
    tracker = Tracker(
        distance_function=keypoints_distance,
        distance_threshold=distance_threshold,
        detection_threshold=detection_threshold,
        point_transience=2,
    )

    # initialize accumulator for this video
    Accumulator.create_acc(input_path=input_path)

    for frame_number, detections in enumerate(all_detections):
        if frame_number % frame_skip_period == 0:
            tracked_objects = tracker.update(
                detections=detections, period=frame_skip_period
            )
        else:
            tracked_objects = tracker.update()

        # save new frame on output video file
        if make_video:
            frame = Video_file.get_frame()
            frame = drawing.draw_boxes(frame, detections=detections)
            frame = drawing.draw_tracked_boxes(frame=frame, objects=tracked_objects)
            Video_file.update(frame=frame)

        # update output text file
        if save_pred:
            Text_file.update_text_file(predictions=tracked_objects)

        Accumulator.update(predictions=tracked_objects)

Accumulator.compute_metrics()
Accumulator.print_metrics()

if save_metrics:
    Accumulator.save_metrics(save_path=save_path)
