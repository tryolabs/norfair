import sys
import os.path
import numpy as np
import norfair
from norfair import Detection, Tracker, Video, lib_metrics, video

# get arguments from terminal
(
    videos,
    input_path,
    save_path,
    make_video,
    show_metrics,
    save_pred,
) = lib_metrics.get_arguments()

frame_skip_period = 1
detection_threshold = 0.01
distance_threshold = 0.4

Accumulator = lib_metrics.accumulators(input_path=input_path)

# proccess every video
for testing_video in videos:
    input_path_det = os.path.join(input_path, testing_video, "det/det.txt")

    # Search vertical resolution in seqinfo.ini
    seqinfo_path = os.path.join(input_path, testing_video, "seqinfo.ini")
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

    all_detections = lib_metrics.Det_from_file(
        input_path=input_path, file_name=testing_video
    )

    last_frame = int(
        all_detections.matrix_detections[
            all_detections.matrix_detections.shape[0] - 1, 0
        ]
    )  # last frame number

    # inirialize output text file
    if save_pred:
        Text_file = lib_metrics.text_file(
            input_path=input_path, save_path=save_path, file_name=testing_video
        )

    if make_video:  # initialize video()
        Video_file = video.video_from_frames(
            input_path=input_path, save_path=save_path, file_name=testing_video
        )

    # initialize tracker
    tracker = Tracker(
        distance_function=keypoints_distance,
        distance_threshold=distance_threshold,
        detection_threshold=detection_threshold,
        point_transience=2,
    )

    # initialize accumulator for this video
    Accumulator.create_acc(file_name=testing_video)

    for actual_frame in np.arange(1, last_frame + 1):
            detections = []  # initialize list with detections
            if actual_frame % frame_skip_period == 0:
            detections = all_detections.get_dets_from_frame(frame_number=actual_frame)
                tracked_objects = tracker.update(
                    detections=detections, period=frame_skip_period
                )
            else:
                tracked_objects = tracker.update()
            # save new frame on output video file
        if make_video:
            Video_file.update_video(detections=detections, predictions=tracked_objects)
        # update output text file
        if save_pred:
            Text_file.update_text_file(predictions=tracked_objects)

        Accumulator.update(predictions=tracked_objects)

    if make_video:
        Video_file.close_video()
    if save_pred:
        Text_file.close_file()

    Accumulator.end_acc()

Accumulator.compute_metrics(save_path=save_path)

if show_metrics:
    Accumulator.print_metrics()
