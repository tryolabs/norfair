import argparse
import sys
import os.path
import numpy as np
import norfair
import cv2
from norfair import Detection, Tracker, Video, lib_metrics
from rich.progress import Progress

#Flag to decide if making an output video or not
make_video_output = False

# Insert the path to your openpose instalation folder here
frame_skip_period = 1
detection_threshold = 0.01
distance_threshold = 0.4

# detection location
input_path = "/home/aguscas/MOT17/MOT17/train/"
save_path = "/home/aguscas/preds/"  # location to save predictions

# get the name of the testing videos to proccess from command line
args = sys.argv[1:]

# proccess every video
for testing_video in args:
    input_path_det = input_path + testing_video + "/det/det.txt"

    out_file_name = os.path.join(save_path + testing_video + ".txt")

    # Search vertical resolution in seqinfo.ini
    seqinfo_path = input_path + testing_video + "/seqinfo.ini"
    v_resolution = lib_metrics.search_value_on_document(seqinfo_path, "imHeight")

    # things that are only necessary if making output video
    if make_video_output:
        # search framerate in seqinfo.ini
        fps = lib_metrics.search_value_on_document(seqinfo_path, "frameRate")

        # Search horizontal reolution in seqinfo.ini
        h_resolution = lib_metrics.search_value_on_document(seqinfo_path, "imWidth")
        image_size = (h_resolution, v_resolution)

        video_path = save_path + "videos/" + testing_video + ".mp4"  # video file name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(video_path, fourcc, fps, image_size)  # video file

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

    matrix_det = np.loadtxt(input_path_det, dtype="f", delimiter=",")

    row_order = np.argsort(matrix_det[:, 0])

    matrix_det = matrix_det[row_order]

    # detections coordinates refer to box corners
    matrix_det[:, 4] = matrix_det[:, 2] + matrix_det[:, 4]
    matrix_det[:, 5] = matrix_det[:, 3] + matrix_det[:, 5]

    # frames with detections
    first_frame = int(matrix_det[0, 0])  # first frame number
    last_frame = int(matrix_det[len(row_order) - 1, 0])  # last frame number

    actual_frame = first_frame  # actual frame number
    i = 1  # index that select row number on matrix_det

    # create file in which predictions will be saved
    out_file = open(out_file_name, "w+")

    tracker = Tracker(
        distance_function=keypoints_distance,
        distance_threshold=distance_threshold,
        detection_threshold=detection_threshold,
        point_transience=2,
    )

    with Progress() as progress:
        task = progress.add_task("[red]" + testing_video, total=last_frame)
        while actual_frame <= last_frame:
            progress.update(task, advance=1)
            if actual_frame % frame_skip_period == 0:
                actual_det = []
                while (i < len(row_order)) & (matrix_det[i - 1, 0] == actual_frame):
                    actual_det.append(matrix_det[i, :])
                    i += 1
                actual_det = np.array(actual_det)
                detections = []
                if actual_det.shape[0] > 0:
                    for j in range(actual_det.shape[0]):
                        points = [actual_det[j, [2, 3]], actual_det[j, [4, 5]]]
                        points = np.array(points)
                        conf = actual_det[j, 6]
                        new_detection = Detection(
                            points, np.array([1, 1])
                        )  # set to [1,1] or [conf,conf]
                        detections.append(new_detection)

                tracked_objects = tracker.update(
                    detections=detections, period=frame_skip_period
                )
            else:
                tracked_objects = tracker.update()

            # save new frame on output video file
            if make_video_output:
                frame_location = (
                    input_path
                    + testing_video
                    + "/img1/"
                    + str(actual_frame).zfill(6)
                    + ".jpg"
                )
                lib_metrics.write_video(
                    out_video, frame_location, detections, tracked_objects
                )

            lib_metrics.write_predictions(
                frame_number=actual_frame, objects=tracked_objects, out_file=out_file
            )
            actual_frame += 1

    if make_video_output:
        cv2.destroyAllWindows()
        out_video.release()

    out_file.close()
