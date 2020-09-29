import argparse
import sys
import os.path
import numpy as np

import norfair
from norfair import Detection, Tracker, Video

# Insert the path to your openpose instalation folder here
frame_skip_period = 1
detection_threshold = 0.01
distance_threshold = 0.4

# detection location
input_path = "/home/aguscas/MOT17/MOT17/train/"
testing_video = "MOT17-13-FRCNN"
input_path_det = input_path + testing_video + "/det/det.txt"

save_path = "/home/aguscas/preds/"  # location to save predictions
out_file_name = os.path.join(save_path + testing_video + ".txt")


# Search vertical resolution in seqinfo.ini file
with open(input_path + testing_video + "/seqinfo.ini", "r") as myfile:
    seqinfo = myfile.read()
position = seqinfo.find("imHeight")
position = position + len("imHeight")
while not seqinfo[position].isdigit():
    position += 1
v_resolution_str = ""
while seqinfo[position].isdigit():
    v_resolution_str = v_resolution_str + seqinfo[position]
    position += 1
v_resolution = int(v_resolution_str)


def keypoints_distance(detected_pose, tracked_pose):
    ps = [np.inf]
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
        v_resolution * (diagonal < v_resolution / 3) / 50
        + diagonal * (diagonal >= v_resolution / 3) / 25
    )

    match_num = np.count_nonzero(
        (distances < keypoint_dist_threshold)
        # * (detected_pose.scores > detection_threshold)
        # * (tracked_pose.last_detection.scores > detection_threshold)
    )
    return 1 / (1 + match_num)


matrix_det = np.loadtxt(input_path_det, dtype="f", delimiter=",")

row_order = np.argsort(matrix_det[:, 0])

matrix_det = matrix_det[row_order]

# detections coordinates refer to box corners
matrix_det[:, 4] = matrix_det[:, 2] + matrix_det[:, 4]
matrix_det[:, 5] = matrix_det[:, 3] - matrix_det[:, 5]

# frames with detections
first_frame = int(matrix_det[0, 0])  # first frame number
last_frame = int(matrix_det[len(row_order) - 1, 0])  # last frame number

actual_frame = first_frame  # actual frame number
i = 1  # index that select row number on matrix_det

out_file = open(out_file_name, "w+")

tracker = Tracker(
    distance_function=keypoints_distance,
    distance_threshold=distance_threshold,
    detection_threshold=detection_threshold,
    point_transience=2,
)

while actual_frame <= last_frame:
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
                conf = actual_det[
                    j, 6
                ]  # set it to 1 or to detection confidence given on det.txt
                new_detection = Detection(points, np.array([conf, conf]))
                detections.append(new_detection)

        tracked_objects = tracker.update(
            detections=detections, period=frame_skip_period
        )
    else:
        tracked_objects = tracker.update()
    for t in range(len(tracked_objects)):
        frame_str = str(int(actual_frame))
        id_str = str(int(tracked_objects[t].id))
        bb_left_str = str((tracked_objects[t].estimate[0, 0]))
        bb_top_str = str((tracked_objects[t].estimate[0, 1]))
        bb_width_str = str(
            (tracked_objects[t].estimate[1, 0] - tracked_objects[t].estimate[0, 0])
        )
        bb_height_str = str(
            (tracked_objects[t].estimate[0, 1] - tracked_objects[t].estimate[1, 1])
        )
        row_text_out = (
            frame_str
            + ","
            + id_str
            + ","
            + bb_left_str
            + ","
            + bb_top_str
            + ","
            + bb_width_str
            + ","
            + bb_height_str
            + ",-1,-1,-1,-1"
        )
        out_file.write(row_text_out)
        out_file.write("\n")

    actual_frame += 1
out_file.close()
