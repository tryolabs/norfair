import sys

import numpy as np
import yaml

import norfair
from norfair import Detection, Tracker, Video

frame_skip_period = 3
detection_threshold = 0.2
distance_threshold = 0.3


class OpenposeDetector:
    def __init__(self):
        with open("./demos/openpose/openpose_config.yml", "r") as stream:
            open_pose_config = yaml.safe_load(stream)["openpose"]
        openpose_dir = open_pose_config["dir"]
        sys.path.append(openpose_dir + "/build/python/openpose")
        from openpose import OpenPose  # noqa

        open_pose_config["default_model_folder"] = openpose_dir + "/models/"
        self.detector = OpenPose(open_pose_config)

    def __call__(self, image):
        return self.detector.forward(image, False)


def keypoints_distance(detected_pose, tracked_pose):
    distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)
    match_num = np.count_nonzero(
        (distances < keypoint_dist_threshold)
        * (detected_pose.scores > detection_threshold)
        * (tracked_pose.last_detection.scores > detection_threshold)
    )
    return 1 / (1 + match_num)


pose_detector = OpenposeDetector()
video = Video(input_path="video.mp4")
tracker = Tracker(
    distance_function=keypoints_distance,
    distance_threshold=distance_threshold,
    detection_threshold=detection_threshold,
    point_transience=2,
)
keypoint_dist_threshold = video.input_height / 30

for i, frame in enumerate(video):
    if i % frame_skip_period == 0:
        detected_poses = pose_detector(frame)
        detections = (
            []
            if not detected_poses.any()
            else [
                Detection(p, scores=s)
                for (p, s) in zip(detected_poses[:, :, :2], detected_poses[:, :, 2])
            ]
        )
        tracked_objects = tracker.update(
            detections=detections, period=frame_skip_period
        )
        norfair.draw_points(frame, detections)
    else:
        tracked_objects = tracker.update()
    norfair.draw_tracked_objects(frame, tracked_objects)
    video.write(frame)
