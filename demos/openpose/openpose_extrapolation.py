import argparse
import sys

import numpy as np

import norfair
from norfair import Detection, Tracker, Video

# Insert the path to your openpose instalation folder here
openpose_install_path = "openpose/openpose"
frame_skip_period = 3
detection_threshold = 0.01
distance_threshold = 0.4


class OpenposeDetector:
    def __init__(self):
        config = {}
        config["dir"] = openpose_install_path
        config["logging_level"] = 3
        config["output_resolution"] = "-1x-1"  # 320x176
        config["net_resolution"] = "-1x768"  # 320x176
        config["model_pose"] = "BODY_25"
        config["alpha_pose"] = 0.6
        config["scale_gap"] = 0.3
        config["scale_number"] = 1
        config["render_threshold"] = 0.05
        config[
            "num_gpu_start"
        ] = 0  # If GPU version is built, and multiple GPUs are available, set the ID here
        config["disable_blending"] = False
        openpose_dir = config["dir"]
        sys.path.append(openpose_dir + "/build/python/openpose")
        from openpose import OpenPose  # noqa

        config["default_model_folder"] = openpose_dir + "/models/"
        self.detector = OpenPose(config)

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
parser = argparse.ArgumentParser(description="Track human poses in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
args = parser.parse_args()

for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=keypoints_distance,
        distance_threshold=distance_threshold,
        detection_threshold=detection_threshold,
        pointwise_hit_counter_max=2,
    )
    keypoint_dist_threshold = video.input_height / 25

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
