import argparse
import sys

import numpy as np

import norfair
from norfair import Detection, Tracker, Video

# Import openpose
openpose_install_path = (
    "/openpose"  # Insert the path to your openpose instalation folder here
)
try:
    sys.path.append(openpose_install_path + "/build/python")
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?"
    )
    raise e


# Define constants
DETECTION_THRESHOLD = 0.01
DISTANCE_THRESHOLD = 0.4
HIT_INERTIA_MIN = 5
INITIALIZATION_DELAY = 0
POINT_TRANSIENCE = 2

# Wrapper implementation for OpenPose detector
class OpenposeDetector:
    def __init__(self, num_gpu_start=None):
        # Set OpenPose flags
        config = {}
        config["model_folder"] = openpose_install_path + "/models/"
        config["model_pose"] = "BODY_25"
        config["logging_level"] = 3
        config["output_resolution"] = "-1x-1"
        config["net_resolution"] = "-1x768"
        config["num_gpu"] = 1
        config["alpha_pose"] = 0.6
        config["render_threshold"] = 0.05
        config["scale_number"] = 1
        config["scale_gap"] = 0.3
        config["disable_blending"] = False

        # If GPU version is built, and multiple GPUs are available,
        # you can change the ID using the num_gpu_start parameter
        if num_gpu_start is not None:
            config["num_gpu_start"] = num_gpu_start

        # Starting OpenPose
        self.detector = op.WrapperPython()
        self.detector.configure(config)
        self.detector.start()

    def __call__(self, image):
        return self.detector.emplaceAndPop(image)


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
