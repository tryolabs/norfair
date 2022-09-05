import argparse
import sys

import numpy as np

import norfair
from norfair import Detection, Tracker, Video
from norfair.distances import create_keypoints_voting_distance

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
DETECTION_THRESHOLD = 0.1
DISTANCE_THRESHOLD = 0.4
INITIALIZATION_DELAY = 4
HIT_COUNTER_MAX = 30
POINTWISE_HIT_COUNTER_MAX = 10

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


if __name__ == "__main__":

    # CLI configuration
    parser = argparse.ArgumentParser(description="Track human poses in a video.")
    parser.add_argument("files", type=str, nargs="+", help="Video files to process")
    parser.add_argument(
        "--skip-frame", dest="skip_frame", type=int, default=1, help="Frame skip period"
    )
    parser.add_argument(
        "--select-gpu",
        dest="select_gpu",
        help="Number of the gpu that you want to use",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    # Process Videos
    detector = OpenposeDetector(args.select_gpu)
    datum = op.Datum()

    for input_path in args.files:
        print(f"Video: {input_path}")
        video = Video(input_path=input_path)
        KEYPOINT_DIST_THRESHOLD = video.input_height / 40

        tracker = Tracker(
            distance_function=create_keypoints_voting_distance(
                keypoint_distance_threshold=KEYPOINT_DIST_THRESHOLD,
                detection_threshold=DETECTION_THRESHOLD,
            ),
            distance_threshold=DISTANCE_THRESHOLD,
            detection_threshold=DETECTION_THRESHOLD,
            initialization_delay=INITIALIZATION_DELAY,
            hit_counter_max=HIT_COUNTER_MAX,
            pointwise_hit_counter_max=POINTWISE_HIT_COUNTER_MAX,
        )

        for i, frame in enumerate(video):
            if i % args.skip_frame == 0:
                datum.cvInputData = frame
                detector(op.VectorDatum([datum]))
                detected_poses = datum.poseKeypoints

                if detected_poses is None:
                    tracked_objects = tracker.update(period=args.skip_frame)
                    continue

                detections = (
                    []
                    if not detected_poses.any()
                    else [
                        Detection(p, scores=s)
                        for (p, s) in zip(
                            detected_poses[:, :, :2], detected_poses[:, :, 2]
                        )
                    ]
                )
                tracked_objects = tracker.update(
                    detections=detections, period=args.skip_frame
                )
                norfair.draw_points(frame, detections)
            else:
                tracked_objects = tracker.update(period=args.skip_frame)

            norfair.draw_tracked_objects(frame, tracked_objects)
            video.write(frame)
