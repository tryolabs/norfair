import norfair
import numpy as np
import yaml  # For openpose
import sys  # For openpose

class OpenposeDetector():
    def __init__(self):
        with open("./open_pose_config.yml", 'r') as stream:
            open_pose_config = yaml.safe_load(stream)["openpose"]
        openpose_dir = open_pose_config['dir']
        sys.path.append(openpose_dir + "/build/python/openpose")
        from openpose import OpenPose  # noqa
        open_pose_config["default_model_folder"] = openpose_dir + "/models/"
        self.detector = OpenPose(open_pose_config)

    def __call__(self, image):
        return self.detector.forward(image, False)

def convert_and_filter(poses):
    # Get final poses by filtering out parts of the detections we don't want to track
    poses = poses[:, [1, 8]]  # We'll only track neck(1) and midhip(8)
    # Create filter for objects for which we haven't detected any of the parts we want to track
    poses = poses[np.any(poses > 0, axis=(1, 2)), :, :]
    return [p for p in poses]

def keypoints_distance(detected_pose, person):
    detected_points = detected_pose[:, :2]  # We ignore confidence score for matching
    # Find min torax size
    torax_length_detected_person = np.linalg.norm(detected_points[0] - detected_points[1])
    estimated_pose = person.estimate
    torax_length_estimated_person = np.linalg.norm(estimated_pose[0] - estimated_pose[1])
    min_torax_size = min(torax_length_estimated_person, torax_length_detected_person)

    # Keypoints distance in terms of torax size
    substraction = detected_points - estimated_pose
    dists_per_point = np.linalg.norm(substraction, axis=1)
    keypoints_distance = np.mean(dists_per_point) / min_torax_size

    return keypoints_distance

video = norfair.Video(input_path="/home/lalo/data/videos/in/peatonal_sarandi/hard_10s.mp4")
pose_detector = OpenposeDetector()
tracker = norfair.Tracker(distance_function=keypoints_distance)

for frame in video:
    detected_poses = pose_detector(frame)
    converted_detections = convert_and_filter(detected_poses)
    estimates = tracker.update(converted_detections, dt=1)
    norfair.draw_points(frame, converted_detections)
    norfair.draw_estimates(frame, estimates)
    video.write(frame)
    # video.show(frame, downsample_ratio=1)
