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

def convert_and_filter_poses(poses):
    # Get final poses by filtering out parts of the detections we don't want to track
    body_parts_to_track_indices = [1, 8]  # Neck and midhip (torax length)
    poses = poses[:, body_parts_to_track_indices]
    # Create filter for objects for which we haven't detected the parts we want to track
    # TODO: Check if this is necessary
    poses = poses[np.any(poses > 0, axis=(1, 2)), :, :]

    # Convert to list
    # TODO: Check if there is one-line way of doing this
    # Or if it is even necessary, maybe defining iterable as protocol is better than list
    detections = []
    for p in poses:
        detections.append(p[:, :2])  # Remove probabilities
    return detections

def keypoints_distance(detected_pose, person):
    # Find min torax size
    torax_length_detected_person = np.linalg.norm(detected_pose[0] - detected_pose[1])
    estimated_pose = person.estimate
    torax_length_estimated_person = np.linalg.norm(estimated_pose[0] - estimated_pose[1])
    min_torax_size = min(torax_length_estimated_person, torax_length_detected_person)

    # Keypoints distance in terms of torax size
    substraction = detected_pose - estimated_pose
    dists_per_point = np.linalg.norm(substraction, axis=1)
    keypoints_distance = np.mean(dists_per_point) / min_torax_size

    return keypoints_distance

video = norfair.Video(input_path="/home/lalo/data/videos/in/peatonal_sarandi/hard_10s.mp4")
pose_detector = OpenposeDetector()
tracker = norfair.Tracker(distance_function=keypoints_distance)

for frame in video:
    detected_poses = pose_detector(frame)
    converted_detections = convert_and_filter_poses(detected_poses)
    estimates = tracker.update(converted_detections, dt=3)
    # norfair.draw_midpoint(frame, estimates)
    # norfair.draw_pose(frame, converted_detections, colors.blue)
    video.write(frame)
    print(estimates)
    # video.show(frame, downsample_ratio=4)

