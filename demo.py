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
    return poses[:, :, :2]  # Remove probabilities from keypoints

def keypoints_distance(detected_pose, person):
    # Find min torax size
    torax_length_detected_person = np.linalg.norm(detected_pose[0] - detected_pose[1])
    predicted_pose = person.prediction
    torax_length_predicted_person = np.linalg.norm(predicted_pose[0] - predicted_pose[1])
    min_torax_size = min(torax_length_predicted_person, torax_length_detected_person)

    # Keypoints distance in terms of torax size
    substraction = detected_pose - predicted_pose
    dists_per_point = np.linalg.norm(substraction, axis=1)
    keypoints_distance = np.mean(dists_per_point) / min_torax_size

    return keypoints_distance

video = norfair.Video(input_path="/home/lalo/data/facial_masks_detection/shortest.mp4")
pose_detector = OpenposeDetector()
tracker = norfair.Tracker(distance_function=keypoints_distance)

for frame in video:
    detected_poses = pose_detector(frame)
    converted_detections = convert_and_filter(detected_poses)
    predictions = tracker.update(converted_detections, dt=1)
    norfair.draw_points(frame, converted_detections)
    norfair.draw_predictions(frame, predictions)
    video.write(frame)
    # video.show(frame, downsample_ratio=1)
