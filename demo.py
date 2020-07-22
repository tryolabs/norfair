import norfair
from norfair import Detection, Tracker, Video

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

def convert_and_filter(detections):
    keypoints_to_track = [1, 8]  # We'll only track neck=1 and midhip=8
    poses = detections[:, keypoints_to_track, :2]
    scores = detections[:, keypoints_to_track, 2]
    return [Detection(p, scores=s) for (p, s) in zip(poses, scores) if np.any(s > 0)]

def keypoints_distance(detected_pose, tracked_pose):
    detected_points = detected_pose.points
    # Find min torax size
    torax_length_detected_person = np.linalg.norm(detected_points[0] - detected_points[1])
    estimated_pose = tracked_pose.estimate
    torax_length_estimated_person = np.linalg.norm(estimated_pose[0] - estimated_pose[1])
    min_torax_size = min(torax_length_estimated_person, torax_length_detected_person)

    # Keypoints distance in terms of torax size
    substraction = detected_points - estimated_pose
    dists_per_point = np.linalg.norm(substraction, axis=1)
    keypoints_distance = np.mean(dists_per_point) / min_torax_size

    return keypoints_distance

pose_detector = OpenposeDetector()
for v in [
        "/home/lalo/data/videos/in/peatonal_sarandi/hard_10s.mp4",
]:
    video = Video(input_path=v, output_path="/home/lalo/out")
    tracker = Tracker(distance_function=keypoints_distance)

    import random
    for i, frame in enumerate(video):
        detection_period = random.randint(1, 5)
        if i % detection_period == 0:
            detected_poses = pose_detector(frame)
            detections = convert_and_filter(detected_poses)
            estimates = tracker.update(detections=detections, period=detection_period)
        else:
            estimates = tracker.update()
        norfair.draw_points(frame, detections)
        norfair.draw_estimates(frame, estimates)
        video.write(frame)
        # video.show(frame, downsample_ratio=1)
