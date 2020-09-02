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
    if detections.shape[0] > 0:
        # keypoints_to_track = [1, 8]  # We'll only track neck=1 and midhip=8
        poses = detections[:, :, :2]
        scores = detections[:, :, 2]
        return [Detection(p, scores=s) for (p, s) in zip(poses, scores) if p[[1, 8]].any()]

def keypoints_distance(detected_pose, tracked_pose):
    detected_points = detected_pose.points
    # Find min torax size
    torax_length_detected_person = np.linalg.norm(detected_points[1] - detected_points[8])
    estimated_pose = tracked_pose.estimate
    torax_length_estimated_person = np.linalg.norm(estimated_pose[1] - estimated_pose[8])
    min_torax_size = min(torax_length_estimated_person, torax_length_detected_person)

    # Keypoints distance in terms of torax size
    substraction = detected_points[[1, 8], :] - estimated_pose[[1, 8], :]
    dists_per_point = np.linalg.norm(substraction, axis=1)
    keypoints_distance = np.mean(dists_per_point) / min_torax_size

    return keypoints_distance

import random
detection_period = 1 # random.randint(1, 5)
pose_detector = OpenposeDetector()
for v in [
        "/home/lalo/data/videos/in/peatonal_sarandi/hard_10s.mp4",
        "/home/lalo/data/facial_masks_detection/short.mp4",
        "/home/lalo/data/videos/in/cu.mp4",
        "/home/lalo/data/videos/in/trr/trr_cut_short.mp4"
]:
    video = Video(input_path=v, output_path="/home/lalo/data/videos/out/norfair/")
    tracker = Tracker(distance_function=keypoints_distance, distance_threshold=1, detection_threshold=0.5)

    for i, frame in enumerate(video):
        if i % detection_period == 0:
            detected_poses = pose_detector(frame)
            detections = convert_and_filter(detected_poses)
            tracked_objects = tracker.update(detections=detections, period=detection_period)
            # norfair.draw_points(frame, detections)
        else:
            tracked_objects = tracker.update()
        norfair.draw_tracked_objects(frame, tracked_objects)
        # norfair.draw_debug_metrics(frame, tracker.tracked_objects)
        video.write(frame)
        # for o in tracker.tracked_objects:
        #     # if o.id == 29:
        #     if o.initializing_id == 109:
        #         video.show(frame, downsample_ratio=1)
        #         print()
        #         print(o.last_detection.points[10], o.last_detection.scores[10])
        #         print(o.estimate[10], o.live_points[10])
        #         # print(o.live_points)
        #         import ipdb; ipdb.set_trace()
