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

def keypoints_distance(detected_pose, tracked_pose):
    distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)
    match_num = np.count_nonzero((distances < keypoint_dist_threshold) * (detected_pose.scores > 0.2))
    distance = 1 / (1 + match_num)
    return distance

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
    keypoint_dist_threshold = video.input_height / 40
    tracker = Tracker(distance_function=keypoints_distance, distance_threshold=0.3,
                      detection_threshold=0.3)

    for i, frame in enumerate(video):
        if i % detection_period == 0:
            detected_poses = pose_detector(frame)
            detections = [
                Detection(p, scores=s)
                for (p, s) in zip(detected_poses[:, :, :2], detected_poses[:, :, 2])
                # if p[[1, 8]].any()
            ]
            tracked_objects = tracker.update(detections=detections, period=detection_period)
            # norfair.draw_points(frame, detections)
        else:
            tracked_objects = tracker.update()
        norfair.draw_tracked_objects(frame, tracked_objects)
        # norfair.draw_debug_metrics(frame, tracker.tracked_objects, score_threshold=0.3)
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
