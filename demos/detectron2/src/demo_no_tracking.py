import argparse

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from norfair import Detection, Tracker, Video, draw_tracked_objects
from norfair.drawing import Color, draw_points

parser = argparse.ArgumentParser(description="Track centroid of vehicles in a video")
parser.add_argument("file", type=str, help="Input video file")
args = parser.parse_args()

# Set up Detectron2 object detector
cfg = get_cfg()
cfg.merge_from_file("./detectron2_config.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "/model/model_final_f10217.pkl"
detector = DefaultPredictor(cfg)

# Norfair
video = Video(input_path=args.file, output_path="traffic_no_tracking_out.mp4")

for frame in video:
    detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Wrap Detectron2 detections in Norfair's Detection objects
    detections = [
        Detection(p)
        for p, c in zip(
            detections["instances"].pred_boxes.get_centers().cpu().numpy(),
            detections["instances"].pred_classes,
        )
        # Restrict to cars only
        if c == 2
    ]
    draw_points(
        frame,
        detections,
        color=Color.blue,
        radius=int(frame.shape[0] * 0.01),
        thickness=int(frame.shape[0] * 0.01),
    )
    video.write(frame)
