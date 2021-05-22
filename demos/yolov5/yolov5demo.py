import argparse

import cv2
import numpy as np
import torch
import yolov5
from typing import Union

import norfair
from norfair import Detection, Tracker, Video

max_distance_between_points: int = 30


class YOLO:
    def __init__(self, model_path: str, use_cuda: bool = True):
        if use_cuda and not torch.cuda.is_available():
            raise Exception(
                "Selected use_cuda=True, but cuda is not available to Pytorch"
            )
        self.use_cuda = use_cuda
        self.model = yolov5.load(model_path, device="cuda" if use_cuda else "cpu")

        if self.use_cuda:
            self.model.cuda()

    def __call__(self, img: Union[str, np.ndarray], conf_threshold: float  = 0.25, iou_threshold: float = 0.45):
        width, height = 416, 416
        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        detections_temp = self.model(img)
        detections: list = []
        for detection in detections_temp.xyxy[0]:
            detections.append(detection.tolist())
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_centroid(yolo_box, img_height, img_width):
    x1 = yolo_box[0] * img_width
    y1 = yolo_box[1] * img_height
    x2 = yolo_box[2] * img_width
    y2 = yolo_box[3] * img_height
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


parser = argparse.ArgumentParser(description="Track human poses in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
args = parser.parse_args()

model = YOLO("yolov5m6.pt")  # set use_cuda=False if using CPU

for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )

    for frame in video:
        detections = model(frame)
        detections = [
            Detection(get_centroid(box, frame.shape[0], frame.shape[1]), data=box)
            for box in detections
            if box[-1] == 2
        ]
        tracked_objects = tracker.update(detections=detections)
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        video.write(frame)
