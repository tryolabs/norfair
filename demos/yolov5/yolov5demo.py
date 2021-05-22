import argparse

import numpy as np
import torch
import yolov5
from typing import Union, List, Optional

import norfair
from norfair import Detection, Tracker, Video

max_distance_between_points: int = 30


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        if device is None:
            device = "cuda:0" if torch.cuda_is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None
    ):

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        detections_as_xyxy = detections.xyxy[0]
        return detections_as_xyxy


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_centroid(yolo_box):
    x1 = yolo_box[0]
    y1 = yolo_box[1]
    x2 = yolo_box[2]
    y2 = yolo_box[3]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf_thres", type=float, default="0.25", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)

for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )

    for frame in video:
        detections = model(
            frame,
            conf_threshold=args.conf_thres,
            iou_threshold=args.conf_thres,
            image_size=args.img_size,
            classes=args.classes
        )
        detections = [
            Detection(
                get_centroid(list(map(int, detection[:4].tolist()))),
                data=detection
            )
            for detection in detections
        ]
        tracked_objects = tracker.update(detections=detections)
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        video.write(frame)
