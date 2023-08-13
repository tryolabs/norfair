import argparse
from typing import List, Optional, Union

import numpy as np
import torch

import norfair
from norfair import Detection, Paths, Tracker, Video

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000


import super_gradients


class YOLO_NAS:
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load model
        else:
            self.model = super_gradients.training.models.get(
                "yolo_nas_l", pretrained_weights="coco"
            ).cuda()

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        if classes is not None:
            self.model.classes = classes

        detections = self.model.predict(img, iou_threshold, conf_threshold)
        return detections


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )

    elif track_points == "bbox":

        ## yolo_nas detections
        detections_as_xyxy = yolo_detections[0]
        class_names = detections_as_xyxy.class_names
        labels = detections_as_xyxy.prediction.labels
        confidence = detections_as_xyxy.prediction.confidence
        bboxes = detections_as_xyxy.prediction.bboxes_xyxy

        for i, (label, conf, bbox_yolo) in enumerate(zip(labels, confidence, bboxes)):
            bbox = np.array(
                [
                    [bbox_yolo[0], bbox_yolo[1]],
                    [bbox_yolo[2], bbox_yolo[3]],
                ]
            )

            scores = np.array([conf, conf])
            norfair_detections.append(
                Detection(points=bbox, scores=scores, label=class_names[int(label)])
            )

        return norfair_detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument(
    "--model-name", type=str, default="yolovnas", help="YOLOv5 model name"
)
parser.add_argument(
    "--img-size", type=int, default="720", help="YOLO_nas inference size (pixels)"
)
parser.add_argument(
    "--conf-threshold",
    type=float,
    default="0.25",
    help="YOLOv5 object confidence threshold",
)
parser.add_argument(
    "--iou-threshold", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS"
)
parser.add_argument(
    "--classes",
    nargs="+",
    type=int,
    help="Filter by class: --classes 0, or --classes 0 2 3",
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Inference device: 'cpu' or 'cuda'"
)
parser.add_argument(
    "--track-points",
    type=str,
    default="bbox",
    help="Track points: 'centroid' or 'bbox'",
)
args = parser.parse_args()

model = YOLO_NAS(args.model_name, device=args.device)

for input_path in args.files:
    video = Video(input_path=input_path)
    distance_function = "iou" if args.track_points == "bbox" else "euclidean"
    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX
        if args.track_points == "bbox"
        else DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    for frame in video:
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            image_size=args.img_size,
        )

        detections = yolo_detections_to_norfair_detections(
            yolo_detections, track_points=args.track_points
        )
        tracked_objects = tracker.update(detections=detections)
        if args.track_points == "centroid":
            norfair.draw_points(frame, detections)
            norfair.draw_tracked_objects(frame, tracked_objects)
        elif args.track_points == "bbox":
            norfair.draw_boxes(frame, detections)
            norfair.draw_boxes(frame, tracked_objects, draw_ids=True)
        video.write(frame)
