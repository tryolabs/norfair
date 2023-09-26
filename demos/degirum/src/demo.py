#!/usr/bin/env python3

"""Demo application using DeGirum with Norfair"""

import argparse
import numpy as np
import degirum as dg

import norfair
from norfair import Detection, Tracker, Video

parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument(
    "input", type=str, nargs='?', help="Video file to process"
)
parser.add_argument(
    "--camera", type=int, help="Camera ID"
)
parser.add_argument(
    "--show", action='store_true', help="Display in real-time"
)
parser.add_argument(
    "--zoo-url", type=str, required=True, help="DeGirum Cloud Platform Model Zoo URL"
)
parser.add_argument(
    "--token", type=str, required=True, help="Token for DeGirum cloud platform"
)
parser.add_argument(
    "--device", type=str, default='cloud', help="'local' 'cloud' or device hostname"
)
parser.add_argument(
    "--model", type=str, default="yolo_v5s_coco--512x512_quant_n2x_orca_1", help="Model"
)
parser.add_argument(
    "--conf-threshold", type=float, default="0.25", help="Object confidence threshold"
)
parser.add_argument(
    "--iou-threshold", type=float, default="0.45", help="IOU threshold for NMS"
)
parser.add_argument(
    "--classes", nargs="+", type=str, help="Filter by class label(s)"
)
args = parser.parse_args()

device = {"local": dg.LOCAL, "cloud": dg.CLOUD}.get(args.device, args.device)
zoo = dg.connect(device, args.zoo_url, args.token)
model = zoo.load_model(args.model)
model.output_conf_threshold = args.conf_threshold
model.output_nms_threshold = args.iou_threshold

if args.camera is None and args.input is None:
    raise RuntimeError("Either camera id or input path is required")
if args.camera is not None and args.input is not None:
    raise RuntimeError("Only one of camera id or input path must be specified")

if args.camera is not None:
    video = Video(camera=args.camera)
else:
    video = Video(input_path=args.input)

tracker = Tracker(
    distance_function="iou",
    distance_threshold=0.7
)

for output in model.predict_batch(video):
    detections = [
      Detection(
        points=np.array(det["bbox"]).reshape((2, 2)),
        scores=np.array([det["score"]] * 2),
        label=det["label"]
      ) for det in output.results if (not args.classes or det["label"] in args.classes)
    ]
    tracked_objects = tracker.update(detections=detections)
    frame = output.image.copy()  # Results contain corresponding input frame
    norfair.draw_boxes(frame, detections)
    norfair.draw_points(frame, tracked_objects)
    if args.input is not None:
        video.write(frame)
    if args.camera is not None or args.show:
        if video.show(frame) == ord('q'):
            break
