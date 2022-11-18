import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils import (
    driving_area_mask,
    increment_path,
    lane_line_mask,
    letterbox,
    non_max_suppression,
    scale_coords,
    show_seg_result,
    split_for_trace_model,
    yolop_detections_to_norfair_detections,
)

import norfair
from norfair import Tracker, Video
from norfair.distances import iou


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source", type=str, help="Path to the input data"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolopv2.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.3, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    return parser


def detect():
    # setting and directories
    source, weights, save_txt, imgsz = (
        opt.source,
        opt.weights,
        opt.save_txt,
        opt.img_size,
    )
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Load model
    model = torch.jit.load(weights)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"  # half precision only supported on CUDA
    model = model.to(device)

    # Norfair Tracker init
    tracker = Tracker(
        distance_function="iou",
        distance_threshold=0.7,
    )

    if half:
        model.half()  # to FP16
    model.eval()

    # Run inference
    if device != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()

    video = Video(input_path=source)

    for frame in video:
        # Padded resize
        img0 = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, imgsz, stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        [pred, anchor_grid], seg, ll = model(img)

        pred = split_for_trace_model(pred, anchor_grid)

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Resize bbox to im0 size
        for det in pred:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            show_seg_result(img0, (da_seg_mask, ll_seg_mask), is_demo=True)

        # Track detections with Norfair
        # Transfrom YOLOP detections to Norfair detecions
        detections = yolop_detections_to_norfair_detections(pred)
        tracked_objects = tracker.update(detections=detections)
        norfair.draw_tracked_boxes(img0, tracked_objects)

        video.write(img0)

    print(f"\nDone. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    opt = make_parser().parse_args()
    print(opt)

    if not os.path.exists(opt.weights):
        os.system(
            f"wget https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt -O {opt.weights}"
        )

    with torch.no_grad():
        detect()
