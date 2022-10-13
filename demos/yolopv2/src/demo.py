import argparse
import os
import time
from pathlib import Path

import cv2
import torch

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils import (
    AverageMeter,
    LoadImages,
    driving_area_mask,
    increment_path,
    lane_line_mask,
    non_max_suppression,
    plot_one_box,
    scale_coords,
    select_device,
    show_seg_result,
    split_for_trace_model,
    time_synchronized,
    yolop_detections_to_norfair_detections,
)

import norfair
from norfair import Tracker
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
    save_img = not opt.nosave and not source.endswith(".txt")  # save inference images

    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA
    model = model.to(device)

    # Norfair Tracker init
    tracker = Tracker(
        distance_function=iou,
        distance_threshold=0.7,
    )

    if half:
        model.half()  # to FP16
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version
        # but this problem will not appear in offical version
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Data
        p, s, im0, _ = path, "", im0s, getattr(dataset, "frame", 0)

        # Track detections with Norfair
        # Resize bbox to im0 size
        for det in pred:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Transfrom YOLOP detections to Norfair detecions
        detections = yolop_detections_to_norfair_detections(pred)
        tracked_objects = tracker.update(detections=detections)
        norfair.draw_tracked_boxes(im0, tracked_objects)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            s += "%gx%g " % img.shape[2:]  # print string

            # Print time (inference)
            print(f"{s}Done. ({t2 - t1:.3f}s)")
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w, h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer.write(im0)

    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))
    print("inf : (%.4fs/frame)   nms : (%.4fs/frame)" % (inf_time.avg, nms_time.avg))
    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    opt = make_parser().parse_args()
    print(opt)

    if not os.path.exists(opt.weights):
        os.system(
            f"wget https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt -O {opt.weights}"
        )

    with torch.no_grad():
        detect()
