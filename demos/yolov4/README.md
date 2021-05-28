# YOLOv4 example

Simplest possible example of tracking. Based on [pytorch YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master).

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. Clone [pytorch YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master) and download the [weights](https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ) published on the repo into your local clone of the repo.
3. Copy `yolov4demo.py` into your local clone of the repo and run `python yolov4demo.py <video file>`.

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by YOLOv4.

![Norfair YOLOv4 demo](../../docs/yolov4_cars.gif)
