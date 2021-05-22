# YOLOv4 example

Simplest possible example of tracking. Based on [YOLOv5-pip](https://github.com/fcakyon/yolov5-pip).

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. Install YOLOv5 with `pip install yolov5`.
3. Copy `yolov5demo.py` into your local clone of the repo and run `python yolov5demo.py <video file>`.

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by YOLOv5.

![Norfair YOLOv5 demo](../../docs/yolov5_cars.gif)
