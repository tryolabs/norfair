# YOLOv5 Example

Simplest possible example of tracking. Based on [YOLOv5-pip](https://github.com/fcakyon/yolov5-pip).

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. Install YOLOv5 with `pip install yolov5`.
3. Run `python yolov5demo.py <video file>`.
4. Bonus: Use additional arguments `--detector_path`, `--img_size`, `--iou_thres`,`--conf_thres`, `--classes`, `--track_points` as you wish.

## Explanation

This example tracks objects using a single or double point per detection: the centroid or the two corners of the bounding boxes around objects returned by YOLOv5.

## Tracking cars

![Norfair tracking cars using YOLOv5](../../docs/yolov5_cars.gif)

## Tracking pedestrians

![Norfair tracking pedestrians using YOLOv5](../../docs/yolov5_pedestrian.gif)

## Tracking all

![Norfair tracking cars and pedestrians using YOLOv5](../../docs/yolov5_all.gif)
