# YOLOv5 Example

Simplest possible example of tracking. Based on [Yolov7](https://github.com/WongKinYiu/yolov7).

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. Install Yolov7 with `pip install -r https://raw.githubusercontent.com/WongKinYiu/yolov7/main/requirements.txt`.
3. Run `python yolov7demo.py <video file>`.
4. Bonus: Use additional arguments `--detector_path`, `--img_size`, `--iou_thres`,`--conf_thres`, `--classes`, `--track_points` as you wish.

## Explanation

This example tracks objects using a single or double point per detection: the centroid or the two corners of the bounding boxes around objects returned by YOLOv5.


## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by YOLOv4.

![Norfair YOLOv4 demo](../../docs/yolov7_cars.gif)
