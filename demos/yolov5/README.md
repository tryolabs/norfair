# YOLOv5 Example

Simplest possible example of tracking. Based on [YOLOv5-pip](https://github.com/fcakyon/yolov5-pip).

## Instructions 

1. Install Norfair with `pip install norfair[video]`.
2. Install YOLOv5 with `pip install yolov5`.
3. Copy `yolov5demo.py` into your local and run `python yolov5demo.py <video file>`.
4. Bonus: Use additional arguments `--detector_path`, `--img_size`, `--iou_thres`,`--conf_thres`, `--classes`, `--track_points` as you wish.

## Explanation

This example tracks objects using a single or double point per detection: the centroid or the two corners of the bounding boxes around objects returned by YOLOv5.

## Car tracking demo:

<img src="https://github.com/fcakyon/public-files/raw/main/norfair/yolov5_cars.gif" width="800" >

## Pedestrian tracking demo:

<img src="https://github.com/fcakyon/public-files/raw/main/norfair/yolov5_pedestrian.gif" width="800" >

## All tracking demo:

<img src="https://github.com/fcakyon/public-files/raw/main/norfair/yolov5_all.gif" width="800" >