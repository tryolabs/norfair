# Track multiple classes

An example of how to use Norfair to track objects from multiple classes using both keypoints and bounding boxes. This example is based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) version 1.7 and [YOLOv5-pip](https://github.com/fcakyon/yolov5-pip).

## Instructions

1. Install YOLOv5 with `pip install yolov5`.
2. Install Norfair with `pip install norfair[video]`.
3. Install [OpenPose version 1.7](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.7.0). You can follow [this](./install_openpose.ipynb) instructions to install and compile OpenPose.
4. Download the [example video](https://user-images.githubusercontent.com/92468171/162247647-d4c13cdd-a127-455e-967f-531e24cf20cb.mp4) with `wget "https://user-images.githubusercontent.com/92468171/162247647-d4c13cdd-a127-455e-967f-531e24cf20cb.mp4" -O production_ID_4791196_10s.mp4`
5. Run `python keypoints_bounding_boxes_demo.py production_ID_4791196_10s.mp4 --classes 1 2 3 5 --track_points bbox --conf_thres 0.4`.

Alternatively the example can be executed entirely within `keypoints_bounding_boxes_demo.ipynb`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tryolabs/norfair/blob/demo-keypoints-bboxes/demos/keypoints_bounding_boxes/keypoints_bounding_boxes_demo.ipynb)

## Explanation

This example aims at showing the possibilities that Norfair offers to track objects from multiple classes using a single `Tracker` instance. Also the example shows Norfair's ability to draw keypoints and bounding boxes for different types of objects.

![keypoints_bounding_boxes_demo](../../docs/keypoints_bounding_boxes_demo.gif)