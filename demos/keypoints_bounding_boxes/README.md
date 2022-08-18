# Track multiple classes

An example of how to use Norfair to track objects from multiple classes using both keypoints and bounding boxes. This example is based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) version 1.7 and [YOLOv5-pip](https://github.com/fcakyon/yolov5-pip).

## Instructions


1. Build and run the Docker container with:
    ```bash
        ./run_docker.sh
    ``` 

4. In the container, display the demo instructions: 
    ```bash
        python demo.py --help 
    ``` 

Alternatively the example can be executed entirely within `keypoints_bounding_boxes_demo.ipynb`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tryolabs/norfair/blob/demo-keypoints-bboxes/demos/keypoints_bounding_boxes/keypoints_bounding_boxes_demo.ipynb)


## Explanation

This example aims at showing the possibilities that Norfair offers to track objects from multiple classes using a single `Tracker` instance. Also the example shows Norfair's ability to draw keypoints and bounding boxes for different types of objects.

![keypoints_bounding_boxes_demo](../../docs/keypoints_bounding_boxes_demo.gif)