# YOLOv4 example

Simplest possible example of tracking. Based on [pytorch YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tryolabs/norfair/blob/master/demos/yolov4/yolov4_demo.ipynb)

## Instructions

Build and run the Docker container with `./run_gpu.sh`.

For more advanced use cases, you can manually run the Docker container and display the demo instructions using `python demo.py --help`.

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by YOLOv4.

![Norfair YOLOv4 demo](../../docs/yolov4_cars.gif)
