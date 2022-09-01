# YOLOv4 example

Simplest possible example of tracking. Based on [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tryolabs/norfair/blob/master/demos/yolov4/yolov4_demo.ipynb)

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by YOLOv4.

![Norfair YOLOv4 demo](../../docs/yolov4_cars.gif)
