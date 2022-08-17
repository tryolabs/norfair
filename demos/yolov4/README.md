# YOLOv4 example

Simplest possible example of tracking. Based on [pytorch YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tryolabs/norfair/blob/master/demos/yolov4/yolov4_demo.ipynb)

## Instructions


1. Build and run the docker container with:
    ```bash
        ./run_docker.sh
    ``` 

4. In the container, display the demo instructions: 
    ```bash
        python demo.py --help 
    ``` 
    In the container, use the `/demo` folder as a volume to share files with the container.
    ```bash
        python3 demo.py /demo/video.mp4 --output-path /demo/
    ``` 

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by YOLOv4.

![Norfair YOLOv4 demo](../../docs/yolov4_cars.gif)