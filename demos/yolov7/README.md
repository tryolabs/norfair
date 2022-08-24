# YOLOv7 Example

Simplest possible example of tracking. Based on [Yolov7](https://github.com/WongKinYiu/yolov7).

## Instructions

1. Build and run the Docker container with:
    ```bash
        ./run_docker.sh
    ``` 

1. In the container, display the demo instructions: 
    ```bash
        python demo.py --help 
    ``` 
   Bonus: Use additional arguments `--detector-path`, `--img-size`, `--iou-threshold`,`--conf-threshold`, `--classes`, `--track-points` as you wish.


## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by Yolov7.

![Norfair Yolov7 demo](../../docs/yolov7_cars.gif)
