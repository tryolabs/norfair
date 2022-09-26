# SAHI: Slicing Aided Hyper Inference Demo

An example of how to use Norfair along with [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
to perform detection and tracking on small objects using [YOLOv5x](https://github.com/ultralytics/yolov5) as the detector.

Comparison of not using SAHI (left) vs using it (right)

https://user-images.githubusercontent.com/31422367/192333533-e9d51791-3db6-44a4-8aa6-7a4f8e167b1d.mp4

## Instructions

1. Build and run the Docker container with `./run.sh`
2. In the container, run the demo with `python demo.py <video>.mp4`.

   This will generate an output video, `output.mp4` with the result of the tracking using SAHI.
