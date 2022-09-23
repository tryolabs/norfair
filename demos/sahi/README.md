# SAHI: Slicing Aided Hyper Inference Demo

An example of how to use Norfair along with [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi )
to perform detection and tracking on small objects.

Comparison of not using SAHI (left) vs using it (right)

https://user-images.githubusercontent.com/31422367/192024585-b2162e7d-8bb1-4b74-a439-907f61301f2d.mp4

## Instructions

1. Build and run the Docker container with `./run.sh`
2. In the container, run the demo with `python demo.py <video>.mp4`.

   This will generate an output video, `output.mp4` with the result of the tracking using SAHI.

