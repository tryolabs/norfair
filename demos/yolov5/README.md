# YOLOv5 Example

Simplest possible example of tracking. Based on [YOLOv5](https://github.com/ultralytics/yolov5).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This example cat track objects using a single or double point per detection: the centroid or the two corners of the bounding boxes around objects returned by YOLOv5.

## Tracking cars

https://user-images.githubusercontent.com/3588715/189720485-31a6302c-d046-4f3c-ae01-5ae53b9fde35.mp4
