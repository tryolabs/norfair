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

This example tracks objects using a single or double point per detection: the centroid or the two corners of the bounding boxes around objects returned by YOLOv5.

## Tracking cars

https://user-images.githubusercontent.com/3588715/189129861-ad2e5eb9-e124-43d9-96f6-1685566639f9.mp4

## Tracking pedestrians

https://user-images.githubusercontent.com/3588715/189129865-e7c8ccea-b74a-4cd3-a21c-7dc5883c0e95.mp4

## Tracking all

https://user-images.githubusercontent.com/3588715/189129869-83ff94b9-efc9-4789-9d7c-4fd22495fde3.mp4
