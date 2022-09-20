# Track multiple classes

An example of how to use Norfair to track objects from multiple classes using both keypoints and bounding boxes. This example is based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [YOLOv5](https://github.com/ultralytics/yolov5).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This example aims at showing the possibilities that Norfair offers to track objects from multiple classes using a single `Tracker` instance. Also the example shows Norfair's ability to draw keypoints and bounding boxes for different types of objects.

https://user-images.githubusercontent.com/3588715/189385171-8e1d5a5d-3796-4408-8729-26103d906063.mp4