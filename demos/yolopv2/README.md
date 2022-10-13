# YOLOPv2 Example

Simplest possible example of tracking generic objects. Based on [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This demo uses YOLOPv2 capabilities to object detection, area segmentation, and line detection. Norfair is used to track object detections.

https://user-images.githubusercontent.com/67343574/195704838-eee83fd3-652b-4b27-a670-6e7929d64c00.mp4
