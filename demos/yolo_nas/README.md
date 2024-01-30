# YOLO nas example

Simplest possible example of tracking. Based on [YOLO-NAS-L](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes around cars returned by YOLO-NAS-L

https://github.com/agosl/norfair/assets/35232517/3faffb87-6d18-4bcd-9321-3742080ef2e4

