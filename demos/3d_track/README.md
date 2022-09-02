# 3D-Tracking Example

3D-tracking example based on [MediaPipe Objectron](https://google.github.io/mediapipe/solutions/objectron.html).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py video.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`. For the demo, we are using [this footage](https://www.pexels.com/video/man-showing-the-steps-in-hip-hop-dancing-2795737/).

## Explanation

This example tracks objects using 3D bounding boxes.

https://user-images.githubusercontent.com/70915567/187967263-ad45c0c7-483f-4f15-b1b9-27678456080b.mp4
