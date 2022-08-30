# MMdetection example

Simplest possible example of tracking using [MMDetection](https://github.com/open-mmlab/mmdetection).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

    ```bash
    python demo.py <video>.mp4
    ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes returned by MMDetection.

![Norfair MMDetection demo](../../docs/traffic_mmdet.gif)
