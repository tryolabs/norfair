# Detectron2 example

Simplest possible example of tracking. Based on [Detectron2](https://github.com/facebookresearch/detectron2).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes returned by Detectron2.

https://user-images.githubusercontent.com/3588715/189415541-d0b8f963-b813-449c-8f69-3d0c52f488cf.mp4