# Moving Camera Demo

In this example, we show how to estimate the camera movement in Norfair.

What's the motivation for estimating camera movement?

- When the camera moves, the apparent movement of the objects can be quite erratic and confuse the tracker; by estimating the camera movement we can stabilize the objects and improve tracking.
- By estimating the position of objects in a fixed reference we can correctly calculate their trajectory. This can help you if you are trying to determine when objects enter a predefined zone in the scene or trying to draw their trajectory.

Keep in mind that the estimation of the camera movement works best with a static background. If the scene is too chaotic with a lot of movement, the estimation will lose accuracy. Nevertheless, even when the estimation is incorrect it will not hurt the tracking.

## Example 1: Translation

This method only works for camera pans and tilts.

<img src="/docs/img/pan_tilt.png" alt="Pan and tilt" width="350px">

For an example of results, see the following videos. On the left, the tracker lost the person 4 times (as seen by the increasing id, and the color of the bounding box changing). However, on the right the tracker is able to maintain the tracked object throughout the video:

https://user-images.githubusercontent.com/3588715/189200533-6fb3031f-8f03-4116-afc8-bd3ec4265119.mp4

> Videos generated using command `python demo.py --transformation none --draw-objects --track-boxes --id-size 1.8 --distance-threshold 200 --save <video>.mp4` and `python demo.py --transformation translation --fixed-camera-scale 2 --draw-objects --track-boxes --id-size 1.8 --distance-threshold 200 --save <video>.mp4`

## Example 2: Homographies

This method can work with any camera movement, including pan, tilt, rotation, movement in any direction, and zoom.

In the following video, the players are tracked and their trajectories are drawn, even as the camera moves:

https://user-images.githubusercontent.com/3588715/189200514-b1b25601-2b45-4d2f-9c2f-224b09c9b020.mp4

> Video generated using command `python demo.py --transformation homography --draw-paths --path-history 150 --distance-threshold 200 --track-boxes --max-points=900 --min-distance=14 --save --model yolov5x --hit-counter-max 3 <video>.mp4` on a snippet of this [video](https://www.youtube.com/watch?v=CGFgHjeEkbY&t=1200s).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.
