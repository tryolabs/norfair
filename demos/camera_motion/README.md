# Moving Camera Demo

In this example, we show how to estimate the camera movement in Norfair.

What's the motivation for estimating camera movement?

- When the camera moves, the apparent movement of the objects can be quite erratic and confuse the tracker; by estimating the camera movement we can stabilize the objects and improve tracking.
- By estimating the position of objects in a fixed reference we can correctly calculate their trajectory. This can help you if you are trying to determine when objects enter a predefined zone on the scene or trying to draw their trajectory

Keep in mind that for estimating the camera movement we rely on a static background, if the scene is too chaotic with a lot of movement the estimation will lose accuracy. Nevertheless, even when the estimation is incorrect it will not hurt the tracking.  

## First Example - Translation

This method only works for camera pans and tilts. 

![Pan and Tilt](/docs/pan_tilt.png)

The following video shows on the left we lost the person 2 times while on the right we were able to maintain the tracked object throughout the video:

![camera_stabilization](/docs/camera_stabilization.gif)

> videos generated using command `python demos/camera_motion/demo.py --transformation none --draw-boxes --downsample-ratio 2 --distance-threshold 200 --save vidoe.mp4` and `python demos/camera_motion/demo.py --transformation translation --fixed-camera-scale 2 --draw-boxes --downsample-ratio 2 --distance-threshold 200 --save vidoe.mp4`

## Second Example - Homographies

This method can work with any camera movement, this includes pan, tilt, rotation, traveling in any direction, and zoom.

In the following video, the correct trajectory of the players is drawn even as the camera moves:

![soccer](/docs/soccer.gif)

> video generated using command `python demos/camera_motion/demo.py --transformation homography --draw-paths --path-history 150 --downsample-ratio 2 --distance-threshold 200 --track-boxes --max-points=900 --min-distance=14 --save --model yolov5x --hit-counter-max 3 video.mp4` on a snippet of this [video](https://www.youtube.com/watch?v=CGFgHjeEkbY&t=1200s)


## Setup

1. Install norfair with `pip install norfair[video]`
2. Install YOLOv5 with `pip install yolov5`.
3. Download and run the [demo.py](./demo.py) on any video