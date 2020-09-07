# OpenPose frame detection interpolation demo

Demo for interpolating frames with no detections. Based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) version 1.4.

## Instructions

1. [Follow the instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md) to install OpenPose.
2. Run `openpose_interpolation.py`.

## Explanation

If you just want to accelerate your application such as pose detection, you can setup your detector to skip some frames, and let Norfair interpolate the detections through the rest of the frames.

In this example, we are skipping 2 out of every 3 frames, which should make your video process 3 times faster. This is because the time added by running the tracker itself is negligible when compared to not having to run 2 inferences on a deep neural network.

This is how the results look like:

![openpose_skip_3_frames](../../docs/openpose_skip_3_frames.gif)
