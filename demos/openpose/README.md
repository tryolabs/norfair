# Speed OpenPose inference using tracking

Demo for extrapolating detections through skipped frames. Based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) version 1.4.

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. Install [OpenPose version 1.4](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.4.0).
3. Run `python openpose_extrapolation.py`.

## Explanation

If you just want to speed up inference on a detector, you can make your detector skip frames, and use Norfair to extrapolate the detections through these skipped frames.

In this example, we are skipping 2 out of every 3 frames, which should make the video process 3 times faster. This is because the time added by running the Norfair itself is negligible when compared to not having to run 2 inferences on a deep neural network.

This is how the results look like:

![openpose_skip_3_frames](../../docs/openpose_skip_3_frames.gif)
