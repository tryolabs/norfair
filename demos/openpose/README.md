# Speed OpenPose inference using tracking

Demo for extrapolating detections through skipped frames. Based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) version 1.7.

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. Install [OpenPose version 1.7](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.7.0). You can follow [these](./openpose_extrapolation.ipynb) instructions to install and compile OpenPose.
3. Run `python openpose_extrapolation.py <video file> --skip-frame 5`.
4. Use additional arguments `--skip-frame`, `--select-gpu` as you wish.

Alternatively the example can be executed entirely within `openpose_extrapolation.ipynb`.

## Explanation

If you just want to speed up inference on a detector, you can make your detector skip frames, and use Norfair to extrapolate the detections through these skipped frames.

In this example, we are skipping 4 out of every 5 frames, which should make the video process 5 times faster. This is because the time added by running the Norfair itself is negligible when compared to not having to run 4 inferences on a deep neural network.

This is how the results look like (original videos can be found at [Kaggle](https://www.kaggle.com/datasets/ashayajbani/oxford-town-centre?select=TownCentreXVID.mp4)):

![openposev17_1_skip_5_frames_short](https://user-images.githubusercontent.com/92468171/172702968-ae986ecc-9cfd-4cd2-9132-92c19ff36608.gif)

![openposev17_2_skip_5_frames_short](https://user-images.githubusercontent.com/92468171/172703046-e769a9fa-4c0e-4111-9478-eb2d8ad2cead.gif)
