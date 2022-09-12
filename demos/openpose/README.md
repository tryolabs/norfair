# OpenPose example

Demo for extrapolating detections through skipped frames. Based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) version 1.7.

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

If you just want to speed up inference on a detector, you can make your detector skip frames, and use Norfair to extrapolate the detections through these skipped frames.

In this example, we are skipping 1 out of every 2 frames, which should make processing the video 2 times faster. This is because the time added by running the Norfair itself is negligible when compared to not having to run 4 inferences on a deep neural network.

This is what the results look like (original videos can be found at [Kaggle](https://www.kaggle.com/datasets/ashayajbani/oxford-town-centre?select=TownCentreXVID.mp4)):

https://user-images.githubusercontent.com/3588715/189412164-9685072c-24d0-431e-a622-2db6d410e174.mp4