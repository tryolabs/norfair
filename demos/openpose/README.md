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

In this example, we are skipping 4 out of every 5 frames, which should make processing the video 5 times faster. This is because the time added by running the Norfair itself is negligible when compared to not having to run 4 inferences on a deep neural network.

This is what the results look like (original videos can be found at [Kaggle](https://www.kaggle.com/datasets/ashayajbani/oxford-town-centre?select=TownCentreXVID.mp4)):

![openposev17_1_skip_5_frames_short](https://user-images.githubusercontent.com/92468171/172702968-ae986ecc-9cfd-4cd2-9132-92c19ff36608.gif)

![openposev17_2_skip_5_frames_short](https://user-images.githubusercontent.com/92468171/172703046-e769a9fa-4c0e-4111-9478-eb2d8ad2cead.gif)
