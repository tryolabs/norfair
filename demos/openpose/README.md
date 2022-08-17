# Openpose example

Demo for extrapolating detections through skipped frames. Based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) version 1.7.

## Instructions


1. Build and run the docker container with:
    ```bash
        ./run_docker.sh
    ``` 

4. In the container, display the demo instructions: 
    ```bash
        python demo.py --help 
    ``` 

## Explanation

If you just want to speed up inference on a detector, you can make your detector skip frames, and use Norfair to extrapolate the detections through these skipped frames.

In this example, we are skipping 4 out of every 5 frames, which should make the video process 5 times faster. This is because the time added by running the Norfair itself is negligible when compared to not having to run 4 inferences on a deep neural network.

This is how the results look like (original videos can be found at [Kaggle](https://www.kaggle.com/datasets/ashayajbani/oxford-town-centre?select=TownCentreXVID.mp4)):

![openposev17_1_skip_5_frames_short](https://user-images.githubusercontent.com/92468171/172702968-ae986ecc-9cfd-4cd2-9132-92c19ff36608.gif)

![openposev17_2_skip_5_frames_short](https://user-images.githubusercontent.com/92468171/172703046-e769a9fa-4c0e-4111-9478-eb2d8ad2cead.gif)
