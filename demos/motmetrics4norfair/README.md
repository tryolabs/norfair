# Compute MOTChallenge metrics

Demos on how to evaluate different trackers on the [MOTChallenge](https://motchallenge.net). 

What does each script do?

1. `motmetrics4norfair.py` evaluates a Norfair tracker.
2. `motmetrics4norfair_xyah.py` is similar to `motmetrics4norfair.py`, but using the coordinates of `[center_x, center_y, asp_ratio, height]` and its velocities for the state vector of the Kalman Filter, as it is done in [ByteTrack's Kalman Filter](https://github.com/ifzhang/ByteTrack/blob/d742a3321c14a7412f024f2218142c7441c1b699/yolox/tracker/kalman_filter.py#L23).
3. `motmetrics4bytetrack.py` evaluates a [ByteTrack](https://github.com/ifzhang/ByteTrack) tracker.

## Instructions

1. Install Norfair with `pip install norfair[metrics,video]`.
2. Download the [MOT17](https://motchallenge.net/data/MOT17/) dataset running:

    ```bash
    curl -O https://motchallenge.net/data/MOT17Labels.zip  # To download Detections + Ground Truth (9.7 MB)
    unzip MOT17Labels.zip
    ```

    or the following if you want to download the images as well (much larger download):

    ```bash
    curl -O https://motchallenge.net/data/MOT17.zip  # To download Detections + Ground Truth + Images (5.5GB)
    unzip MOT17.zip
    ```

    Given that the ground truth files for the testing set are not publicly available, you will only be able to use motmetrics4norfair with the training set.

3. Clone repos and go to `demos/motmetrics4norfair/`.
    ```bash
    git clone git@github.com:tryolabs/norfair.git
    # Optional: Clone ByteTrack repo and follow its install instructions
    cd norfair/demos/motmetrics4norfair
    ```

4. Display the motmetrics4norfair instructions (replace `motmetrics4norfair.py` with `motmetrics4norfair_xyah.py` or `motmetrics4bytetrack.py` as needed):
    ```bash
        python motmetrics4norfair.py --help
    ``` 

    or run the following for a quick test

    ```bash
    python motmetrics4norfair.py ../../train/
    ```

## Important consideration

Hyperparameters were tuned for reaching a high `MOTA` on this dataset. They may not be ideal for more general use cases, use the default hyperparameters for those. ID switches suffer specially due to this optimization. If you want to improve ID switches use a higher `hit_counter_max`, or just use the default hyperparameters.
