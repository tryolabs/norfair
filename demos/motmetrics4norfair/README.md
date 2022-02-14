# Compute MOTChallenge metrics

Demo on how to evaluate a Norfair tracker on the [MOTChallenge](https://motchallenge.net).

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

3. Clone this repo and go to `demos/motmetrics4norfair/`.
    ```bash
    git clone git@github.com:tryolabs/norfair.git
    cd norfair/demos/demos/motmetrics4norfair
    ```

4. Display the motmetrics4norfair instructions: 
    ```bash
        python motmetrics4norfair.py --help 
    ``` 

    or run the following for a quick test

    ```bash
    python motmetrics4norfair.py ../../train/
    ```

## Important consideration

Hyperparameters were tuned for reaching a high `MOTA` on this dataset. They may not be ideal for more general use cases, use the default hyperparameters for those. ID switches suffer specially due to this optimization. If you want to improve ID switches use a higher margin between `hit_inertia_min` and `hit_inertia_max`, or just use the default hyperparameters.
