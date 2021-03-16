# Compute MOTChallenge metrics

Demo on how to evaluate a Norfair tracker on the [MOTChallenge](https://motchallenge.net).

## Instructions

1. Install Norfair with `pip install norfair[metrics,video]`.
2. Download the [MOT17](https://motchallenge.net/data/MOT17/) dataset running:

    ```bash
    curl -O https://motchallenge.net/data/MOT17.zip  # To download Detections + Ground Truth + Images (5.5GB)
    unzip MOT17.zip
    ```

    or

    ```bash
    curl -O https://motchallenge.net/data/MOT17Labels.zip  # To download Detections + Ground Truth (9.7 MB)
    unzip MOT17Labels.zip
    ```

    Given that the ground truth files for the testing set are not publicly available, you will only be able to use motmetrics4norfair with the training set.

3. Display the motmetrics4norfair instructions: 
```bash
    python motmetrics4norfair.py --help 
``` 

## Remarks

Hyperparameters were tuned for reaching a high `MOTA` on this dataset. They may not be ideal for more general use cases, use the default hyperparameters for those. ID switches suffer specially due to this optimization. If you want to improve ID switches use a higher margin between `hit_inertia_min` and `hit_inertia_max`, or just use the default hyperparameters.
