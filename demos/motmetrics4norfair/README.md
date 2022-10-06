# Compute MOTChallenge metrics

Demos on how to evaluate different trackers on the [MOTChallenge](https://motchallenge.net).

What does each script do?

1. `motmetrics4norfair.py` evaluates a Norfair tracker.
2. `motmetrics4norfair_xyah.py` is similar to `motmetrics4norfair.py`, but using the coordinates of `[center_x, center_y, asp_ratio, height]` and its velocities for the state vector of the Kalman Filter, as it is done in [ByteTrack's Kalman Filter](https://github.com/ifzhang/ByteTrack/blob/d742a3321c14a7412f024f2218142c7441c1b699/yolox/tracker/kalman_filter.py#L23).
3. `motmetrics4bytetrack.py` evaluates a [ByteTrack](https://github.com/ifzhang/ByteTrack) tracker.

## Instructions

1. Build and run the Docker container with `./run.sh`.
2. Within the container, run `python motmetrics4norfair.py /MOT17/train/` to evaluate on `MOT17` dataset or  `python motmetrics4norfair.py /MOT20/train/` to evaluate on `MOT20` dataset.
3. For more advanced use cases, within the container you can display the motmetrics4norfair instructions (replace `motmetrics4norfair.py` with `motmetrics4norfair_xyah.py` or `motmetrics4bytetrack.py` as needed):

```bash
python motmetrics4norfair.py --help
```

or run the following for a quick test

```bash
python motmetrics4norfair.py /MOT17/train/
```

## Important consideration

Hyperparameters were tuned for reaching a high `MOTA` on this dataset. They may not be ideal for more general use cases. Id switches suffer especially due to this optimization. If you want to improve Id switches, use a higher `hit_counter_max`, or just use the default hyperparameters.
