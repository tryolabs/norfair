import os.path
import sys
import warnings

# Check for NumPy 2.0+ compatibility
import numpy as np

if np.__version__.split(".")[0] >= "2":
    warnings.warn(
        "MOT metrics tests are not compatible with NumPy 2.0+. "
        "Skipping these tests. To run MOT metrics tests, use NumPy < 2.0."
    )
    sys.exit(0)

import pandas as pd
from norfair import FilterPyKalmanFilterFactory, Tracker, metrics

DATASET_PATH = "train/train"
MOTA_ERROR_THRESHOLD = 0.0

DETECTION_THRESHOLD = 0.01
DISTANCE_THRESHOLD = 0.9
POINTWISE_HIT_COUNTER_MAX = 3
HIT_COUNTER_MAX = 2


def mot_metrics():
    """Tests that Norfair's MOT metrics didn't get worse

    Configurable so that it allows some margin on how much worse metrics could get before
    the test fails. Margin configured through MOTA_ERROR_THRESHOLD.

    Raises:
        If the previous metrics file its not found.
    """
    # Load previous metrics
    try:
        metrics_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "metrics.txt"
        )
        previous_metrics = pd.read_fwf(metrics_path)
        previous_metrics.columns = [
            column_name.lower() for column_name in previous_metrics.columns
        ]
        previous_metrics = previous_metrics.set_index(previous_metrics.columns[0])
    except FileNotFoundError as e:
        raise e

    accumulator = metrics.Accumulators()
    sequences_paths = [
        element.path for element in os.scandir(DATASET_PATH) if element.is_dir()
    ]
    for input_path in sequences_paths:
        # Search vertical resolution in seqinfo.ini
        seqinfo_path = os.path.join(input_path, "seqinfo.ini")
        info_file = metrics.InformationFile(file_path=seqinfo_path)

        all_detections = metrics.DetectionFileParser(
            input_path=input_path, information_file=info_file
        )

        tracker = Tracker(
            distance_function="iou",
            distance_threshold=DISTANCE_THRESHOLD,
            detection_threshold=DETECTION_THRESHOLD,
            pointwise_hit_counter_max=POINTWISE_HIT_COUNTER_MAX,
            hit_counter_max=HIT_COUNTER_MAX,
            filter_factory=FilterPyKalmanFilterFactory(),
        )

        # Initialize accumulator for this video
        accumulator.create_accumulator(
            input_path=input_path, information_file=info_file
        )

        for detections in all_detections:
            tracked_objects = tracker.update(
                detections=detections,
            )

            accumulator.update(predictions=tracked_objects)

    accumulator.compute_metrics()
    new_metrics = accumulator.summary_dataframe
    new_metrics.columns = [column_name.lower() for column_name in new_metrics.columns]

    # Unify the scores to be able to compare them. new metrics is the percentage
    # expressed between 0 and 1, the previous metrics have the percentage as a string
    # with the % character at the end
    new_overall_mota = np.around(new_metrics.loc["OVERALL", "mota"] * 100, 1)
    previous_overall_mota = np.around(
        float(previous_metrics.loc["OVERALL", "mota"][:-1]), 1
    )

    accumulator.print_metrics()
    assert new_overall_mota >= previous_overall_mota * (
        1 - MOTA_ERROR_THRESHOLD
    ), f"New overall MOTA score: {new_overall_mota} is too low, previous overall MOTA score: {previous_overall_mota}"


if __name__ == "__main__":
    mot_metrics()
