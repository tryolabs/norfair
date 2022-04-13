import os.path

import numpy as np
import pandas as pd

from norfair import Tracker, metrics

DATASET_PATH = "train"
# Set the percentage that the overall MOTA can decrease without being an error
MOTA_ERROR_THRESHOLD = 0.05


def test_mot_metrics() -> None:
    """Tests that the new MOT metrics (on average) are at least 95% (configurable thropugh
    MOTA_ERROR_THRESHOLD) as good as the previous metrics. Otherwise the test will fail.

    Raises:
        If the previous metrics file its not found.
    """
    # Load previous metrics
    try:
        previous_metrics = pd.read_fwf("metrics.txt")
        previous_metrics.columns = [column_name.lower() for column_name in previous_metrics.columns]
        previous_metrics = previous_metrics.set_index(previous_metrics.columns[0])
    except FileNotFoundError as e:
        raise e

    accumulator = metrics.Accumulators()
    sequences_paths = [element.path for element in os.scandir(DATASET_PATH) if element.is_dir()]
    for input_path in sequences_paths:
        # Search vertical resolution in seqinfo.ini
        seqinfo_path = os.path.join(input_path, "seqinfo.ini")
        info_file = metrics.InformationFile(file_path=seqinfo_path)

        all_detections = metrics.DetectionFileParser(input_path=input_path, information_file=info_file)

        tracker = Tracker(
            distance_function=metrics.mot_keypoints_distance,
            distance_threshold=metrics.MotParameters.DISTANCE_THRESHOLD,
            detection_threshold=metrics.MotParameters.DETECTION_THRESHOLD,
            hit_inertia_min=10,
            hit_inertia_max=12,
            point_transience=4,
        )

        # Initialize accumulator for this video
        accumulator.create_accumulator(input_path=input_path, information_file=info_file)

        for frame_number, detections in enumerate(all_detections):
            if frame_number % metrics.MotParameters.FRAME_SKIP_PERIOD == 0:
                tracked_objects = tracker.update(
                    detections=detections, period=metrics.MotParameters.FRAME_SKIP_PERIOD
                )
            else:
                detections = []
                tracked_objects = tracker.update()

            accumulator.update(predictions=tracked_objects)

    accumulator.compute_metrics()
    new_metrics = accumulator.summary_dataframe
    new_metrics.columns = [column_name.lower() for column_name in new_metrics.columns]

    # Unify the scores to be able to compare them. new metrics is the percentage
    # expressed between 0 and 1, the previous metrics have the percentage as a string
    # with the % character at the end
    new_overall_mota = new_metrics.loc["OVERALL", "mota"] * 100
    previous_overall_mota = float(previous_metrics.loc["OVERALL", "mota"][:-1])

    assert new_overall_mota >= previous_overall_mota * (1 - MOTA_ERROR_THRESHOLD), (
        f"New overall MOTA score: {new_overall_mota} is too low, previous overall MOTA score: "
        f"{previous_overall_mota}"
    )
